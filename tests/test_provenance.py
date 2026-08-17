"""Reproducibility metadata must be complete, stable, and honest about gaps."""

from __future__ import annotations

import json
import subprocess

import pytest

from grove.provenance import (
    PROVENANCE_SCHEMA,
    REQUIRED_MODELS,
    REQUIRED_SECTIONS,
    canonical_hash,
    collect_provenance,
    directory_digest,
    file_digest,
    git_revision,
    missing_fields,
    verifier_suite_digest,
    worker_metadata,
)


def test_canonical_hash_ignores_key_order_but_not_content():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_file_digest_marks_missing_files_instead_of_omitting_them(tmp_path):
    present = tmp_path / "present.txt"
    present.write_text("grove")

    assert len(file_digest(present)) == 64
    assert file_digest(present) == file_digest(present)
    assert file_digest(tmp_path / "absent.txt").startswith("unavailable:")


def test_directory_digest_detects_a_changed_model_file(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"\x00\x01")
    (model / "config.json").write_text("{}")

    first = directory_digest(model)
    unchanged = directory_digest(model)
    (model / "weights.bin").write_bytes(b"\x00\x02")
    changed = directory_digest(model)

    assert first["file_count"] == 2
    assert first["aggregate_sha256"] == unchanged["aggregate_sha256"]
    assert changed["aggregate_sha256"] != first["aggregate_sha256"]
    assert set(first["files"]) == {"weights.bin", "config.json"}


def test_directory_digest_reports_a_missing_model_directory(tmp_path):
    record = directory_digest(tmp_path / "not-there")

    assert record["aggregate_sha256"].startswith("unavailable:")


def test_directory_digest_marks_an_empty_model_directory_as_a_gap(tmp_path):
    model = tmp_path / "empty-model"
    model.mkdir()

    record = directory_digest(model)

    assert record["path"] == str(model)
    assert record["file_count"] == 0
    assert record["files"] == {}
    assert record["aggregate_sha256"].startswith("unavailable:")
    assert "models.base.aggregate_sha256" in missing_fields(
        {"models": {"base": record}}
    )


def test_directory_digest_marks_no_matching_files_as_a_gap(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "README.txt").write_text("not a weight file")

    record = directory_digest(model, patterns=("*.safetensors",))

    assert record["file_count"] == 0
    assert record["aggregate_sha256"].startswith("unavailable:")


def test_git_revision_status_failure_is_a_provenance_gap(tmp_path, monkeypatch):
    def fake_run(arguments, **_kwargs):
        class Completed:
            def __init__(self, returncode, stdout):
                self.returncode = returncode
                self.stdout = stdout

        command = tuple(arguments[-2:])
        if command == ("rev-parse", "HEAD"):
            return Completed(0, "a" * 40 + "\n")
        if command == ("status", "--porcelain"):
            return Completed(128, "")
        if command == ("rev-parse", "HEAD^{tree}"):
            return Completed(0, "b" * 40 + "\n")
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_run)

    record = git_revision(tmp_path)

    assert record["revision"] == "a" * 40
    assert record["dirty"].startswith("unavailable:")
    assert record["status_sha256"].startswith("unavailable:")
    assert "source.dirty" in missing_fields({"source": record})


def test_verifier_suite_digest_changes_when_hidden_cases_change():
    stable = verifier_suite_digest("v1", [{"task": "a", "cases": 3}])
    edited = verifier_suite_digest("v1", [{"task": "a", "cases": 4}])

    # The version label is identical; the digest is what exposes the edit.
    assert stable["suite_version"] == edited["suite_version"] == "v1"
    assert stable["suites_sha256"] != edited["suites_sha256"]


def test_git_revision_records_commit_and_dirty_state(tmp_path):
    probe = subprocess.run(["git", "--version"], capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("git unavailable")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.txt").write_text("one")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "one"], check=True)

    clean = git_revision(tmp_path)
    (tmp_path / "a.txt").write_text("two")
    dirty = git_revision(tmp_path)

    assert len(clean["revision"]) == 40
    assert clean["dirty"] is False
    assert dirty["dirty"] is True
    assert dirty["status_sha256"] != clean["status_sha256"]


def test_git_revision_on_a_non_repository_is_marked_unavailable(tmp_path):
    record = git_revision(tmp_path / "nowhere")

    assert record["revision"].startswith("unavailable:")


def test_collect_provenance_is_stable_and_sensitive_to_training_config(tmp_path):
    (tmp_path / "uv.lock").write_text("lock")
    (tmp_path / "pyproject.toml").write_text("[project]")
    arguments = {
        "repo_root": tmp_path,
        "base_model": "model@abc",
        "verifier_suite_version": "suite-v1",
        "decoding_config": {"temperature": 0.0},
    }

    first = collect_provenance(**arguments, training_config={"iters": 200})
    repeat = collect_provenance(**arguments, training_config={"iters": 200})
    other = collect_provenance(**arguments, training_config={"iters": 400})

    assert first["schema"] == PROVENANCE_SCHEMA
    assert first["provenance_sha256"] == repeat["provenance_sha256"]
    assert first["provenance_sha256"] != other["provenance_sha256"]
    assert first["training_config_sha256"] != other["training_config_sha256"]
    assert len(first["lockfile_sha256"]) == 64


def test_missing_fields_lists_every_unresolved_collector(tmp_path):
    record = collect_provenance(
        repo_root=tmp_path,
        base_model="model@abc",
        verifier_suite_version="suite-v1",
        model_paths={"base": tmp_path / "absent-model"},
    )

    gaps = missing_fields(record)

    assert "lockfile_sha256" in gaps
    assert "pyproject_sha256" in gaps
    assert "models.base.aggregate_sha256" in gaps
    assert "sandbox_image.fingerprint" in gaps


def test_provenance_record_carries_no_environment_or_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GROVE_TEST_SECRET", "super-secret-token-value")

    record = collect_provenance(
        repo_root=tmp_path, base_model="model@abc", verifier_suite_version="suite-v1"
    )
    serialized = json.dumps(record, default=str)

    assert "super-secret-token-value" not in serialized
    assert "GROVE_TEST_SECRET" not in serialized
    assert not {"env", "environ", "environment"} & set(record)


# --------------------------------------------------------------------------
# Required sections: an omitted field must never read as a complete record.
# --------------------------------------------------------------------------


def test_every_required_section_is_emitted_even_when_unresolvable(tmp_path):
    record = collect_provenance(
        repo_root=tmp_path, base_model="model@abc", verifier_suite_version="suite-v1"
    )

    for section in REQUIRED_SECTIONS:
        assert section in record, section
    for name in REQUIRED_MODELS:
        assert name in record["models"], name
    assert record["models"]["base"]["aggregate_sha256"].startswith("unavailable:")
    assert record["worker"]["status"].startswith("unavailable:")


def test_absent_model_and_worker_data_become_gaps(tmp_path):
    record = collect_provenance(
        repo_root=tmp_path, base_model="model@abc", verifier_suite_version="suite-v1"
    )

    gaps = missing_fields(record)

    assert "models.base.aggregate_sha256" in gaps
    assert "worker.status" in gaps
    assert "worker.host" in gaps


def test_missing_fields_flags_a_deleted_required_section():
    """A record that simply drops a section is not a record without gaps."""
    stripped = {"schema": PROVENANCE_SCHEMA}

    gaps = missing_fields(stripped)

    assert "models" in gaps
    assert "models.base" in gaps
    assert "worker" in gaps
    assert "source" in gaps


def test_missing_fields_flags_an_empty_models_record():
    gaps = missing_fields({"models": {}, "worker": {"status": "ok"}})

    assert "models" in gaps
    assert "models.base" in gaps
    assert "worker" not in gaps


def test_worker_metadata_is_unavailable_until_it_is_collected():
    record = worker_metadata()

    assert record["status"].startswith("unavailable:")
    assert record["host"].startswith("unavailable:")


def test_worker_metadata_records_a_failed_preflight_instead_of_raising():
    def explode():
        raise TimeoutError("ssh timed out")

    record = worker_metadata(explode, host="worker-host")

    assert record["host"] == "worker-host"
    assert record["status"] == "unavailable: worker preflight failed: TimeoutError"
    assert "worker.status" in missing_fields({"worker": record})


def test_worker_metadata_keeps_versions_and_drops_everything_else():
    def preflight():
        return {
            "status": "ok",
            "machine": "arm64",
            "python": "3.14.3",
            "mlx": "0.32.0",
            "mlx_lm": "0.31.3",
            "identity_file": "/home/user/.ssh/secret_key",
            "token": "super-secret-token-value",
        }

    record = worker_metadata(preflight, host="grove-worker-1")

    assert record["host"] == "grove-worker-1"
    assert record["mlx"] == "0.32.0"
    assert record["mlx_lm"] == "0.31.3"
    assert "identity_file" not in record
    assert "super-secret-token-value" not in json.dumps(record)


def test_a_reachable_worker_still_gaps_the_remote_model_digest():
    """A framework version is not a model-file hash, and must not stand in."""

    def preflight():
        return {"status": "ok", "machine": "arm64", "mlx": "0.32.0"}

    record = worker_metadata(preflight, host="worker")

    assert record["status"] == "ok"
    assert record["model_manifest_sha256"].startswith("unavailable:")
    assert "worker.model_manifest_sha256" in missing_fields({"worker": record})


def test_a_worker_supplied_manifest_closes_the_gap():
    def preflight():
        return {"status": "ok", "model_manifest_sha256": "a" * 64}

    record = worker_metadata(preflight, host="worker")

    assert record["model_manifest_sha256"] == "a" * 64
    # The model digest is closed, but the worker still reports no checkout and
    # no framework versions, and those stay named as gaps rather than absent.
    assert missing_fields({"worker": record}) == [
        "models",
        "models.base",
        "sandbox_image",
        "source",
        "verifiers",
        "worker.checkout.dirty",
        "worker.checkout.revision",
        "worker.checkout.status_sha256",
        "worker.checkout.tree",
        "worker.checkout.worktree_sha256",
        "worker.framework_versions_sha256",
    ]


def test_null_worker_manifest_becomes_an_explicit_gap():
    """Finding 13: ``setdefault`` left an explicit null in place.

    A null read as collected-and-empty, so ``missing_fields`` saw nothing wrong
    and the arm-pairing check treated two nulls as a match.
    """

    def preflight():
        return {"status": "ok", "model_manifest_sha256": None, "mlx": None}

    record = worker_metadata(preflight, host="worker")

    assert record["model_manifest_sha256"].startswith("unavailable:")
    assert record["mlx"].startswith("unavailable:")
    gaps = missing_fields({"worker": record})
    assert "worker.model_manifest_sha256" in gaps
    assert "worker.mlx" in gaps


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_worker_identity_becomes_an_explicit_gap(blank):
    def preflight():
        return {
            "status": blank,
            "mlx": blank,
            "model_manifest_sha256": blank,
            "checkout": {"revision": blank},
        }

    record = worker_metadata(preflight, host=blank)

    assert record["host"].startswith("unavailable:")
    assert record["status"].startswith("unavailable:")
    assert record["mlx"].startswith("unavailable:")
    assert record["model_manifest_sha256"].startswith("unavailable:")
    assert record["checkout"]["revision"].startswith("unavailable:")


def test_missing_fields_marks_blank_nested_identity():
    gaps = missing_fields({"worker": {"checkout": {"revision": "   "}}})

    assert "worker.checkout.revision" in gaps


def test_missing_fields_marks_null_nested_identity():
    gaps = missing_fields({"worker": {"checkout": {"revision": None}}})

    assert "worker.checkout.revision" in gaps


def test_worker_checkout_identity_is_recorded():
    """A worker that reports its checkout closes the source-identity gap."""

    def preflight():
        return {
            "status": "ok",
            "python": "3.14.3",
            "mlx": "0.32.0",
            "mlx_lm": "0.31.3",
            "checkout": {
                "revision": "1" * 40,
                "tree": "2" * 40,
                "dirty": False,
                "status_sha256": "3" * 64,
                "worktree_sha256": "4" * 64,
            },
        }

    record = worker_metadata(preflight, host="worker")

    assert record["checkout"]["revision"] == "1" * 40
    assert record["checkout"]["dirty"] is False
    assert len(record["framework_versions_sha256"]) == 64
    gaps = missing_fields({"worker": record})
    # The worker still returns no model manifest; that gap is declared in both
    # shipped specs and must stay visible rather than disappear.
    assert [gap for gap in gaps if gap.startswith("worker.")] == [
        "worker.model_manifest_sha256"
    ]


def test_missing_worker_checkout_identity_is_a_gap():
    """Framework versions describe libraries, never which code ran."""

    def preflight():
        return {"status": "ok", "python": "3.14.3", "mlx": "0.32.0"}

    record = worker_metadata(preflight, host="worker")

    gaps = missing_fields({"worker": record})
    assert "worker.checkout.revision" in gaps
    assert "worker.checkout.worktree_sha256" in gaps
    # mlx_lm is absent, so the framework digest must not resolve either.
    assert record["framework_versions_sha256"].startswith("unavailable:")
    assert "worker.framework_versions_sha256" in gaps


def test_collected_worker_metadata_reaches_the_record(tmp_path):
    record = collect_provenance(
        repo_root=tmp_path,
        base_model="model@abc",
        verifier_suite_version="suite-v1",
        worker={"host": "worker", "status": "ok", "mlx": "0.32.0"},
    )

    assert record["worker"]["mlx"] == "0.32.0"
    assert "worker.status" not in missing_fields(record)


def test_collect_provenance_completes_caller_worker_mapping_gaps(tmp_path):
    record = collect_provenance(
        repo_root=tmp_path,
        base_model="model@abc",
        verifier_suite_version="suite-v1",
        worker={"host": "worker", "status": "ok", "mlx": "0.32.0"},
    )

    worker = record["worker"]
    assert worker["host"] == "worker"
    assert worker["status"] == "ok"
    assert worker["checkout"]["revision"].startswith("unavailable:")
    assert worker["model_manifest_sha256"].startswith("unavailable:")
    assert worker["framework_versions_sha256"].startswith("unavailable:")
    gaps = missing_fields(record)
    assert "worker.checkout.revision" in gaps
    assert "worker.model_manifest_sha256" in gaps
    assert "worker.framework_versions_sha256" in gaps

# --------------------------------------------------------------------------
# Finding 2: the host worktree digest hashes bytes, not names
# --------------------------------------------------------------------------


def _commit_repository(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()

    def git(*arguments):
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (root / "tracked.txt").write_text("committed\n")
    git("add", "tracked.txt")
    git("commit", "-qm", "initial")
    return root, git


def test_worktree_digest_changes_when_untracked_bytes_change(tmp_path):
    from grove.provenance import git_runner, worktree_digest

    root, _ = _commit_repository(tmp_path)
    note = root / "note.txt"
    git = git_runner(root)

    note.write_text("one")
    first = worktree_digest(git, root)
    note.write_text("two")
    second = worktree_digest(git, root)

    assert first != second


def test_worktree_digest_records_the_executable_bit(tmp_path):
    from grove.provenance import git_runner, worktree_digest

    root, _ = _commit_repository(tmp_path)
    script = root / "run.sh"
    script.write_text("#!/bin/sh\n")
    git = git_runner(root)

    plain = worktree_digest(git, root)
    script.chmod(0o755)
    executable = worktree_digest(git, root)

    assert plain != executable


def test_worktree_digest_is_stable_for_an_unchanged_tree(tmp_path):
    from grove.provenance import git_runner, worktree_digest

    root, _ = _commit_repository(tmp_path)
    git = git_runner(root)

    assert worktree_digest(git, root) == worktree_digest(git, root)


def test_worktree_digest_records_a_rename_with_both_paths(tmp_path):
    from grove.provenance import git_runner, worktree_entries

    root, git_command = _commit_repository(tmp_path)
    git_command("mv", "tracked.txt", "renamed.txt")

    entries = worktree_entries(git_runner(root), root)

    renamed = [item for item in entries if item["path"] == "renamed.txt"]
    assert renamed, entries
    assert renamed[0]["origin_path"] == "tracked.txt"


def test_worktree_digest_names_its_ignored_file_policy():
    """Scope stated, not silently omitted."""
    from grove.provenance import WORKTREE_IGNORED_POLICY

    assert WORKTREE_IGNORED_POLICY == "ignored-paths-excluded"


def test_worktree_digest_marks_a_failed_git_status(tmp_path):
    from grove.provenance import worktree_digest

    assert worktree_digest(lambda *a, **k: None, tmp_path).startswith("unavailable:")


# --------------------------------------------------------------------------
# Reviewer B blocker 1: a non-digest is not a model identity.
# --------------------------------------------------------------------------

VALID_DIGEST = "a" * 64
NON_DIGEST_IDENTITIES = [
    False,
    True,
    0,
    1,
    {},
    {"sha256": VALID_DIGEST},
    [],
    [VALID_DIGEST],
    "",
    "   ",
    "not-a-digest",
    "A" * 64,
    "a" * 63,
    "a" * 65,
    f"sha256:{VALID_DIGEST}",
    "g" * 64,
    f" {VALID_DIGEST} ",
]


@pytest.mark.parametrize("value", NON_DIGEST_IDENTITIES)
def test_a_non_digest_model_manifest_is_a_named_gap(value):
    """``model_manifest_sha256: false`` is not reported model identity.

    The old rule rejected only null and blank, so a worker could answer
    ``false``, ``0`` or ``{}`` and have it counted as a resolved digest -- and
    two arms answering the same malformed value compared equal.
    """
    record = worker_metadata(
        lambda: {
            "status": "ok",
            "python": "3.12.0",
            "mlx": "0.18.0",
            "mlx_lm": "0.19.0",
            "model_manifest_sha256": value,
        },
        host="worker-host",
    )

    assert isinstance(record["model_manifest_sha256"], str)
    assert record["model_manifest_sha256"].startswith("unavailable:")


def test_a_real_digest_still_resolves():
    record = worker_metadata(
        lambda: {
            "status": "ok",
            "python": "3.12.0",
            "mlx": "0.18.0",
            "mlx_lm": "0.19.0",
            "model_manifest_sha256": VALID_DIGEST,
        },
        host="worker-host",
    )

    assert record["model_manifest_sha256"] == VALID_DIGEST


@pytest.mark.parametrize("value", NON_DIGEST_IDENTITIES)
def test_a_non_digest_worker_checkout_digest_is_a_named_gap(value):
    from grove.provenance import worker_checkout

    checkout = worker_checkout(
        {
            "revision": "9" * 40,
            "tree": "8" * 40,
            "dirty": False,
            "status_sha256": value,
            "worktree_sha256": value,
        }
    )

    assert checkout["status_sha256"].startswith("unavailable:")
    assert checkout["worktree_sha256"].startswith("unavailable:")
    # Non-digest fields are untouched: this rule is about digests only.
    assert checkout["revision"] == "9" * 40
    assert checkout["dirty"] is False


@pytest.mark.parametrize("value", NON_DIGEST_IDENTITIES)
def test_missing_fields_names_the_root_digest_not_a_nested_path(value, tmp_path):
    """A malformed mapping must not produce nested paths under a fake digest."""
    record = collect_provenance(
        repo_root=tmp_path,
        base_model="model@abc",
        verifier_suite_version="suite-v1",
        worker={
            "host": "worker-host",
            "status": "ok",
            "python": "3.12.0",
            "mlx": "0.18.0",
            "mlx_lm": "0.19.0",
            "model_manifest_sha256": value,
        },
    )

    gaps = missing_fields(record)
    assert "worker.model_manifest_sha256" in gaps
    assert not any(
        gap.startswith("worker.model_manifest_sha256.") for gap in gaps
    )


def test_missing_fields_accepts_a_real_worker_digest(tmp_path):
    record = collect_provenance(
        repo_root=tmp_path,
        base_model="model@abc",
        verifier_suite_version="suite-v1",
        worker={
            "host": "worker-host",
            "status": "ok",
            "python": "3.12.0",
            "mlx": "0.18.0",
            "mlx_lm": "0.19.0",
            "model_manifest_sha256": VALID_DIGEST,
        },
    )

    assert "worker.model_manifest_sha256" not in missing_fields(record)


def test_is_sha256_hex_accepts_only_64_lowercase_hex_characters():
    from grove.provenance import is_sha256_hex

    assert is_sha256_hex(VALID_DIGEST) is True
    assert is_sha256_hex("0123456789abcdef" * 4) is True
    for value in NON_DIGEST_IDENTITIES:
        assert is_sha256_hex(value) is False, value


def test_an_existing_gap_reason_is_not_rewritten_as_a_digest_error():
    """The worker's own reason survives; it is more useful than the format."""
    record = worker_metadata(
        lambda: {
            "status": "ok",
            "python": "3.12.0",
            "mlx": "0.18.0",
            "mlx_lm": "0.19.0",
            "model_manifest": {"path": "/models/base", "reason": "path not found"},
        },
        host="worker-host",
    )

    assert record["model_manifest_sha256"] == (
        "unavailable: worker does not return model file digests"
    )
