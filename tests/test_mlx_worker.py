"""Worker-side identity: what the worker says about its own code and model.

The control host cannot see either. These tests pin the two ways that
self-report used to be uninformative: a content digest that ignored untracked
and binary bytes, and a model manifest that did not exist at all.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from grove.mlx_worker import _worker_checkout, model_manifest_digest
from grove.provenance import git_runner, worktree_digest


def _repository(tmp_path):
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
    return root


def test_worker_worktree_hash_changes_when_untracked_bytes_change(tmp_path):
    """Finding 2: the old digest hashed ``git diff HEAD`` and nothing else.

    An untracked file's *name* appeared in the status digest, so adding one was
    visible. Changing its contents was not: both digests stayed identical.
    """
    root = _repository(tmp_path)
    untracked = root / "note.txt"

    untracked.write_text("one")
    first = _worker_checkout(root)

    untracked.write_text("two")
    second = _worker_checkout(root)

    assert first["status_sha256"] == second["status_sha256"]
    assert first["worktree_sha256"] != second["worktree_sha256"]


def test_worker_worktree_hash_changes_when_binary_bytes_change(tmp_path):
    """A binary diff is elided by git, so a text diff could not have caught it."""
    root = _repository(tmp_path)
    binary = root / "weights.bin"

    binary.write_bytes(b"\x00\x01\x02")
    first = _worker_checkout(root)

    binary.write_bytes(b"\x00\x01\x03")
    second = _worker_checkout(root)

    assert first["worktree_sha256"] != second["worktree_sha256"]


def test_worker_worktree_hash_changes_when_a_tracked_file_changes(tmp_path):
    root = _repository(tmp_path)
    clean = _worker_checkout(root)

    (root / "tracked.txt").write_text("edited\n")
    dirty = _worker_checkout(root)

    assert clean["dirty"] is False
    assert dirty["dirty"] is True
    assert clean["worktree_sha256"] != dirty["worktree_sha256"]


def test_worker_worktree_hash_records_a_deletion(tmp_path):
    root = _repository(tmp_path)
    clean = _worker_checkout(root)

    (root / "tracked.txt").unlink()
    deleted = _worker_checkout(root)

    assert clean["worktree_sha256"] != deleted["worktree_sha256"]


def test_worker_worktree_hash_survives_awkward_paths(tmp_path):
    """NUL-delimited status output, so a space or quote cannot split a record."""
    root = _repository(tmp_path)
    (root / "a file with spaces.txt").write_text("x")
    (root / 'quoted"name.txt').write_text("y")

    first = _worker_checkout(root)
    (root / 'quoted"name.txt').write_text("z")
    second = _worker_checkout(root)

    assert first["worktree_sha256"] != second["worktree_sha256"]


def test_worker_and_host_use_the_same_worktree_digest(tmp_path):
    """One helper, so a host digest and a worker digest are comparable."""
    root = _repository(tmp_path)
    (root / "note.txt").write_text("shared")

    assert _worker_checkout(root)["worktree_sha256"] == worktree_digest(
        git_runner(root), root
    )


def test_worker_checkout_of_a_non_repository_is_empty(tmp_path):
    assert _worker_checkout(tmp_path) == {}


def test_worker_model_manifest_is_normalized_when_missing_or_null(tmp_path):
    """An absent or empty model directory is a named gap, never a digest.

    Hashing zero files produces a perfectly stable value that means nothing.
    """
    assert model_manifest_digest(None)["sha256"] is None
    assert model_manifest_digest(None)["reason"] == "no model path requested"

    absent = model_manifest_digest(tmp_path / "nowhere")
    assert absent["sha256"] is None
    assert absent["reason"] == "not a directory"

    empty = tmp_path / "empty"
    empty.mkdir()
    blank = model_manifest_digest(empty)
    assert blank["sha256"] is None
    assert blank["reason"] == "no model files found"


def test_worker_model_manifest_covers_file_bytes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"\x00\x01")

    first = model_manifest_digest(model)
    (model / "weights.safetensors").write_bytes(b"\x00\x02")
    second = model_manifest_digest(model)

    assert first["file_count"] == 2
    assert first["sha256"] is not None
    assert first["sha256"] != second["sha256"]


def test_worker_model_manifest_reaches_provenance(tmp_path):
    from grove.provenance import missing_fields, worker_metadata

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    manifest = model_manifest_digest(model)

    record = worker_metadata(
        lambda: {
            "status": "ok",
            "python": "3.14.3",
            "mlx": "0.32.0",
            "mlx_lm": "0.31.3",
            "model_manifest": manifest,
            "model_manifest_sha256": manifest["sha256"],
        },
        host="worker",
    )

    assert record["model_manifest_sha256"] == manifest["sha256"]
    assert "worker.model_manifest_sha256" not in missing_fields({"worker": record})


def test_a_refused_model_manifest_becomes_a_named_gap():
    from grove.provenance import missing_fields, worker_metadata

    record = worker_metadata(
        lambda: {
            "status": "ok",
            "model_manifest": {
                "path": "/Users/grove-worker/model",
                "sha256": None,
                "reason": "no model files found",
            },
            "model_manifest_sha256": None,
        },
        host="worker",
    )

    assert record["model_manifest_sha256"].startswith("unavailable:")
    # The worker's own reason survives into the gap text.
    assert "no model files found" in record["model_manifest_sha256"]
    assert "worker.model_manifest_sha256" in missing_fields({"worker": record})


@pytest.mark.parametrize("argv", [["preflight"], ["preflight", "--model", "/tmp"]])
def test_preflight_command_accepts_an_optional_model_path(argv):
    """The remote call path has to be able to carry the model path."""
    from grove.mlx_worker import build_parser

    args = build_parser().parse_args(argv)

    assert args.command == "preflight"
    assert args.model == (None if len(argv) == 1 else "/tmp")

def test_ssh_worker_passes_model_and_repository_paths_to_remote_command():
    from grove.remote import MlxSshWorker, SshWorkerConfig

    sent: list[str] = []

    class Recorder(MlxSshWorker):
        def _ssh(self, command, *, input_text=None):
            sent.append(command)
            return subprocess.CompletedProcess([], 0, stdout="{}", stderr="")

    model_path = "/Users/grove-worker/grove/cache/model dir"
    repository = "/Users/grove-worker/grove/runtime/repo with spaces"
    Recorder(SshWorkerConfig(remote_repository=repository)).preflight(model_path)

    assert "mlx_worker preflight --model" in sent[0]
    assert f"--repository '{repository}'" in sent[0]
    # Paths with spaces must survive the shell.
    assert f"--model '{model_path}'" in sent[0]

def test_preflight_hashes_the_configured_repository(monkeypatch, tmp_path):
    from grove import mlx_worker

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.default_device = lambda: "cpu"
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setattr(mlx_worker.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(
        mlx_worker.os,
        "statvfs",
        lambda _: types.SimpleNamespace(f_bavail=1, f_frsize=1),
    )
    seen: list[object] = []
    monkeypatch.setattr(
        mlx_worker,
        "_worker_checkout",
        lambda root: seen.append(root) or {},
    )
    repository = tmp_path / "configured-repository"

    result = mlx_worker.preflight(repository=repository)

    assert result["checkout"] == {}
    assert seen == [repository]


def test_preflight_parser_accepts_a_configured_repository(tmp_path):
    from grove.mlx_worker import build_parser

    args = build_parser().parse_args(
        ["preflight", "--repository", str(tmp_path / "repo")]
    )

    assert args.repository == str(tmp_path / "repo")


# --------------------------------------------------------------------------
# EXP-004: per-request decoding must be honored, echoed, and refusable
# --------------------------------------------------------------------------


def _install_fake_inference(monkeypatch, *, seedable: bool = True) -> dict:
    """A worker-side MLX stack that records seeding and sampler construction.

    Real seeding support was verified on the worker (mx.random.seed exists and
    deterministically drives make_sampler's categorical draws); these fakes pin
    the worker *protocol* around it without Apple hardware.
    """
    calls: dict = {"seeds": [], "sampler_temps": [], "generate": []}
    fake_core = types.ModuleType("mlx.core")
    random_namespace = types.SimpleNamespace()
    if seedable:
        random_namespace.seed = lambda value: calls["seeds"].append(value)
    fake_core.random = random_namespace
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core

    class _Tokenizer:
        def add_eos_token(self, token):
            pass

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return messages[-1]["content"]

    fake_lm = types.ModuleType("mlx_lm")
    fake_lm.load = lambda model_id, adapter_path=None: ("model", _Tokenizer())

    def generate(model, tokenizer, *, prompt, max_tokens, sampler, verbose):
        calls["generate"].append(
            {"prompt": prompt, "max_tokens": max_tokens, "sampler": sampler}
        )
        return f"out:{prompt}"

    fake_lm.generate = generate
    fake_sample = types.ModuleType("mlx_lm.sample_utils")

    def make_sampler(temp=0.0):
        calls["sampler_temps"].append(temp)
        return ("sampler", temp)

    fake_sample.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample)
    return calls


def test_infer_honors_and_echoes_per_request_temperature_and_seed(monkeypatch):
    calls = _install_fake_inference(monkeypatch)
    from grove.mlx_worker import infer

    result = infer(
        {
            "model": "m",
            "temperature": 0.0,
            "max_tokens": 64,
            "requests": [
                {"id": "greedy", "prompt": "a"},
                {"id": "sampled", "prompt": "b", "temperature": 0.8, "seed": 1234},
            ],
        }
    )

    greedy, sampled = result["outputs"]
    # Echo, not assumption: the caller verifies what actually ran.
    assert greedy == {"id": "greedy", "text": "out:a", "temperature": 0.0, "seed": None}
    assert sampled["temperature"] == 0.8
    assert sampled["seed"] == 1234
    # The PRNG was seeded exactly once, for the request that asked.
    assert calls["seeds"] == [1234]
    # One sampler per distinct temperature, greedy and sampled.
    assert sorted(calls["sampler_temps"]) == [0.0, 0.8]


def test_infer_supports_per_request_max_tokens(monkeypatch):
    calls = _install_fake_inference(monkeypatch)
    from grove.mlx_worker import infer

    infer(
        {
            "model": "m",
            "max_tokens": 64,
            "requests": [{"id": "x", "prompt": "p", "max_tokens": 7}],
        }
    )

    assert calls["generate"][0]["max_tokens"] == 7


def test_infer_refuses_a_seed_the_runtime_cannot_honor(monkeypatch):
    """A silently unseeded 'seeded' run would fake reproducibility."""
    _install_fake_inference(monkeypatch, seedable=False)
    from grove.mlx_worker import infer

    with pytest.raises(RuntimeError, match="mx.random.seed"):
        infer({"model": "m", "requests": [{"id": "x", "prompt": "p", "seed": 7}]})
