"""Reproducibility metadata for a Grove run.

The 2026-08-06 audit rated reproducibility "above average but incomplete": the
experiment recorded adapter and dataset hashes but not the surrounding world.
This module collects the missing pieces -- source revision, lockfile hash, model
file hashes, sandbox image fingerprint, training config hash and verifier suite
hash -- into one canonically hashable record that can be embedded in a report
and compared across runs.

Every collector is best-effort and records ``"unavailable: <reason>"`` instead of
raising or silently omitting a field, because a missing field that looks absent
is exactly how an unreproducible run passes for a reproducible one. Only paths,
versions and digests are recorded; no environment dump, no credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA = "grove-provenance-v2"
# Sections that must appear in every record. An omitted section is a gap, not
# an absence: silence is how an unreproducible run passes for a complete one.
REQUIRED_SECTIONS = ("source", "models", "worker", "verifiers", "sandbox_image")
REQUIRED_MODELS = ("base",)

# Grove stores raw SHA-256 hex digests, not OCI ``sha256:``-prefixed strings:
# ``canonical_hash`` and every sealed digest field are 64 lowercase hex
# characters. A value that is not that is not an identity, however truthy it
# looks. ``False``, ``0``, ``{}`` and ``"unknown"`` all used to pass for one.
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def is_sha256_hex(value: Any) -> bool:
    """True only for exactly 64 lowercase hexadecimal characters."""
    return isinstance(value, str) and SHA256_HEX.fullmatch(value) is not None


def is_digest_field(path: str | None) -> bool:
    """True when a dotted path's final component names a raw digest."""
    if not path:
        return False
    return path.rsplit(".", 1)[-1].endswith("_sha256")


def _unavailable(reason: str) -> str:
    return f"unavailable: {reason}"


# Grove canonical JSON v1. RFC 8785 (JCS) is the reference for deterministic
# JSON; a full JCS implementation would additionally pin ECMAScript number
# formatting, which needs a dependency this repository does not carry. What is
# fixed here instead is documented exactly, and pinned by tests:
#
#   * UTF-8 output, no ASCII escaping, so the same text hashes the same way;
#   * object keys sorted by Unicode code point;
#   * array order preserved;
#   * no insignificant whitespace;
#   * NaN and the infinities rejected, as I-JSON requires;
#   * any value that is not a JSON primitive, list or string-keyed object
#     rejected outright.
#
# The last rule matters most. The old encoder passed ``default=str``, so a
# value JSON cannot represent was silently replaced by its ``repr`` and hashed
# as if it were data. A digest must commit to what the document actually says,
# or it commits to nothing.
CANONICAL_JSON_SCHEMA = "grove-canonical-json-v1"


class UnrepresentableValue(TypeError):
    """A value a canonical digest must not silently stringify."""


def _reject_unrepresentable(payload: Any, path: str = "") -> None:
    where = path or "<root>"
    if payload is None or isinstance(payload, str | bool | int):
        return
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise UnrepresentableValue(
                f"{where}: NaN and infinity are not JSON"
            )
        return
    if isinstance(payload, list | tuple):
        for index, item in enumerate(payload):
            _reject_unrepresentable(item, f"{path}[{index}]")
        return
    if isinstance(payload, Mapping):
        for key, item in payload.items():
            if not isinstance(key, str):
                raise UnrepresentableValue(
                    f"{where}: object keys must be strings, got {type(key).__name__}"
                )
            _reject_unrepresentable(item, f"{path}.{key}" if path else key)
        return
    raise UnrepresentableValue(
        f"{where}: {type(payload).__name__} has no JSON representation; "
        "convert it explicitly rather than letting a digest hash its repr"
    )


def canonical_json(payload: Any) -> bytes:
    """Deterministic UTF-8 bytes for a JSON-representable payload."""
    _reject_unrepresentable(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    """Stable SHA-256 over a JSON-representable payload.

    Every commitment in the repository -- spec seals, provenance digests, run
    manifests, decision-input bindings -- goes through this one encoder, so a
    digest computed by the library and a digest computed by the standalone
    checker cannot drift apart.
    """
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_digest(path: str | Path) -> str:
    path = Path(path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        return _unavailable(f"{type(error).__name__} reading {path}")


def directory_digest(path: str | Path, *, patterns: Iterable[str] = ("*",)) -> dict:
    """Per-file digests plus an aggregate for a model or adapter directory."""
    root = Path(path)
    if not root.is_dir():
        return {"path": str(root), "aggregate_sha256": _unavailable("not a directory")}
    selected: set[Path] = set()
    for pattern in patterns:
        selected.update(item for item in root.rglob(pattern) if item.is_file())
    files: dict[str, str] = {}
    total = 0
    for item in sorted(selected):
        relative = str(item.relative_to(root))
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        files[relative] = digest
        total += item.stat().st_size
    if not files:
        return {
            "path": str(root),
            "files": files,
            "file_count": 0,
            "size_bytes": total,
            "aggregate_sha256": _unavailable("no files matched digest patterns"),
        }
    aggregate = hashlib.sha256()
    for relative, digest in files.items():
        aggregate.update(relative.encode())
        aggregate.update(digest.encode())
    return {
        "path": str(root),
        "files": files,
        "file_count": len(files),
        "size_bytes": total,
        "aggregate_sha256": aggregate.hexdigest(),
    }


WORKTREE_DIGEST_SCHEMA = "grove-worktree-digest-v2"
# Ignored paths are deliberately excluded. Saying so here, and in the digest
# payload itself, is the difference between a documented scope and a silent
# omission: an ignored executable is invisible to this fingerprint.
WORKTREE_IGNORED_POLICY = "ignored-paths-excluded"


def git_runner(repo_root: str | Path) -> Any:
    """A ``git -C <root>`` caller that returns ``None`` instead of raising."""
    root = Path(repo_root)

    def _git(*arguments: str, strip: bool = True) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() if strip else completed.stdout

    return _git


def _path_content(target: Path) -> dict[str, Any]:
    """What a path currently holds, by bytes rather than by name."""
    try:
        if target.is_symlink():
            return {
                "kind": "symlink",
                "target_sha256": hashlib.sha256(
                    os.readlink(target).encode()
                ).hexdigest(),
            }
        if not target.exists():
            # A staged deletion, or a path git named that is already gone. The
            # tombstone keeps the absence in the digest instead of dropping it.
            return {"kind": "deleted"}
        if target.is_dir():
            # A submodule or a nested checkout. Its own commit identifies it;
            # recursing here would hash somebody else's repository.
            return {"kind": "directory"}
        return {
            "kind": "file",
            "mode": "100755" if target.stat().st_mode & 0o111 else "100644",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    except OSError as error:
        return {"kind": "unreadable", "reason": type(error).__name__}


def worktree_entries(git: Any, repo_root: str | Path) -> list[dict[str, Any]] | None:
    """Every path git reports as changed, with its current content digest.

    ``git status --porcelain -z --untracked-files=all`` is NUL-delimited, so a
    path containing a space, a quote or a newline survives intact. A rename
    emits the new path followed by the original one; both are kept, because
    losing either loses the change.
    """
    output = git("status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False)
    if output is None:
        return None
    root = Path(repo_root)
    fields = output.split("\0")
    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if len(field) < 4:
            continue
        code, path = field[:2], field[3:]
        origin = None
        if "R" in code or "C" in code:
            origin = fields[index] if index < len(fields) else None
            index += 1
        entry = {
            "path": path,
            "status": code,
            "content": _path_content(root / path),
        }
        if origin is not None:
            entry["origin_path"] = origin
            entry["origin_content"] = _path_content(root / origin)
        entries.append(entry)
    return sorted(entries, key=lambda item: (item["path"], item["status"]))


def worktree_digest(git: Any, repo_root: str | Path) -> str:
    """Content digest of everything the commit does not describe.

    ``git status --porcelain`` names the dirty paths; it does not say what is
    in them, and ``git diff HEAD`` describes tracked text while saying nothing
    about untracked or binary bytes. Two runs from the same commit with
    different uncommitted content shared a digest, so a paired experiment could
    call two arms identical when they were not.

    This hashes the bytes: modified tracked files, untracked files, symlink
    targets, executable bits, and a tombstone for each deleted path. Host and
    worker call the same function so the two digests are comparable.
    """
    entries = worktree_entries(git, repo_root)
    if entries is None:
        return _unavailable("git status failed")
    return canonical_hash(
        {
            "schema": WORKTREE_DIGEST_SCHEMA,
            "ignored_paths": WORKTREE_IGNORED_POLICY,
            "entries": entries,
        }
    )


def git_revision(repo_root: str | Path) -> dict:
    """Commit, dirty flag, tracked-tree hash and worktree content hash.

    ``dirty`` matters more than the commit: a run made from an edited worktree is
    not reproducible from the commit alone, so the status digest and a digest of
    the actual uncommitted content are recorded too.
    """
    root = Path(repo_root)
    _git = git_runner(root)

    revision = _git("rev-parse", "HEAD")
    if revision is None:
        return {
            "revision": _unavailable("git rev-parse failed"),
            "dirty": _unavailable("git rev-parse failed"),
            "status_sha256": _unavailable("git status not run"),
            "worktree_sha256": _unavailable("git status not run"),
        }
    status = _git("status", "--porcelain")
    tree = _git("rev-parse", "HEAD^{tree}") or _unavailable("tree unavailable")
    if status is None:
        return {
            "revision": revision,
            "dirty": _unavailable("git status failed"),
            "status_sha256": _unavailable("git status failed"),
            "worktree_sha256": _unavailable("git status failed"),
            "tree": tree,
        }
    return {
        "revision": revision,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "worktree_sha256": worktree_digest(_git, root),
        "tree": tree,
    }


def sandbox_image_fingerprint(image: str) -> dict:
    """LXD image fingerprint for the verifier sandbox."""
    try:
        completed = subprocess.run(
            ["lxc", "image", "info", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"image": image, "fingerprint": _unavailable(type(error).__name__)}
    if completed.returncode != 0:
        return {"image": image, "fingerprint": _unavailable("lxc image info failed")}
    fingerprint = _unavailable("fingerprint not present in lxc output")
    for line in completed.stdout.splitlines():
        if line.strip().lower().startswith("fingerprint:"):
            fingerprint = line.split(":", 1)[1].strip()
            break
    return {"image": image, "fingerprint": fingerprint}


def verifier_suite_digest(
    suite_version: str, suites: Iterable[Mapping[str, Any]] = ()
) -> dict:
    """Hash the hidden verifier suites, not just their version string.

    A version label can stay fixed while cases change underneath it. Hashing the
    declared cases makes a silently edited verifier detectable.
    """
    entries = [dict(suite) for suite in suites]
    return {
        "suite_version": suite_version,
        "suite_count": len(entries),
        "suites_sha256": canonical_hash(entries),
    }


# Version fields a worker must report before its framework digest resolves. A
# partial set hashes to a value that looks authoritative but is not, so an
# incomplete set stays an explicit gap instead.
WORKER_VERSION_FIELDS = ("python", "mlx", "mlx_lm")
WORKER_OPTIONAL_VERSION_FIELDS = ("machine", "system", "release", "macos")
# Which checkout the worker actually executed. Framework versions describe the
# libraries; they say nothing about the code. Two arms trained by different
# worker revisions are not a controlled comparison.
WORKER_CHECKOUT_FIELDS = (
    "revision",
    "tree",
    "dirty",
    "status_sha256",
    "worktree_sha256",
)


def _resolved_or_gap(value: Any, reason: str, *, path: str | None = None) -> Any:
    """Turn null, blank or malformed self-reported identity into a named gap.

    A field whose name ends in ``_sha256`` must be a raw 64-character lowercase
    SHA-256 digest. Accepting anything else let a worker answer ``false`` or
    ``{}`` and have it counted as reported model identity, and let two arms
    "match" on the same malformed value.
    """
    if isinstance(value, str) and value.strip().startswith("unavailable:"):
        # Already a named gap; re-wrapping it would bury the original reason.
        return value
    if value is None or (isinstance(value, str) and not value.strip()):
        return _unavailable(reason)
    if is_digest_field(path) and not is_sha256_hex(value):
        return _unavailable(
            f"{reason}: expected a 64-character lowercase SHA-256 hex digest, "
            f"got {type(value).__name__} {value!r}"
        )
    return value


def worker_checkout(payload: Mapping[str, Any] | None) -> dict:
    """Source identity of the worker checkout, or an explicit gap per field."""
    reported = dict(payload) if isinstance(payload, Mapping) else {}
    return {
        field: _resolved_or_gap(
            reported.get(field),
            f"worker does not report checkout {field}",
            path=f"worker.checkout.{field}",
        )
        for field in WORKER_CHECKOUT_FIELDS
    }


def worker_metadata(preflight: Any = None, *, host: str | None = None) -> dict:
    """Describe the remote training worker, or say plainly that it is unknown.

    ``preflight`` is a zero-argument callable returning the worker's own report.
    It is never called implicitly: a provenance command must not hang on a
    network stall, so an uncollected worker is recorded as an explicit gap
    rather than quietly omitted.

    Only host name, architecture, framework versions and checkout identity are
    kept. Credentials, identity files and connection strings are deliberately
    excluded.

    The worker's word is the only evidence here. This records what it claimed,
    not proof of what it ran; nothing short of a worker-side signed attestation
    would give that.
    """
    record: dict[str, Any] = {"host": host or _unavailable("no worker host supplied")}
    payload: dict[str, Any] = {}
    if preflight is None:
        record["status"] = _unavailable("worker metadata was not collected")
        return _complete_worker_record(record, payload)
    try:
        payload = dict(preflight())
    except Exception as error:  # noqa: BLE001 - any failure is a recorded gap
        record["status"] = _unavailable(
            f"worker preflight failed: {type(error).__name__}"
        )
        return _complete_worker_record(record, {})
    allowed = ("status", "machine", "system", "release", "python", "mlx", "mlx_lm")
    for key in allowed:
        if key in payload:
            record[key] = _resolved_or_gap(
                payload[key], f"worker reported a null {key}"
            )
    record.setdefault("status", "ok")
    return _complete_worker_record(record, payload)


def _complete_worker_record(
    record: dict[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    # Caller-supplied records start here too, not only records assembled by
    # worker_metadata. Keep the required identity claims explicit when a caller
    # omits them; an omitted field must not look like a completed collection.
    record["host"] = _resolved_or_gap(
        record.get("host"), "no worker host supplied"
    )
    record["status"] = _resolved_or_gap(
        record.get("status"), "worker status was not reported"
    )
    versions = {
        field: record[field]
        for field in (*WORKER_VERSION_FIELDS, *WORKER_OPTIONAL_VERSION_FIELDS)
        if field in record and not _is_gap(record[field])
    }

    if "macos" in payload and not _is_gap(payload.get("macos")):
        versions["macos"] = payload["macos"]
    record["framework_versions"] = versions
    unresolved = [field for field in WORKER_VERSION_FIELDS if field not in versions]
    record["framework_versions_sha256"] = (
        _unavailable("worker framework versions incomplete: " + ",".join(unresolved))
        if unresolved
        else canonical_hash(versions)
    )
    record["checkout"] = worker_checkout(payload.get("checkout"))
    # The control host cannot hash a path that lives on the worker, so this is
    # the worker's own digest over its own model files. When the worker declines
    # to produce one it says why, and the reason becomes the gap text rather
    # than a bare null nobody can act on.
    manifest = payload.get("model_manifest")
    manifest_reason = (
        manifest.get("reason")
        if isinstance(manifest, Mapping) and manifest.get("sha256") is None
        else None
    )
    if isinstance(manifest, Mapping):
        record["model_manifest"] = {
            key: manifest.get(key)
            for key in ("path", "file_count", "reason")
            if key in manifest
        }
    record["model_manifest_sha256"] = _resolved_or_gap(
        payload.get(
            "model_manifest_sha256",
            _unavailable("worker does not return model file digests"),
        ),
        f"worker reported no model manifest digest: {manifest_reason}"
        if manifest_reason
        else "worker reported a null model manifest digest",
        path="worker.model_manifest_sha256",
    )
    return record


def _is_gap(value: Any) -> bool:
    return value is None or (
        isinstance(value, str)
        and (not value.strip() or value.strip().startswith("unavailable:"))
    )


def collect_provenance(
    *,
    repo_root: str | Path,
    base_model: str,
    verifier_suite_version: str,
    training_config: Mapping[str, Any] | None = None,
    decoding_config: Mapping[str, Any] | None = None,
    verifier_suites: Iterable[Mapping[str, Any]] = (),
    model_paths: Mapping[str, str | Path] | None = None,
    sandbox_image: str | None = None,
    worker: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """Assemble the full reproducibility record and hash it.

    Every required section is emitted whether or not it could be resolved, so a
    reader can tell "not collected" from "not applicable" and ``missing_fields``
    can count the holes.
    """
    root = Path(repo_root)
    paths = dict(model_paths or {})
    models = {name: directory_digest(path) for name, path in paths.items()}
    for name in REQUIRED_MODELS:
        models.setdefault(
            name,
            {
                "path": None,
                "aggregate_sha256": _unavailable("no model path supplied"),
            },
        )
    supplied_worker = (
        _complete_worker_record(dict(worker), worker)
        if worker is not None
        else worker_metadata()
    )
    record: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "source": git_revision(root),
        "lockfile_sha256": file_digest(root / "uv.lock"),
        "pyproject_sha256": file_digest(root / "pyproject.toml"),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "base_model": base_model,
        "training_config_sha256": canonical_hash(dict(training_config or {})),
        "training_config": dict(training_config or {}),
        "decoding_config": dict(decoding_config or {}),
        "verifiers": verifier_suite_digest(verifier_suite_version, verifier_suites),
        "models": models,
        "worker": supplied_worker,
        "sandbox_image": (
            sandbox_image_fingerprint(sandbox_image)
            if sandbox_image
            else {"image": None, "fingerprint": _unavailable("no image requested")}
        ),
    }
    if extra:
        record["extra"] = dict(extra)
    record["provenance_sha256"] = canonical_hash(record)
    return record


def missing_fields(record: Mapping[str, Any]) -> list[str]:
    """Dotted paths of every value the collectors could not resolve.

    A run is still publishable with gaps; it is not publishable with hidden
    gaps. An absent required section counts as a gap, because an omitted field
    is exactly how an unreproducible run passes for a complete one.
    """
    gaps: list[str] = []

    def walk(value: Any, path: str) -> None:
        if _is_gap(value):
            gaps.append(path)
            return
        if is_digest_field(path) and not is_sha256_hex(value):
            # A digest field holding a non-digest is a gap at the digest
            # itself. Walking into it would name nested paths under a value
            # that is not an identity at all, and leave the required field
            # looking resolved.
            gaps.append(path)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))

    walk(record, "")

    for section in REQUIRED_SECTIONS:
        value = record.get(section)
        if value is None or isinstance(value, Mapping) and not value:
            gaps.append(section)
    models = record.get("models")
    for name in REQUIRED_MODELS:
        if not isinstance(models, Mapping) or name not in models:
            gaps.append(f"models.{name}")
    return sorted(set(gaps))
