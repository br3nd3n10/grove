"""Environment-driven host paths used by sandbox integration checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def sandbox_host_targets() -> tuple[Path, Path]:
    """Resolve the host directories the sandbox must not be able to read."""
    return (
        Path(os.environ.get("GROVE_SSH_DIR") or Path.home() / ".ssh"),
        Path(os.environ.get("GROVE_STORAGE_ROOT") or "/srv/storage/grove"),
    )


@pytest.mark.parametrize("mode", ("unset", "empty", "set"))
def test_sandbox_host_targets_follow_environment(monkeypatch, mode: str):
    for name in ("GROVE_SSH_DIR", "GROVE_STORAGE_ROOT"):
        monkeypatch.delenv(name, raising=False)

    if mode == "empty":
        monkeypatch.setenv("GROVE_SSH_DIR", "")
        monkeypatch.setenv("GROVE_STORAGE_ROOT", "")
    elif mode == "set":
        monkeypatch.setenv("GROVE_SSH_DIR", "/tmp/configured-ssh")
        monkeypatch.setenv("GROVE_STORAGE_ROOT", "/tmp/configured-storage")

    ssh_dir, storage_root = sandbox_host_targets()
    if mode == "set":
        assert ssh_dir == Path("/tmp/configured-ssh")
        assert storage_root == Path("/tmp/configured-storage")
    else:
        assert ssh_dir == Path.home() / ".ssh"
        assert storage_root == Path("/srv/storage/grove")
