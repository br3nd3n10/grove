"""Environment-driven defaults for the SSH worker configuration."""

from pathlib import Path

import pytest

from grove.remote import SshWorkerConfig


@pytest.fixture(
    params=(
        ("GROVE_WORKER_HOST", "host", "grove-worker-1"),
        (
            "GROVE_WORKER_IDENTITY",
            "identity_file",
            str(Path.home() / ".ssh" / "grove_worker"),
        ),
    ),
    ids=("host", "identity"),
)
def worker_setting(request):
    return request.param


def test_unset_worker_environment_uses_default(monkeypatch, worker_setting):
    env_name, field_name, expected = worker_setting
    monkeypatch.delenv(env_name, raising=False)

    assert getattr(SshWorkerConfig(), field_name) == expected


def test_empty_worker_environment_uses_default(monkeypatch, worker_setting):
    env_name, field_name, expected = worker_setting
    monkeypatch.setenv(env_name, "")

    assert getattr(SshWorkerConfig(), field_name) == expected


def test_set_worker_environment_is_used(monkeypatch, worker_setting):
    env_name, field_name, _ = worker_setting
    configured = "configured-worker-value"
    monkeypatch.setenv(env_name, configured)

    assert getattr(SshWorkerConfig(), field_name) == configured


def test_explicit_worker_argument_wins_over_environment(
    monkeypatch, worker_setting
):
    env_name, field_name, _ = worker_setting
    monkeypatch.setenv(env_name, "environment-value")
    explicit = "explicit-worker-value"

    config = SshWorkerConfig(**{field_name: explicit})

    assert getattr(config, field_name) == explicit
