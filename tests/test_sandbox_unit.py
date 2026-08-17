"""Unit tests for LXD sandbox hardening; all subprocess use is mocked.

Motivated by the EXP-002 failure: launch timeouts under load leaked RUNNING
grove-run-* containers, and every subsequent verification failed with
'LXD control command timed out: launch ...'.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from datetime import UTC

import pytest

from grove.sandbox import LxdSandbox, SandboxPolicy


class FakeProcess:
    """Stands in for subprocess.Popen with real pipe fds for the selector."""

    def __init__(self, stdout: bytes = b"ok\n", stderr: bytes = b"", returncode: int = 0):
        read_out, write_out = os.pipe()
        os.write(write_out, stdout)
        os.close(write_out)
        read_err, write_err = os.pipe()
        os.write(write_err, stderr)
        os.close(write_err)
        self.stdout = os.fdopen(read_out, "rb")
        self.stderr = os.fdopen(read_err, "rb")
        self.stdin = io.BytesIO()
        self.returncode = returncode

    def kill(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


class FakeLxd:
    """Scriptable stand-in for subprocess.run / subprocess.Popen."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float | None]] = []
        self.timeout_launches = 0
        self.timeout_exec_setup = False
        self.list_stdout = ""
        self._launch_attempts = 0

    def run(self, command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        self.calls.append((list(command), timeout))
        verb = command[1]
        if verb == "launch":
            self._launch_attempts += 1
            if self._launch_attempts <= self.timeout_launches:
                raise subprocess.TimeoutExpired(command, timeout or 0)
        if verb == "exec" and "install" in command and self.timeout_exec_setup:
            raise subprocess.TimeoutExpired(command, timeout or 0)
        if verb == "list":
            return subprocess.CompletedProcess(command, 0, self.list_stdout, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    def popen(self, command: list[str], **kwargs) -> FakeProcess:
        self.calls.append((list(command), None))
        return FakeProcess()

    def verbs(self) -> list[str]:
        return [call[0][1] for call in self.calls]


@pytest.fixture()
def fake_lxd(monkeypatch: pytest.MonkeyPatch) -> FakeLxd:
    fake = FakeLxd()
    monkeypatch.setattr("grove.sandbox.subprocess.run", fake.run)
    monkeypatch.setattr("grove.sandbox.subprocess.Popen", fake.popen)
    return fake


def _fast_policy(tmp_path, **overrides) -> SandboxPolicy:
    """Policy with no retry sleeps, no wait grace, and an isolated lock."""
    defaults: dict[str, object] = {
        "retry_backoff_seconds": 0.0,
        "launch_wait_grace_seconds": 0.0,
        "launch_lock_path": str(tmp_path / "launch.lock"),
    }
    defaults.update(overrides)
    return SandboxPolicy(**defaults)


def test_policy_defaults_keep_execution_bound_and_raise_control_bound():
    policy = SandboxPolicy()
    assert policy.timeout_seconds == 5.0
    assert policy.control_timeout_seconds >= 120.0
    assert policy.launch_attempts >= 3
    assert policy.launch_wait_grace_seconds > 0.0
    # A contended waiter must be able to sit out several bounded lock
    # holders before degrading to an unthrottled (stampeding) pool op.
    assert policy.lock_wait_seconds >= 2 * policy.control_timeout_seconds

def test_successful_run_records_timings(fake_lxd: FakeLxd):
    result = LxdSandbox().run_python("print('hi')")

    assert result.clean
    timings = result.metadata["timings"]
    assert timings["launch_attempts"] == 1
    for key in ("launch_seconds", "exec_seconds", "delete_seconds"):
        assert key in timings
        assert timings[key] >= 0.0


def test_launch_and_delete_use_generous_timeout(fake_lxd: FakeLxd):
    LxdSandbox().run_python("print('hi')")

    for command, timeout in fake_lxd.calls:
        verb = command[1]
        if verb in {"launch", "delete"} or command[1:3] == ["file", "push"]:
            assert timeout is not None and timeout >= 120.0, command
        elif verb == "exec" and command[0] == "lxc" and timeout is not None:
            assert timeout == 30.0


def test_timed_out_launch_is_retried_after_cleanup_delete(
    fake_lxd: FakeLxd, tmp_path
):
    fake_lxd.timeout_launches = 1

    result = LxdSandbox(_fast_policy(tmp_path)).run_python("print('hi')")

    assert result.clean
    assert result.metadata["timings"]["launch_attempts"] == 2
    verbs = fake_lxd.verbs()
    first_launch = verbs.index("launch")
    second_launch = verbs.index("launch", first_launch + 1)
    # A force-delete of the possibly half-created instance runs between tries.
    between = verbs[first_launch + 1 : second_launch]
    assert "delete" in between
    delete_command = fake_lxd.calls[first_launch + between.index("delete") + 1][0]
    assert "--force" in delete_command


def test_launch_exhausting_all_attempts_returns_infrastructure_error(
    fake_lxd: FakeLxd, tmp_path
):
    policy = _fast_policy(tmp_path, launch_attempts=3)
    fake_lxd.timeout_launches = 3

    result = LxdSandbox(policy).run_python("print('hi')")

    assert not result.clean
    assert result.infrastructure_error is not None
    assert "timed out" in result.infrastructure_error
    assert fake_lxd.verbs().count("launch") == 3
    # No fourth attempt, and delete still ran in the finally path.
    assert fake_lxd.verbs()[-1] == "delete"


def test_two_timeouts_then_success_on_third_attempt(fake_lxd: FakeLxd, tmp_path):
    fake_lxd.timeout_launches = 2

    result = LxdSandbox(_fast_policy(tmp_path)).run_python("print('hi')")

    assert result.clean
    assert result.metadata["timings"]["launch_attempts"] == 3


def test_client_timeout_with_server_side_completion_counts_as_success(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A client-side launch timeout must not discard a server-side success.

    The client timeout does not cancel the LXD operation; when the instance
    reaches Running anyway, the run should use it instead of deleting it and
    starting over.
    """
    fake_lxd.timeout_launches = 10  # every launch times out client-side
    original_run = fake_lxd.run

    def run_with_running_list(command, **kwargs):
        if command[1] == "list":
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            listing = json.dumps([{"name": command[2], "status": "Running"}])
            return subprocess.CompletedProcess(command, 0, listing, "")
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", run_with_running_list)
    policy = _fast_policy(tmp_path, launch_wait_grace_seconds=5.0)

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean
    assert result.metadata["timings"]["launch_attempts"] == 1


def test_client_timeout_with_stopped_instance_is_started_then_adopted(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A stranded Stopped instance must be started, not deleted.

    `lxc launch` is create-then-start driven by the client; killing the
    timed-out client can leave the create finished but the start never
    requested. Waiting for Running without issuing the start would always
    fail and discard a perfectly good instance (the 2026-08-08 soak probe).
    """
    fake_lxd.timeout_launches = 10  # every launch times out client-side
    original_run = fake_lxd.run

    def run_with_stopped_list(command, **kwargs):
        if command[1] == "list":
            status = "Running" if "start" in fake_lxd.verbs() else "Stopped"
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            listing = json.dumps([{"name": command[2], "status": status}])
            return subprocess.CompletedProcess(command, 0, listing, "")
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", run_with_stopped_list)
    policy = _fast_policy(tmp_path, launch_wait_grace_seconds=5.0)

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean
    assert result.metadata["timings"]["launch_attempts"] == 1
    assert "start" in fake_lxd.verbs()


def test_client_timeout_with_instance_not_yet_listed_is_awaited_then_adopted(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """An instance absent from `lxc list` right after a launch timeout is not
    dead — the create operation may still be unpacking the rootfs server-side.

    The 2026-08-08 soak probe failed because absence was read as "never
    materialized": the in-flight instance was deleted and every relaunch hit
    the same timeout. Within the grace window, absence means "not yet".
    """
    fake_lxd.timeout_launches = 10  # every launch times out client-side
    original_run = fake_lxd.run
    list_calls = {"count": 0}

    def run_with_late_appearance(command, **kwargs):
        if command[1] == "list":
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            list_calls["count"] += 1
            if list_calls["count"] < 3:
                return subprocess.CompletedProcess(command, 0, "[]", "")
            listing = json.dumps([{"name": command[2], "status": "Running"}])
            return subprocess.CompletedProcess(command, 0, listing, "")
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", run_with_late_appearance)
    policy = _fast_policy(tmp_path, launch_wait_grace_seconds=10.0)

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean
    assert result.metadata["timings"]["launch_attempts"] == 1
    assert list_calls["count"] >= 3


def test_launch_timeout_with_partial_stderr_is_still_salvaged(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """lxc writes warnings (e.g. the no-network notice) to stderr before a
    slow launch is killed; partial stderr must not disguise the client-side
    timeout and skip salvage of a usable instance."""
    original_run = fake_lxd.run
    launched = {"count": 0}

    def run_with_noisy_timeout(command, **kwargs):
        if command[1] == "launch":
            launched["count"] += 1
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            raise subprocess.TimeoutExpired(
                command,
                kwargs.get("timeout") or 0,
                stderr=b"The instance you are starting does not have any "
                b"network attached to it.\n",
            )
        if command[1] == "list":
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            listing = json.dumps([{"name": command[2], "status": "Running"}])
            return subprocess.CompletedProcess(command, 0, listing, "")
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", run_with_noisy_timeout)
    policy = _fast_policy(tmp_path, launch_wait_grace_seconds=5.0)

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean
    assert launched["count"] == 1  # salvaged, not relaunched


def test_start_rejected_while_create_in_flight_is_retried(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A start issued while the create is still finishing can be rejected;
    the rejection must not be remembered as a completed start request."""
    fake_lxd.timeout_launches = 10  # every launch times out client-side
    original_run = fake_lxd.run
    starts = {"count": 0}

    def run_with_busy_start(command, **kwargs):
        if command[1] == "start":
            starts["count"] += 1
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            if starts["count"] == 1:
                return subprocess.CompletedProcess(
                    command, 1, "", "Error: Instance is busy running a create operation"
                )
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1] == "list":
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            status = "Running" if starts["count"] >= 2 else "Stopped"
            listing = json.dumps([{"name": command[2], "status": status}])
            return subprocess.CompletedProcess(command, 0, listing, "")
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", run_with_busy_start)
    policy = _fast_policy(tmp_path, launch_wait_grace_seconds=10.0)

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean
    assert result.metadata["timings"]["launch_attempts"] == 1
    assert starts["count"] == 2


def test_delete_client_timeout_with_server_side_completion_is_success(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A client-side delete timeout whose server-side delete finishes must
    count as success, not leak a container or trigger pointless retries.

    This is the EXP-002 leak pattern on the delete side: the timed-out
    client call abandoned a delete that was still running server-side.
    """
    original_run = fake_lxd.run
    deletes = {"count": 0}
    lists = {"count": 0}

    def run_with_slow_delete(command, **kwargs):
        if command[1] == "delete":
            deletes["count"] += 1
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout") or 0)
        if command[1] == "list":
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            lists["count"] += 1
            if lists["count"] == 1:
                # Still tearing down on the first poll, gone on the next.
                listing = json.dumps([{"name": command[2], "status": "Stopped"}])
                return subprocess.CompletedProcess(command, 0, listing, "")
            return subprocess.CompletedProcess(command, 0, "[]", "")
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", run_with_slow_delete)
    policy = _fast_policy(tmp_path, launch_wait_grace_seconds=10.0)

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean
    assert deletes["count"] == 1  # verified gone; no retry needed
    assert lists["count"] >= 2


def test_failed_delete_with_instance_still_present_is_retried(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A delete that fails while the instance is still listed must be retried
    rather than silently leaking a RUNNING container."""
    original_run = fake_lxd.run
    deletes = {"count": 0}

    def run_with_flaky_delete(command, **kwargs):
        if command[1] == "delete":
            deletes["count"] += 1
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            if deletes["count"] == 1:
                return subprocess.CompletedProcess(
                    command, 1, "", "Error: websocket: bad handshake"
                )
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1] == "list":
            fake_lxd.calls.append((list(command), kwargs.get("timeout")))
            listing = json.dumps([{"name": command[2], "status": "Running"}])
            return subprocess.CompletedProcess(command, 0, listing, "")
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", run_with_flaky_delete)
    policy = _fast_policy(tmp_path)

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean
    assert deletes["count"] == 2


def test_concurrent_launches_are_serialized_by_host_lock(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The flock throttle must keep at most one `lxc launch` in flight.

    On a dir storage pool, concurrent launches stampede the image unpack and
    their latency grows past any fixed timeout (the 2026-08-08 soak failure).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    state = {"active": 0, "max_active": 0}
    gate = threading.Lock()
    original_run = fake_lxd.run

    def tracked_run(command, **kwargs):
        if command[1] == "launch":
            with gate:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with gate:
                state["active"] -= 1
        return original_run(command, **kwargs)

    monkeypatch.setattr("grove.sandbox.subprocess.run", tracked_run)
    policy = _fast_policy(tmp_path)

    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(
                lambda _: LxdSandbox(policy).run_python("print('hi')"), range(4)
            )
        )

    assert all(result.clean for result in results)
    assert state["max_active"] == 1


def test_concurrent_deletes_share_the_pool_lock_with_launches(
    fake_lxd: FakeLxd, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Launch and delete must serialize together: at most one heavy pool op.

    The 2026-08-08 soak showed unthrottled concurrent deletes saturating the
    dir pool (120s deletes at 8-way concurrency) and dragging launches past
    their timeout even though launches themselves were serialized.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    state = {"active": 0, "max_active": 0}
    gate = threading.Lock()
    original_run = fake_lxd.run

    def tracked_run(command, **kwargs):
        heavy = command[1] in {"launch", "delete"}
        if heavy:
            with gate:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
        try:
            return original_run(command, **kwargs)
        finally:
            if heavy:
                with gate:
                    state["active"] -= 1

    monkeypatch.setattr("grove.sandbox.subprocess.run", tracked_run)
    policy = _fast_policy(tmp_path)

    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(
                lambda _: LxdSandbox(policy).run_python("print('hi')"), range(4)
            )
        )

    assert all(result.clean for result in results)
    assert state["max_active"] == 1
    assert all(
        "delete_queue_seconds" in result.metadata["timings"] for result in results
    )


def test_launch_lock_failure_degrades_to_unthrottled(fake_lxd: FakeLxd, tmp_path):
    """An unopenable lock path must never fail the run, only skip throttling."""
    policy = _fast_policy(
        tmp_path, launch_lock_path=str(tmp_path / "missing-dir" / "launch.lock")
    )

    result = LxdSandbox(policy).run_python("print('hi')")

    assert result.clean


def test_instance_is_deleted_even_when_control_command_times_out(fake_lxd: FakeLxd):
    fake_lxd.timeout_exec_setup = True

    result = LxdSandbox().run_python("print('hi')")

    assert result.infrastructure_error is not None
    last_command, last_timeout = fake_lxd.calls[-1]
    assert last_command[1] == "delete"
    assert "--force" in last_command
    assert last_timeout is not None and last_timeout >= 120.0
    assert "delete_seconds" in result.metadata["timings"]


def _preflight_control(listing: str):
    def fake(*arguments: str, timeout: float = 30.0):
        if arguments[0] == "version":
            return subprocess.CompletedProcess(arguments, 0, "5.21", "")
        if arguments[0] == "image":
            return subprocess.CompletedProcess(arguments, 0, "fingerprint: abc", "")
        if arguments[0] == "profile":
            return subprocess.CompletedProcess(arguments, 0, "devices: {}", "")
        if arguments[0] == "list":
            return subprocess.CompletedProcess(arguments, 0, listing, "")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    return staticmethod(fake)


def test_preflight_fails_loudly_on_leftover_instances(monkeypatch: pytest.MonkeyPatch):
    # No created_at at all: unknown age must be treated as a leak.
    leftovers = json.dumps(
        [{"name": "grove-run-deadbeef0001", "status": "Running"}]
    )
    monkeypatch.setattr(LxdSandbox, "_control", _preflight_control(leftovers))

    with pytest.raises(RuntimeError, match="grove-run-deadbeef0001"):
        LxdSandbox().preflight()


def test_preflight_flags_old_instances_as_leftovers(monkeypatch: pytest.MonkeyPatch):
    from datetime import datetime, timedelta

    stale = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    leftovers = json.dumps(
        [
            {
                "name": "grove-run-deadbeef0002",
                "status": "Running",
                "created_at": stale,
            }
        ]
    )
    monkeypatch.setattr(LxdSandbox, "_control", _preflight_control(leftovers))

    with pytest.raises(RuntimeError, match="grove-run-deadbeef0002"):
        LxdSandbox().preflight()


def test_preflight_tolerates_fresh_instances_from_concurrent_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    """A young grove-run-* instance belongs to an in-flight peer, not a leak.

    The 2026-08-08 soak ran preflight while workers were mid-cycle; treating
    their live instances as leftovers made preflight fail on a healthy host.
    """
    from datetime import datetime

    # LXD-style RFC 3339 with nanoseconds; the parser must trim to micros.
    fresh = (
        datetime.now(UTC)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")
        + "123Z"
    )
    listing = json.dumps(
        [
            {
                "name": "grove-run-cafecafe0003",
                "status": "Running",
                "created_at": fresh,
            }
        ]
    )
    monkeypatch.setattr(LxdSandbox, "_control", _preflight_control(listing))

    from grove.sandbox import SandboxResult

    def healthy_probe(self, source, payload=None):
        return SandboxResult(
            exit_code=0,
            stdout="grove-preflight-ok\n",
            stderr="",
            duration_seconds=1.0,
            execution_seconds=0.2,
            metadata={"timings": {}},
        )

    monkeypatch.setattr(LxdSandbox, "run_python", healthy_probe)

    report = LxdSandbox().preflight()

    assert report["network_attached"] is False


def test_preflight_runs_real_probe_and_fails_on_degraded_sandbox(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(LxdSandbox, "_control", _preflight_control("[]"))

    from grove.sandbox import SandboxResult

    def broken_probe(self, source, payload=None):
        return SandboxResult(
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=1.0,
            infrastructure_error="LXD control command timed out: launch",
        )

    monkeypatch.setattr(LxdSandbox, "run_python", broken_probe)

    with pytest.raises(RuntimeError, match="preflight probe failed"):
        LxdSandbox().preflight()


def test_preflight_success_reports_probe_timings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(LxdSandbox, "_control", _preflight_control("[]"))

    from grove.sandbox import SandboxResult

    def healthy_probe(self, source, payload=None):
        return SandboxResult(
            exit_code=0,
            stdout="grove-preflight-ok\n",
            stderr="",
            duration_seconds=9.5,
            execution_seconds=0.2,
            metadata={
                "timings": {"launch_seconds": 8.0, "exec_seconds": 0.2, "delete_seconds": 1.0}
            },
        )

    monkeypatch.setattr(LxdSandbox, "run_python", healthy_probe)

    report = LxdSandbox().preflight()

    assert report["network_attached"] is False
    assert report["probe_timings"]["launch_seconds"] == 8.0


def test_preflight_rejects_networked_profile(monkeypatch: pytest.MonkeyPatch):
    def fake(*arguments: str, timeout: float = 30.0):
        if arguments[0] == "profile":
            return subprocess.CompletedProcess(
                arguments, 0, "devices:\n  eth0:\n    type: nic\n", ""
            )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(LxdSandbox, "_control", staticmethod(fake))

    with pytest.raises(RuntimeError, match="network device"):
        LxdSandbox().preflight()


def test_control_never_inherits_parent_stdin(monkeypatch: pytest.MonkeyPatch):
    """`lxc launch` reads instance config from non-TTY stdin until EOF.

    A workflow runner holds an open stdin pipe; inheriting it made every
    launch block until the client timeout and killed all 123 EXP-002
    verifications on 2026-08-07. stdin must be DEVNULL, always.
    """
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("grove.sandbox.subprocess.run", fake_run)

    LxdSandbox._control("launch", "grove-python-base", "grove-run-test")

    assert captured.get("stdin") == subprocess.DEVNULL
