from __future__ import annotations

import fcntl
import json
import os
import re
import selectors
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    image: str = "grove-python-base"
    profile: str = "grove-sandbox"
    timeout_seconds: float = 5.0
    maximum_output_bytes: int = 1_000_000
    user_id: int = 1000
    group_id: int = 1000
    # Timeout for host-side LXD control commands (launch/delete/file push).
    # These are trusted operations whose latency tracks host load, not the
    # untrusted program, so the bound is deliberately generous. EXP-002 failed
    # wholesale because launch occasionally exceeded the old 30s bound under
    # load and the timed-out client call leaked a RUNNING container.
    control_timeout_seconds: float = 120.0
    # Launch attempts before giving up. The 2026-08-08 soak showed a single
    # retry is not enough when the host is busy.
    launch_attempts: int = 3
    # Base pause between launch retries (multiplied by the attempt number).
    retry_backoff_seconds: float = 2.0
    # After a client-side launch timeout the server-side operation keeps
    # running; wait this long for the instance to reach Running before
    # declaring the launch dead and deleting it.
    launch_wait_grace_seconds: float = 60.0
    # Cross-process throttle for the two rootfs-heavy pool operations,
    # `lxc launch` and `lxc delete`. On a dir storage pool a launch unpacks
    # the full image and a delete removes the whole rootfs tree; run
    # concurrently they stampede the pool and their latency grows past any
    # fixed bound (the 2026-08-08 soak measured 45s launches and 120s deletes
    # at 8-way concurrency). Serializing them host-wide keeps each near its
    # ~3s sequential cost. Queue wait is recorded separately and does not
    # count against the operation timeout.
    launch_lock_path: str = os.path.join(
        tempfile.gettempdir(), "grove-sandbox-launch.lock"
    )
    # Upper bound on waiting for the pool-op lock. A holder is bounded by
    # control_timeout_seconds, so a contended waiter must be able to sit out
    # several holders; giving up early and proceeding unthrottled recreates
    # the stampede the lock exists to prevent (the 2026-08-08 soak probe
    # failed exactly that way). Only past this bound do we degrade to
    # unthrottled as a last resort, because the lock must never fail a run.
    lock_wait_seconds: float = 600.0
    # A grove-run-* instance younger than this may belong to a concurrent
    # in-flight run (launch + exec + delete under load); only instances older
    # than this age count as leaked. A genuine leak persists indefinitely, so
    # an age gate loses nothing.
    leftover_age_seconds: float = 600.0


@dataclass(frozen=True, slots=True)
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    execution_seconds: float = 0.0
    timed_out: bool = False
    output_limited: bool = False
    infrastructure_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.output_limited
            and self.infrastructure_error is None
        )


class LxdSandbox:
    """Runs an untrusted Python program in a fresh, networkless LXD container."""

    _INSTANCE_PREFIX = "grove-run-"

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def preflight(self) -> dict[str, Any]:
        version = self._control("version").stdout.strip()
        image = self._control("image", "info", self.policy.image)
        profile = self._control("profile", "show", self.policy.profile)
        if image.returncode != 0:
            raise RuntimeError(f"sandbox image unavailable: {image.stderr.strip()}")
        if profile.returncode != 0:
            raise RuntimeError(f"sandbox profile unavailable: {profile.stderr.strip()}")
        if "type: nic" in profile.stdout or "nictype:" in profile.stdout:
            raise RuntimeError("sandbox profile must not contain a network device")

        leftovers = self._leftover_instances()
        if leftovers:
            raise RuntimeError(
                "sandbox is degraded: leftover instances exist "
                f"({', '.join(sorted(leftovers))}, older than "
                f"{self.policy.leftover_age_seconds:.0f}s); delete them with "
                "`lxc delete --force <name>` before starting a run"
            )

        probe = self.run_python('print("grove-preflight-ok")')
        if not probe.clean or "grove-preflight-ok" not in probe.stdout:
            detail = probe.infrastructure_error or probe.stderr.strip() or "unknown"
            raise RuntimeError(
                "sandbox preflight probe failed "
                f"(exit={probe.exit_code}, timed_out={probe.timed_out}): {detail}"
            )

        return {
            "lxc_version": version,
            "image": self.policy.image,
            "profile": self.policy.profile,
            "network_attached": False,
            "probe_timings": probe.metadata.get("timings", {}),
        }

    def run_python(self, source: str, payload: Any = None) -> SandboxResult:
        instance = f"{self._INSTANCE_PREFIX}{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        timings: dict[str, float | int] = {}
        try:
            result = self._run_in_instance(instance, source, payload, started, timings)
        finally:
            # Always force-delete, even after a control timeout: a client-side
            # timeout does not cancel the server-side operation, so a
            # half-created instance may still exist and must not outlive us.
            delete_started = time.monotonic()
            self._delete_instance(instance, timings)
            timings["delete_seconds"] = round(time.monotonic() - delete_started, 3)
        result.metadata["timings"] = timings
        return result

    def _run_in_instance(
        self,
        instance: str,
        source: str,
        payload: Any,
        started: float,
        timings: dict[str, float | int],
    ) -> SandboxResult:
        launch = self._launch_with_retry(instance, timings)
        if launch.returncode != 0:
            return self._infrastructure_result(started, launch.stderr)
        setup = self._control(
            "exec",
            instance,
            "--",
            "install",
            "-d",
            "-m",
            "755",
            "/workspace",
        )
        if setup.returncode != 0:
            return self._infrastructure_result(started, setup.stderr)

        try:
            push = subprocess.run(
                ["lxc", "file", "push", "-", f"{instance}/workspace/candidate.py"],
                input=source,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.policy.control_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return self._infrastructure_result(started, "lxc file push timed out")
        if push.returncode != 0:
            return self._infrastructure_result(started, push.stderr)
        permissions = self._control(
            "exec", instance, "--", "chmod", "0555", "/workspace/candidate.py"
        )
        if permissions.returncode != 0:
            return self._infrastructure_result(started, permissions.stderr)

        command = [
            "lxc",
            "exec",
            "--mode=non-interactive",
            f"--user={self.policy.user_id}",
            f"--group={self.policy.group_id}",
            instance,
            "--",
            "prlimit",
            "--cpu=4",
            "--fsize=1048576",
            "--nofile=64",
            "--nproc=32",
            "--",
            "python3",
            "-I",
            "-B",
            "/workspace/candidate.py",
        ]
        input_bytes = json.dumps(payload).encode() if payload is not None else b""
        return self._run_limited(command, input_bytes, started, instance, timings)

    def _launch_with_retry(
        self, instance: str, timings: dict[str, float | int]
    ) -> subprocess.CompletedProcess[str]:
        attempts = 0
        while True:
            attempts += 1
            with self._pool_slot(timings, "launch_queue_seconds"):
                # Start the clock only once the slot is held: queue wait is
                # reported separately as launch_queue_seconds and must not
                # masquerade as launch latency.
                launch_started = time.monotonic()
                launch = self._control(
                    "launch",
                    self.policy.image,
                    instance,
                    "--profile",
                    self.policy.profile,
                    timeout=self.policy.control_timeout_seconds,
                )
                if self._timed_out_client_side(launch) and self._wait_for_running(
                    instance
                ):
                    # The client gave up, but the server-side operation left
                    # the instance usable; adopt it instead of leaking it.
                    # Salvage runs under the pool slot because the server-side
                    # create/start is still a heavy pool op in flight.
                    launch = subprocess.CompletedProcess(
                        launch.args, 0, launch.stdout, ""
                    )
            timings["launch_seconds"] = round(time.monotonic() - launch_started, 3)
            timings["launch_attempts"] = attempts
            if launch.returncode == 0 or attempts >= max(
                1, self.policy.launch_attempts
            ):
                return launch
            # The failed or timed-out launch may have half-created the
            # instance server-side; clear it by name before the retry so the
            # retry does not collide and nothing leaks.
            self._delete_instance(instance, timings)
            if self.policy.retry_backoff_seconds > 0:
                time.sleep(self.policy.retry_backoff_seconds * attempts)

    def _delete_instance(
        self, instance: str, timings: dict[str, float | int]
    ) -> subprocess.CompletedProcess[str]:
        """Force-delete under the pool-op lock, verifying the instance is gone.

        On a dir pool a delete removes the whole rootfs tree; the 2026-08-08
        soak showed unthrottled concurrent deletes saturating the pool and
        pushing both deletes and launches past the 120s control bound.

        A client-side delete timeout does not cancel the server-side delete,
        which usually finishes moments later; waiting for the instance to
        disappear turns that case into success instead of a leaked RUNNING
        container (the EXP-002 leak pattern). Only a delete that fails with
        the instance still present is retried.
        """
        attempts = 0
        while True:
            attempts += 1
            with self._pool_slot(timings, "delete_queue_seconds"):
                delete = self._control(
                    "delete",
                    "--force",
                    instance,
                    timeout=self.policy.control_timeout_seconds,
                )
            if delete.returncode == 0:
                return delete
            # A timed-out client leaves the server-side delete running; give
            # it the grace window. A plain failure gets one immediate check
            # (it may have failed because the instance never existed).
            grace = (
                self.policy.launch_wait_grace_seconds
                if self._timed_out_client_side(delete)
                else 0.0
            )
            if self._instance_absent(instance, grace):
                return subprocess.CompletedProcess(
                    delete.args, 0, delete.stdout, ""
                )
            if attempts >= max(1, self.policy.launch_attempts):
                return delete
            if self.policy.retry_backoff_seconds > 0:
                time.sleep(self.policy.retry_backoff_seconds * attempts)

    def _instance_absent(self, instance: str, grace_seconds: float) -> bool:
        """True once the instance no longer appears in `lxc list`.

        Polls within grace_seconds to let an in-flight server-side delete
        finish tearing the instance down. An unlistable state counts as
        present: unknown means unresolved, and the caller must retry.
        """
        deadline = time.monotonic() + grace_seconds
        while True:
            listing = self._control("list", instance, "--format", "json")
            if listing.returncode == 0:
                try:
                    entries = json.loads(listing.stdout or "[]")
                except json.JSONDecodeError:
                    entries = None
                if entries is not None and not any(
                    entry.get("name") == instance for entry in entries
                ):
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(1.0)

    @contextmanager
    def _pool_slot(
        self, timings: dict[str, float | int], queue_key: str
    ) -> Iterator[None]:
        """Hold the host-wide pool-op flock; degrade to unthrottled last.

        The lock only trades pool-op parallelism for bounded latency, so any
        failure to acquire it (permissions, unwritable tmpdir, or a peer
        holding it past lock_wait_seconds) must never fail the run itself.
        """
        queue_started = time.monotonic()
        deadline = queue_started + self.policy.lock_wait_seconds
        handle = None
        locked = False
        try:
            try:
                handle = open(  # noqa: SIM115 - held across the yield
                    self.policy.launch_lock_path, "a+"
                )
            except OSError:
                handle = None
            if handle is not None:
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(0.1)
            timings[queue_key] = round(time.monotonic() - queue_started, 3)
            yield
        finally:
            if handle is not None:
                if locked:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                handle.close()

    @staticmethod
    def _timed_out_client_side(
        result: subprocess.CompletedProcess[str],
    ) -> bool:
        return result.returncode == 124 and "timed out" in (result.stderr or "")

    def _wait_for_running(self, instance: str) -> bool:
        """Bring the instance to Running after a client-side launch timeout.

        `lxc launch` is create-then-start driven by the client; killing the
        timed-out client can strand the instance Stopped with the create
        finished but the start never requested. Issue the start ourselves
        rather than deleting a perfectly good instance.

        Two loaded-host wrinkles (the 2026-08-08 soak probe hit the first):
        the instance may not appear in `lxc list` at all while the create
        operation is still unpacking the rootfs, so absence within the grace
        window means "not yet", not "never"; and a start issued while the
        create is still finishing can be rejected, so a failed start request
        must be retried on the next poll instead of being remembered as done.
        """
        deadline = time.monotonic() + self.policy.launch_wait_grace_seconds
        while True:
            listing = self._control("list", instance, "--format", "json")
            if listing.returncode == 0:
                try:
                    entries = json.loads(listing.stdout or "[]")
                except json.JSONDecodeError:
                    entries = []
                matches = [
                    entry for entry in entries if entry.get("name") == instance
                ]
                if matches:
                    status = matches[0].get("status")
                    if status == "Running":
                        return True
                    if status == "Stopped":
                        start = self._control(
                            "start",
                            instance,
                            timeout=self.policy.control_timeout_seconds,
                        )
                        if start.returncode == 0:
                            # The start request itself may lawfully consume
                            # more than the remaining grace (it is bounded by
                            # control_timeout_seconds); once it succeeds, give
                            # the instance a fresh grace window to be observed
                            # Running rather than deleting it mid-start.
                            deadline = time.monotonic() + max(
                                self.policy.launch_wait_grace_seconds, 1.0
                            )
                        continue
                # Absent: the create operation may still be materializing the
                # instance server-side; keep polling until the grace deadline.
            if time.monotonic() >= deadline:
                return False
            time.sleep(1.0)

    def _leftover_instances(self) -> list[str]:
        """Names of grove-run-* instances old enough to be genuine leaks.

        Instances younger than leftover_age_seconds may belong to concurrent
        in-flight runs (the 2026-08-08 soak ran preflight while other workers
        were mid-cycle) and are not reported. An unparseable or missing
        created_at is treated as a leak: unknown age means unknown state.
        """
        listing = self._control("list", self._INSTANCE_PREFIX, "--format", "json")
        if listing.returncode != 0:
            raise RuntimeError(
                f"cannot list sandbox instances: {listing.stderr.strip()}"
            )
        try:
            instances = json.loads(listing.stdout or "[]")
        except json.JSONDecodeError as error:
            raise RuntimeError(f"unparseable lxc list output: {error}") from error
        now = datetime.now(UTC)
        leftovers: list[str] = []
        for item in instances:
            name = str(item.get("name", ""))
            if not name.startswith(self._INSTANCE_PREFIX):
                continue
            created = self._parse_lxd_time(str(item.get("created_at", "")))
            if (
                created is None
                or (now - created).total_seconds()
                >= self.policy.leftover_age_seconds
            ):
                leftovers.append(name)
        return leftovers

    @staticmethod
    def _parse_lxd_time(value: str) -> datetime | None:
        value = value.strip()
        if not value:
            return None
        # LXD emits RFC 3339 with nanosecond precision; fromisoformat only
        # accepts up to microseconds.
        value = re.sub(r"\.(\d{6})\d+", r".\1", value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    def _run_limited(
        self,
        command: list[str],
        input_bytes: bytes,
        started: float,
        instance: str,
        timings: dict[str, float | int],
    ) -> SandboxResult:
        execution_started = time.monotonic()
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(input_bytes)
        process.stdin.close()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        collected = {"stdout": bytearray(), "stderr": bytearray()}
        timed_out = False
        output_limited = False
        # Stop the clock when the guest program is killed, not when LXD has
        # finished tearing the container down. The enforced window is what the
        # policy promises; force-stop and delete are host bookkeeping whose cost
        # varies with host load and would otherwise make the bound untestable.
        stopped_at: float | None = None

        while selector.get_map():
            if time.monotonic() - execution_started > self.policy.timeout_seconds:
                timed_out = True
                process.kill()
                stopped_at = time.monotonic()
                self._control("stop", "--force", instance)
                break
            for key, _ in selector.select(timeout=0.05):
                chunk = os.read(key.fileobj.fileno(), 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                destination = collected[key.data]
                destination.extend(chunk)
                if (
                    sum(len(value) for value in collected.values())
                    > self.policy.maximum_output_bytes
                ):
                    output_limited = True
                    process.kill()
                    stopped_at = time.monotonic()
                    self._control("stop", "--force", instance)
                    break
            if output_limited:
                break

        if stopped_at is None:
            stopped_at = time.monotonic()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        timings["exec_seconds"] = round(stopped_at - execution_started, 3)
        return SandboxResult(
            exit_code=process.returncode,
            stdout=bytes(
                collected["stdout"][: self.policy.maximum_output_bytes]
            ).decode(errors="replace"),
            stderr=bytes(
                collected["stderr"][: self.policy.maximum_output_bytes]
            ).decode(errors="replace"),
            duration_seconds=time.monotonic() - started,
            execution_seconds=stopped_at - execution_started,
            timed_out=timed_out,
            output_limited=output_limited,
            metadata={
                "image": self.policy.image,
                "profile": self.policy.profile,
                "network_attached": False,
            },
        )

    @staticmethod
    def _control(
        *arguments: str, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["lxc", *arguments],
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
                # Never inherit the parent's stdin. Under a workflow runner the
                # parent holds an open, never-closing stdin pipe, and `lxc
                # launch` reads instance config from non-TTY stdin until EOF --
                # so every launch blocked until the client timeout. That single
                # inheritance killed all 123 EXP-002 verifications on
                # 2026-08-07 while the same commands ran in seconds from an
                # interactive shell.
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            # Always append the timeout marker: lxc may have written partial
            # stderr (e.g. the no-network warning) before the client was
            # killed, and _timed_out_client_side keys on this marker. Without
            # it, a timed-out launch with any stderr at all was misread as a
            # plain failure and its salvageable instance was deleted.
            marker = f"LXD control command timed out: {' '.join(arguments)}"
            stderr = f"{stderr.rstrip()}\n{marker}" if stderr.strip() else marker
            return subprocess.CompletedProcess(
                ["lxc", *arguments],
                124,
                stdout,
                stderr,
            )

    @staticmethod
    def _infrastructure_result(started: float, error: str) -> SandboxResult:
        return SandboxResult(
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=time.monotonic() - started,
            execution_seconds=0.0,
            infrastructure_error=error.strip() or "unknown LXD error",
        )
