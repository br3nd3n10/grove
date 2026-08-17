from __future__ import annotations

import json

import pytest
from test_sandbox_paths import sandbox_host_targets

from grove.sandbox import LxdSandbox, SandboxPolicy

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sandbox() -> LxdSandbox:
    instance = LxdSandbox(SandboxPolicy(timeout_seconds=2, maximum_output_bytes=32_768))
    instance.preflight()
    return instance


def test_executes_json_program(sandbox: LxdSandbox):
    result = sandbox.run_python(
        """import json, sys
payload = json.load(sys.stdin)
print(json.dumps({"answer": payload["left"] + payload["right"]}))
""",
        {"left": 2, "right": 5},
    )

    assert result.clean
    assert json.loads(result.stdout) == {"answer": 7}


def test_has_no_network_interface(sandbox: LxdSandbox):
    result = sandbox.run_python(
        """import json, socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.5)
except OSError:
    print(json.dumps({"blocked": True}))
else:
    print(json.dumps({"blocked": False}))
"""
    )

    assert result.clean
    assert json.loads(result.stdout) == {"blocked": True}


def test_cannot_read_host_home(sandbox: LxdSandbox):
    target_paths = json.dumps([str(path) for path in sandbox_host_targets()])
    result = sandbox.run_python(
        f"""import json, pathlib
targets = [pathlib.Path(path) for path in {target_paths}]
print(json.dumps({{"visible": [str(path) for path in targets if path.exists()]}}))
"""
    )

    assert result.clean
    assert json.loads(result.stdout) == {"visible": []}


def test_timeout_terminates_infinite_program(sandbox: LxdSandbox):
    result = sandbox.run_python("while True: pass")

    assert result.timed_out
    # duration_seconds spans `lxc launch` through teardown, so it tracks host
    # load rather than the policy. The 2026-08-06 audit saw it reach 23.0s
    # against a `< 20` bound with the same 2s policy. execution_seconds measures
    # only the window the guest program was allowed to run, which is the number
    # the policy actually governs.
    poll_interval_seconds = 0.05
    assert result.execution_seconds >= sandbox.policy.timeout_seconds
    assert result.execution_seconds <= (
        sandbox.policy.timeout_seconds + 10 * poll_interval_seconds
    )
    # The program really ran to the deadline rather than dying at launch.
    assert result.duration_seconds >= result.execution_seconds


def test_output_limit_terminates_flood(sandbox: LxdSandbox):
    result = sandbox.run_python("while True: print('x' * 1000, flush=True)")

    assert result.output_limited
    assert len(result.stdout.encode()) <= 32_768
