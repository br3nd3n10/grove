from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from grove.models import Task, Verification
from grove.sandbox import LxdSandbox, SandboxResult


class ExactVerifier:
    def verify(self, task: Task, response: str) -> Verification:
        if task.expected is None:
            return Verification(False, 0.0, "task has no expected answer")
        passed = response.strip() == task.expected.strip()
        return Verification(
            passed,
            1.0 if passed else 0.0,
            "exact match" if passed else "response did not match expected answer",
            {"expected": task.expected},
        )


class NumericVerifier:
    def __init__(self, tolerance: float = 1e-9) -> None:
        self.tolerance = tolerance

    def verify(self, task: Task, response: str) -> Verification:
        if task.expected is None:
            return Verification(False, 0.0, "task has no expected answer")
        try:
            actual = float(response.strip())
            expected = float(task.expected.strip())
        except ValueError:
            return Verification(False, 0.0, "response was not numeric")
        passed = math.isclose(actual, expected, abs_tol=self.tolerance)
        return Verification(passed, 1.0 if passed else 0.0, "numeric comparison")


class JsonVerifier:
    def verify(self, task: Task, response: str) -> Verification:
        if task.expected is None:
            return Verification(False, 0.0, "task has no expected JSON")
        try:
            actual = json.loads(response)
            expected = json.loads(task.expected)
        except json.JSONDecodeError as error:
            return Verification(False, 0.0, f"invalid JSON: {error.msg}")
        passed = actual == expected
        return Verification(
            passed, 1.0 if passed else 0.0, "structural JSON comparison"
        )


@dataclass(frozen=True, slots=True)
class PythonCase:
    payload: Any
    expected: Any


@dataclass(frozen=True, slots=True)
class PythonSuite:
    task_id: str
    cases: tuple[PythonCase, ...]
    version: str = "v1"


class SandboxedPythonVerifier:
    """Evaluates a model-written solve(payload) function with host-held cases."""

    def __init__(self, sandbox: LxdSandbox, suites: list[PythonSuite]) -> None:
        self.sandbox = sandbox
        self.suites = {suite.task_id: suite for suite in suites}

    def verify(self, task: Task, response: str) -> Verification:
        suite = self.suites.get(task.id)
        if suite is None:
            return Verification(False, 0.0, "no hidden Python suite registered")
        if not response.strip():
            return Verification(False, 0.0, "empty Python response")
        payloads = [case.payload for case in suite.cases]
        result = self.sandbox.run_python(self._executable_source(response), payloads)
        if not result.clean:
            return Verification(
                False,
                0.0,
                self._sandbox_reason(result),
                self._details(suite, result),
            )
        try:
            outputs = json.loads(result.stdout)
        except json.JSONDecodeError:
            return Verification(
                False,
                0.0,
                "candidate did not produce one valid JSON result",
                self._details(suite, result),
            )
        if not isinstance(outputs, list) or len(outputs) != len(suite.cases):
            return Verification(
                False,
                0.0,
                "candidate returned the wrong number of case results",
                self._details(suite, result),
            )
        passed = [
            actual == case.expected
            for actual, case in zip(outputs, suite.cases, strict=True)
        ]
        score = sum(passed) / len(passed) if passed else 0.0
        details = self._details(suite, result)
        details.update(
            {
                "cases": len(passed),
                "cases_passed": sum(passed),
                "case_pass_bitmap": passed,
            }
        )
        return Verification(
            all(passed),
            score,
            "all hidden cases passed"
            if all(passed)
            else "one or more hidden cases failed",
            details,
        )

    @staticmethod
    def _executable_source(response: str) -> str:
        return (
            response.rstrip()
            + "\n\n"
            + "if __name__ == '__main__':\n"
            + "    import json as _grove_json, sys as _grove_sys\n"
            + "    _grove_inputs = _grove_json.load(_grove_sys.stdin)\n"
            + "    _grove_outputs = [solve(item) for item in _grove_inputs]\n"
            + "    print(_grove_json.dumps(_grove_outputs, separators=(',', ':')))\n"
        )

    @staticmethod
    def _sandbox_reason(result: SandboxResult) -> str:
        if result.infrastructure_error:
            return f"sandbox infrastructure error: {result.infrastructure_error}"
        if result.timed_out:
            return "candidate exceeded wall-clock limit"
        if result.output_limited:
            return "candidate exceeded output limit"
        return f"candidate exited with status {result.exit_code}"

    @staticmethod
    def _details(suite: PythonSuite, result: SandboxResult) -> dict[str, Any]:
        expected_hash = hashlib.sha256(
            json.dumps(
                [case.expected for case in suite.cases],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "suite_version": suite.version,
            "expected_hash": expected_hash,
            "duration_seconds": result.duration_seconds,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "output_limited": result.output_limited,
            "sandbox": result.metadata,
            "stderr_tail": result.stderr[-2000:],
        }


class VerifierRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, object] = {
            "exact": ExactVerifier(),
            "numeric": NumericVerifier(),
            "json": JsonVerifier(),
        }

    def register(self, name: str, verifier: object) -> None:
        self._verifiers[name] = verifier

    def verify(self, task: Task, response: str) -> Verification:
        verifier = self._verifiers.get(task.verifier)
        if verifier is None:
            return Verification(False, 0.0, f"unknown verifier: {task.verifier}")
        return verifier.verify(task, response)  # type: ignore[attr-defined]
