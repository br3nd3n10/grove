from __future__ import annotations

from types import SimpleNamespace

from grove.benchmark import LongitudinalBenchmark
from grove.models import RouteDecision, Task, Verification
from grove.store import GroveStore

PASS = "1"
FAIL = "0"


def task(task_id: str) -> Task:
    return Task(
        id=task_id,
        prompt=f"solve {task_id}",
        expected=PASS,
        verifier="exact",
        cohort="regression",
    )


class ScriptedRuntime:
    def __init__(self, checkpoints: list[dict[str, bool]]) -> None:
        self.checkpoints = checkpoints
        self.index = 0

    def run(self, tasks, *, run_id=None, record=False):
        outcomes = self.checkpoints[self.index]
        self.index += 1
        return [
            SimpleNamespace(
                task=item,
                verification=Verification(
                    outcomes[item.id], 1.0 if outcomes[item.id] else 0.0, "scripted"
                ),
                route=RouteDecision(None, 0.0, "scripted"),
            )
            for item in tasks
        ]


def test_curve_counts_lost_baseline_passes_per_task_not_net_rate(tmp_path):
    store = GroveStore(tmp_path / "grove.db")
    try:
        tasks = [task("replay_a"), task("replay_b")]
        runtime = ScriptedRuntime(
            [
                {"replay_a": True, "replay_b": False},
                {"replay_a": False, "replay_b": True},
            ]
        )
        benchmark = LongitudinalBenchmark(store, runtime)

        baseline = benchmark.evaluate(
            {"regression_known": tasks}, label="baseline", run_id="baseline"
        )
        grown = benchmark.evaluate(
            {"regression_known": tasks}, label="grown", run_id="grown"
        )

        assert baseline["cohorts"]["regression_known"]["pass_rate"] == 0.5
        assert grown["cohorts"]["regression_known"]["pass_rate"] == 0.5
        assert baseline["cohorts"]["regression_known"]["task_outcomes"] == {
            "replay_a": True,
            "replay_b": False,
        }
        assert benchmark.curve()[-1]["forgetting"] == 1.0
        assert benchmark.curve()[-1]["forgetting_scope"] == "routed"
    finally:
        store.close()


def test_curve_reports_unmeasured_forgetting_when_baseline_has_no_passes(tmp_path):
    store = GroveStore(tmp_path / "grove.db")
    try:
        tasks = [task("replay_a"), task("replay_b")]
        runtime = ScriptedRuntime(
            [
                {"replay_a": False, "replay_b": False},
                {"replay_a": True, "replay_b": True},
            ]
        )
        benchmark = LongitudinalBenchmark(store, runtime)

        benchmark.evaluate({"regression_known": tasks}, label="baseline")
        benchmark.evaluate({"regression_known": tasks}, label="grown")

        assert benchmark.curve()[0]["forgetting"] is None
        assert benchmark.curve()[-1]["forgetting"] is None
    finally:
        store.close()
