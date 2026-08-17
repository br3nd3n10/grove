from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from grove.models import Task
from grove.runtime import GroveRuntime
from grove.store import GroveStore


class LongitudinalBenchmark:
    """Records capability, stability, plasticity, and growth at each checkpoint."""

    def __init__(self, store: GroveStore, runtime: GroveRuntime) -> None:
        self.store = store
        self.runtime = runtime

    def evaluate(
        self,
        cohorts: Mapping[str, Sequence[Task]],
        *,
        label: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or f"benchmark_{uuid.uuid4().hex[:12]}"
        cohort_metrics: dict[str, dict[str, Any]] = {}
        total_passed = 0
        total_tasks = 0
        total_routed = 0
        for cohort, tasks in cohorts.items():
            results = self.runtime.run(tasks, run_id=run_id, record=False)
            passed = sum(result.verification.passed for result in results)
            routed = sum(result.route.expert_id is not None for result in results)
            task_outcomes = {
                result.task.id: result.verification.passed for result in results
            }
            cohort_metrics[cohort] = {
                "tasks": len(results),
                "passed": passed,
                "pass_rate": passed / len(results) if results else 0.0,
                "expert_route_rate": routed / len(results) if results else 0.0,
                "task_outcomes": task_outcomes,
            }
            total_passed += passed
            total_tasks += len(results)
            total_routed += routed

        # Lifecycle state and deployment membership are intentionally distinct:
        # an active expert can be temporarily unplugged by a rollback manifest.
        experts = self.store.routable_experts()
        metrics: dict[str, Any] = {
            "capability": total_passed / total_tasks if total_tasks else 0.0,
            "tasks": total_tasks,
            "passed": total_passed,
            "expert_route_rate": total_routed / total_tasks if total_tasks else 0.0,
            "active_experts": len(experts),
            "added_parameters": sum(
                int(expert.artifact.get("parameter_count", 0)) for expert in experts
            ),
            "cohorts": cohort_metrics,
        }
        evaluation_id = self.store.record_evaluation(
            run_id=run_id, label=label, cohort="all", metrics=metrics
        )
        # The row id is what binds a report back to the database. Discarding it
        # left a report that named no evidence and could only be matched by
        # label and recency.
        return {**metrics, "evaluation_id": evaluation_id, "run_id": run_id}

    def curve(self) -> list[dict[str, Any]]:
        """Longitudinal checkpoints, each labelled with its evidence status.

        A curve reads rows straight out of SQLite. A row written before digest
        recording existed cannot be checked, and presenting it silently
        alongside verified rows would let unverifiable history support a
        capability claim. Each checkpoint therefore carries
        ``evidence: "verified" | "unverified"``, and the caller can tell them
        apart without re-querying the store.
        """
        evaluations = self.store.evaluations()
        if not evaluations:
            return []
        baseline = evaluations[0]["metrics"]
        baseline_cohorts = baseline.get("cohorts", {})
        baseline_passed = self._passed_regression_tasks(baseline_cohorts)
        curve: list[dict[str, Any]] = []
        for index, evaluation in enumerate(evaluations):
            metrics = evaluation["metrics"]
            cohorts = metrics.get("cohorts", {})
            current_regression = self._regression_task_outcomes(cohorts)
            stability_rates = [
                cohort["pass_rate"]
                for name, cohort in cohorts.items()
                if name.startswith("regression")
            ]
            plasticity_rates = [
                cohort["pass_rate"]
                for name, cohort in cohorts.items()
                if name.startswith("plasticity")
            ]
            baseline_plasticity = [
                cohort["pass_rate"]
                for name, cohort in baseline_cohorts.items()
                if name.startswith("plasticity")
            ]
            stability = (
                sum(stability_rates) / len(stability_rates) if stability_rates else None
            )
            plasticity = (
                sum(plasticity_rates) / len(plasticity_rates)
                if plasticity_rates
                else None
            )
            plasticity_base = (
                sum(baseline_plasticity) / len(baseline_plasticity)
                if baseline_plasticity
                else None
            )
            forgetting = self._forgetting_rate(baseline_passed, current_regression)
            curve.append(
                {
                    "checkpoint": index,
                    "label": evaluation["label"],
                    "capability": metrics["capability"],
                    "plasticity": plasticity,
                    "plasticity_gain": (
                        plasticity - plasticity_base
                        if plasticity is not None and plasticity_base is not None
                        else None
                    ),
                    "stability": stability,
                    "forgetting": forgetting,
                    # Cohorts are evaluated through the router, so this number
                    # describes the deployed system, not any single adapter. An
                    # adapter that regresses a task the router never sends it
                    # still shows zero here. Adapter-intrinsic forgetting lives
                    # in expert.metrics["forced_regression_rate"].
                    "forgetting_scope": "routed",
                    # "verified" means this exact row still matches the digest
                    # written beside it. "unverified" means the row predates
                    # digest recording: readable history, not evidence.
                    "evidence": self._evidence_state(evaluation),
                    "active_experts": metrics["active_experts"],
                    "added_parameters": metrics["added_parameters"],
                }
            )
        return curve

    def _evidence_state(self, evaluation: Mapping[str, Any]) -> str:
        """Whether this exact stored row still matches its digest."""
        integrity = self.store.verify_evaluations()
        if evaluation["id"] in integrity["tampered_evaluation_ids"]:
            return "tampered"
        if evaluation["id"] in integrity["unhashed_evaluation_ids"]:
            return "unverified"
        return "verified"

    @staticmethod
    def _regression_task_outcomes(
        cohorts: Mapping[str, dict[str, Any]],
    ) -> dict[str, bool]:
        outcomes: dict[str, bool] = {}
        for name, cohort in cohorts.items():
            if not name.startswith("regression"):
                continue
            task_outcomes = cohort.get("task_outcomes")
            if isinstance(task_outcomes, dict):
                outcomes.update(
                    {task_id: bool(passed) for task_id, passed in task_outcomes.items()}
                )
        return outcomes

    @classmethod
    def _passed_regression_tasks(
        cls, cohorts: Mapping[str, dict[str, Any]]
    ) -> set[str]:
        return {
            task_id
            for task_id, passed in cls._regression_task_outcomes(cohorts).items()
            if passed
        }

    @staticmethod
    def _forgetting_rate(
        baseline_passed: set[str], current_outcomes: Mapping[str, bool]
    ) -> float | None:
        if not baseline_passed:
            return None
        if any(task_id not in current_outcomes for task_id in baseline_passed):
            return None
        lost = sum(not current_outcomes[task_id] for task_id in baseline_passed)
        return lost / len(baseline_passed)
