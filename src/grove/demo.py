from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Sequence
from typing import ClassVar

from grove.models import Cluster, Expert, ExpertStatus, Task


class DemoMathBackend:
    """A frozen toy base used to exercise the real Grove control plane."""

    _operations: ClassVar = {
        "add": operator.add,
        "multiply": operator.mul,
        "subtract": operator.sub,
        "divide": operator.truediv,
    }
    _base_capabilities: ClassVar = {"add", "multiply"}

    def generate(self, task: Task, expert: Expert | None = None) -> str:
        operation = str(task.metadata.get("operation", ""))
        if expert is None:
            if operation not in self._base_capabilities:
                return "unsupported"
        elif expert.artifact.get("operation") != operation:
            return "misrouted"
        function = self._operations.get(operation)
        operands = task.metadata.get("operands")
        if function is None or not isinstance(operands, list) or len(operands) != 2:
            return "unsupported"
        value = function(operands[0], operands[1])
        return str(
            int(value) if isinstance(value, float) and value.is_integer() else value
        )


class DemoMathTrainer:
    """Represents isolated adapter training with a deterministic demo artifact."""

    _keywords: ClassVar = {
        "add": ["add", "plus", "sum"],
        "multiply": ["multiply", "times", "product"],
        "subtract": ["subtract", "minus", "difference"],
        "divide": ["divide", "quotient", "over"],
    }

    def train(self, cluster: Cluster, candidate_id: str) -> Expert:
        operations = Counter(
            str(failure.task.metadata.get("operation", "unknown"))
            for failure in cluster.failures
        )
        operation, examples = operations.most_common(1)[0]
        return Expert(
            id=candidate_id,
            name=f"{operation}-expert",
            status=ExpertStatus.CANDIDATE,
            artifact={
                "backend": "demo-math",
                "operation": operation,
                "parameter_count": 1,
                "training_examples": examples,
            },
            routing_profile={
                "tags": [operation],
                "keywords": self._keywords.get(operation, [operation]),
                # Keep the demo route deliberately narrow. Real trainers should fit
                # this profile against both cluster positives and replay negatives.
                "tokens": {},
            },
            born_from=tuple(failure.id for failure in cluster.failures),
            metrics={"training_examples": examples},
        )


def _math_task(
    task_id: str,
    prompt: str,
    answer: int,
    operation: str,
    operands: tuple[int, int],
    *,
    cohort: str,
) -> Task:
    return Task(
        id=task_id,
        prompt=prompt,
        expected=str(answer),
        verifier="numeric",
        cohort=cohort,
        tags=("arithmetic", operation),
        metadata={
            "operation": operation,
            "operands": list(operands),
            "failure_type": operation,
        },
    )


def demo_live_tasks() -> list[Task]:
    return [
        _math_task("live_add_1", "Add 2 plus 5", 7, "add", (2, 5), cohort="live"),
        _math_task("live_add_2", "What is 9 plus 4?", 13, "add", (9, 4), cohort="live"),
        _math_task(
            "live_mul_1", "Multiply 3 times 7", 21, "multiply", (3, 7), cohort="live"
        ),
        _math_task(
            "live_mul_2", "What is 6 times 8?", 48, "multiply", (6, 8), cohort="live"
        ),
        _math_task(
            "live_sub_1", "Subtract 3 from 10", 7, "subtract", (10, 3), cohort="live"
        ),
        _math_task(
            "live_sub_2", "What is 12 minus 5?", 7, "subtract", (12, 5), cohort="live"
        ),
        _math_task(
            "live_sub_3",
            "Find the difference: 20 minus 8",
            12,
            "subtract",
            (20, 8),
            cohort="live",
        ),
        _math_task(
            "live_sub_4", "Subtract 11 from 30", 19, "subtract", (30, 11), cohort="live"
        ),
    ]


def demo_benchmark_cohorts() -> dict[str, Sequence[Task]]:
    return {
        "regression_known_skills": [
            _math_task(
                "reg_add_1",
                "What is 18 plus 7?",
                25,
                "add",
                (18, 7),
                cohort="regression",
            ),
            _math_task(
                "reg_add_2", "Add 41 plus 9", 50, "add", (41, 9), cohort="regression"
            ),
            _math_task(
                "reg_mul_1",
                "What is 11 times 3?",
                33,
                "multiply",
                (11, 3),
                cohort="regression",
            ),
            _math_task(
                "reg_mul_2",
                "Find the product of 9 times 8",
                72,
                "multiply",
                (9, 8),
                cohort="regression",
            ),
        ],
        "plasticity_new_skill": [
            _math_task(
                "new_sub_1",
                "What is 100 minus 44?",
                56,
                "subtract",
                (100, 44),
                cohort="plasticity",
            ),
            _math_task(
                "new_sub_2",
                "Subtract 17 from 50",
                33,
                "subtract",
                (50, 17),
                cohort="plasticity",
            ),
            _math_task(
                "new_sub_3",
                "Find the difference: 91 minus 6",
                85,
                "subtract",
                (91, 6),
                cohort="plasticity",
            ),
        ],
    }
