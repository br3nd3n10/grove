from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence

from grove.interfaces import ModelBackend
from grove.models import AttemptResult, Expert, RouteDecision, Task
from grove.routing import ProfileRouter
from grove.store import GroveStore
from grove.verifiers import VerifierRegistry


class GroveRuntime:
    """Runs tasks through routing, inference, verification, and evidence capture."""

    def __init__(
        self,
        store: GroveStore,
        backend: ModelBackend,
        *,
        router: ProfileRouter | None = None,
        verifiers: VerifierRegistry | None = None,
    ) -> None:
        self.store = store
        self.backend = backend
        self.router = router or ProfileRouter()
        self.verifiers = verifiers or VerifierRegistry()

    def execute(
        self,
        task: Task,
        *,
        run_id: str | None = None,
        experts: Sequence[Expert] | None = None,
        force_expert: Expert | None = None,
        record: bool = True,
    ) -> AttemptResult:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        available = (
            list(experts) if experts is not None else self.store.routable_experts()
        )
        if force_expert is not None:
            route = RouteDecision(
                force_expert.id, 1.0, "expert forced for isolated evaluation"
            )
            selected = force_expert
        else:
            route = self.router.route(task, available)
            selected = next(
                (expert for expert in available if expert.id == route.expert_id), None
            )
        response = self.backend.generate(task, selected)
        verification = self.verifiers.verify(task, response)
        if record:
            attempt_id = self.store.record_attempt(
                task=task,
                run_id=run_id,
                response=response,
                verification=verification,
                route=route,
            )
        else:
            attempt_id = f"ephemeral_{uuid.uuid4().hex[:12]}"
        return AttemptResult(attempt_id, task, response, verification, route)

    def run(
        self,
        tasks: Iterable[Task],
        *,
        run_id: str | None = None,
        record: bool = True,
        experts: Sequence[Expert] | None = None,
        force_expert: Expert | None = None,
    ) -> list[AttemptResult]:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        task_list = list(tasks)
        generate_batch = getattr(self.backend, "generate_batch", None)
        if generate_batch is None:
            return [
                self.execute(
                    task,
                    run_id=run_id,
                    record=record,
                    experts=experts,
                    force_expert=force_expert,
                )
                for task in task_list
            ]

        available = (
            list(experts) if experts is not None else self.store.routable_experts()
        )
        routed: list[tuple[Task, RouteDecision, Expert | None]] = []
        groups: dict[str | None, list[tuple[int, Task]]] = defaultdict(list)
        for index, task in enumerate(task_list):
            if force_expert is not None:
                route = RouteDecision(
                    force_expert.id, 1.0, "expert forced for isolated evaluation"
                )
                selected = force_expert
            else:
                route = self.router.route(task, available)
                selected = next(
                    (expert for expert in available if expert.id == route.expert_id),
                    None,
                )
            routed.append((task, route, selected))
            groups[selected.id if selected else None].append((index, task))

        responses: list[str | None] = [None] * len(task_list)
        for expert_id, indexed_tasks in groups.items():
            selected = (
                force_expert
                if force_expert is not None and force_expert.id == expert_id
                else next(
                    (expert for expert in available if expert.id == expert_id), None
                )
            )
            batch_tasks = [task for _, task in indexed_tasks]
            generated = generate_batch(batch_tasks, selected)
            if len(generated) != len(batch_tasks):
                raise RuntimeError("model backend returned an incomplete batch")
            for (index, _), response in zip(indexed_tasks, generated, strict=True):
                responses[index] = response

        results = []
        for (task, route, _), response in zip(routed, responses, strict=True):
            assert response is not None
            verification = self.verifiers.verify(task, response)
            attempt_id = (
                self.store.record_attempt(
                    task=task,
                    run_id=run_id,
                    response=response,
                    verification=verification,
                    route=route,
                )
                if record
                else f"ephemeral_{uuid.uuid4().hex[:12]}"
            )
            results.append(
                AttemptResult(attempt_id, task, response, verification, route)
            )
        return results
