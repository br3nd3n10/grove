from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from grove.models import Cluster, Expert, Failure, Task, Verification


class ModelBackend(Protocol):
    """Produces a response with the frozen base or one removable expert."""

    def generate(self, task: Task, expert: Expert | None = None) -> str: ...


class ExpertTrainer(Protocol):
    """Trains an isolated candidate; it must not mutate active experts or the base."""

    def train(self, cluster: Cluster, candidate_id: str) -> Expert: ...


class Verifier(Protocol):
    def verify(self, task: Task, response: str) -> Verification: ...


class Clusterer(Protocol):
    def cluster(self, failures: Sequence[Failure]) -> list[Cluster]: ...


class Router(Protocol):
    def route(self, task: Task, experts: Sequence[Expert]): ...
