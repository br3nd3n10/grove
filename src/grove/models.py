from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    prompt: str
    expected: str | None = None
    verifier: str = "exact"
    cohort: str = "live"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class Verification:
    passed: bool
    score: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class ExpertStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    REJECTED = "rejected"
    REMOVED = "removed"


class DatasetRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TARGET = "target"
    REGRESSION = "regression"
    FUTURE = "future"


@dataclass(frozen=True, slots=True)
class Expert:
    id: str
    name: str
    status: ExpertStatus
    artifact: dict[str, Any]
    routing_profile: dict[str, Any]
    born_from: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    expert_id: str | None
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class AttemptResult:
    attempt_id: str
    task: Task
    response: str
    verification: Verification
    route: RouteDecision


@dataclass(frozen=True, slots=True)
class Failure:
    id: str
    attempt_id: str
    task: Task
    response: str
    correction: str | None
    fingerprint: str
    resolved_by: str | None = None


@dataclass(frozen=True, slots=True)
class Cluster:
    id: str
    label: str
    failures: tuple[Failure, ...]
    profile: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GateDecision:
    passed: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CycleReport:
    cycle_id: str
    clusters_seen: int
    candidates_trained: int
    experts_admitted: tuple[str, ...]
    experts_rejected: tuple[str, ...]
    skipped: tuple[dict[str, Any], ...]
    started_at: str
    finished_at: str


@dataclass(frozen=True, slots=True)
class Correction:
    id: str
    failure_id: str
    response: str
    source: str
    verification: Verification
    accepted: bool
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class DatasetExample:
    id: str
    task_id: str
    role: DatasetRole
    content_hash: str
    failure_id: str | None = None
    correction_id: str | None = None
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class DeploymentManifest:
    id: str
    sequence: int
    base_model_revision: str
    expert_ids: tuple[str, ...]
    router_version: str
    verifier_suite_version: str
    decoding_config: dict[str, Any]
    reason: str
    current: bool = True
    created_at: str = field(default_factory=utc_now)
