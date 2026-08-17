from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Self

from grove.models import (
    Artifact,
    Correction,
    DatasetExample,
    DatasetRole,
    DeploymentManifest,
    Expert,
    ExpertStatus,
    Failure,
    RouteDecision,
    Task,
    Verification,
    utc_now,
)


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load(value: str) -> Any:
    return json.loads(value)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


EVALUATION_DIGEST_SCHEMA = "grove-evaluation-v2"


def _evaluation_digest(
    *,
    evaluation_id: str,
    run_id: str,
    label: str,
    cohort: str,
    metrics_json: str,
    created_at: str,
) -> str:
    """Bind an evaluation's identity, selectors, metrics, and creation time."""
    return _digest(
        _dump(
            {
                "schema": EVALUATION_DIGEST_SCHEMA,
                "id": evaluation_id,
                "run_id": run_id,
                "label": label,
                "cohort": cohort,
                "metrics_json": metrics_json,
                "created_at": created_at,
            }
        )
    )


LEDGER_CHAIN_SCHEMA = "grove-ledger-chain-v1"


def _ledger_entry_digest(
    *,
    event_type: str,
    entity_id: str,
    payload_sha256: str,
    created_at: str,
    previous_sha256: str,
) -> str:
    """Hash one ledger entry together with the digest of the one before it."""
    return _digest(
        _dump(
            {
                "schema": LEDGER_CHAIN_SCHEMA,
                "event_type": event_type,
                "entity_id": entity_id,
                "payload_sha256": payload_sha256,
                "created_at": created_at,
                "previous_sha256": previous_sha256,
            }
        )
    )


# Three states, in precedence order. Collapsing "nothing disagrees" into
# "everything checks out" is what let a database with no digests at all report
# itself as intact.
EVIDENCE_STATES = ("tampered", "unverified", "clean")


def _evidence_verdict(
    *,
    checked: int,
    tampered: list,
    unhashed: list,
    tampered_key: str,
    unhashed_key: str,
    unhashed_reason: str,
) -> dict[str, Any]:
    """Report verified, missing and disagreeing evidence as distinct outcomes.

    ``authoritative`` is the single question a caller actually has: may this
    record be quoted as ground truth? Only fully hashed, fully intact evidence
    may. Absent evidence is readable history, not proof.
    """
    if tampered:
        status = "tampered"
        reason = f"{len(tampered)} record(s) disagree with their stored digest"
    elif unhashed:
        status = "unverified"
        reason = f"{unhashed_reason}: {len(unhashed)} record(s) cannot be checked"
    else:
        status = "clean"
        reason = f"{checked} record(s) verified against their stored digests"
    return {
        "status": status,
        "reason": reason,
        "checked": checked,
        tampered_key: tampered,
        unhashed_key: unhashed,
        # Retained for readers that only ask "did anything disagree?", but it is
        # deliberately not the answer to "is this evidence".
        "intact": status == "clean",
        "authoritative": status == "clean",
    }


class GroveStore:
    """SQLite event store: operational state plus an append-only audit ledger."""

    def __init__(self, path: str | Path = ".grove/grove.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self.initialize()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                expected TEXT,
                verifier TEXT NOT NULL,
                cohort TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL REFERENCES tasks(id),
                response TEXT NOT NULL,
                passed INTEGER NOT NULL,
                score REAL NOT NULL,
                verification_json TEXT NOT NULL,
                expert_id TEXT,
                route_score REAL NOT NULL,
                route_reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS failures (
                id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id),
                correction TEXT,
                fingerprint TEXT NOT NULL,
                resolved_by TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_json TEXT NOT NULL,
                routing_profile_json TEXT NOT NULL,
                born_from_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cycles (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                report_json TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                label TEXT NOT NULL,
                cohort TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                metrics_sha256 TEXT NOT NULL DEFAULT '',
                evaluation_digest_schema TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ledger (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL DEFAULT '',
                previous_sha256 TEXT NOT NULL DEFAULT '',
                entry_sha256 TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                failure_id TEXT NOT NULL REFERENCES failures(id),
                response TEXT NOT NULL,
                source TEXT NOT NULL,
                verification_json TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_examples (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(id),
                role TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                failure_id TEXT REFERENCES failures(id),
                correction_id TEXT REFERENCES corrections(id),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                size_bytes INTEGER NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS deployments (
                id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                base_model_revision TEXT NOT NULL,
                expert_ids_json TEXT NOT NULL,
                router_version TEXT NOT NULL,
                verifier_suite_version TEXT NOT NULL,
                decoding_config_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                current INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_attempts_task ON attempts(task_id);
            CREATE INDEX IF NOT EXISTS idx_attempts_passed ON attempts(passed);
            CREATE INDEX IF NOT EXISTS idx_failures_resolved ON failures(resolved_by);
            CREATE INDEX IF NOT EXISTS idx_experts_status ON experts(status);
            CREATE INDEX IF NOT EXISTS idx_corrections_failure ON corrections(failure_id);
            CREATE INDEX IF NOT EXISTS idx_dataset_role ON dataset_examples(role);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_deployments_current
                ON deployments(current) WHERE current = 1;
            """
        )
        self._add_missing_columns()
        self._connection.commit()

    # Integrity columns added after the first databases were written. An older
    # file opens without them, so add them rather than refuse the file; rows
    # written before the chain existed keep an empty digest and are reported as
    # unhashed rather than as verified.
    _INTEGRITY_COLUMNS: ClassVar[dict[str, tuple[tuple[str, str], ...]]] = {
        "evaluations": (
            ("metrics_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("evaluation_digest_schema", "TEXT NOT NULL DEFAULT ''"),
        ),
        "ledger": (
            ("payload_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("previous_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("entry_sha256", "TEXT NOT NULL DEFAULT ''"),
        ),
    }

    def _add_missing_columns(self) -> None:
        for table, columns in self._INTEGRITY_COLUMNS.items():
            existing = {
                row["name"]
                for row in self._connection.execute(f"PRAGMA table_info({table})")
            }
            for name, declaration in columns:
                if name not in existing:
                    self._connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                    )

    def save_task(self, task: Task) -> None:
        self._connection.execute(
            """
            INSERT INTO tasks
                (id, prompt, expected, verifier, cohort, tags_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                prompt=excluded.prompt, expected=excluded.expected,
                verifier=excluded.verifier, cohort=excluded.cohort,
                tags_json=excluded.tags_json, metadata_json=excluded.metadata_json
            """,
            (
                task.id,
                task.prompt,
                task.expected,
                task.verifier,
                task.cohort,
                _dump(task.tags),
                _dump(task.metadata),
                task.created_at,
            ),
        )
        self._connection.commit()

    def save_tasks(self, tasks: Iterable[Task]) -> None:
        for task in tasks:
            self.save_task(task)

    def get_task(self, task_id: str) -> Task:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown task: {task_id}")
        return self._task(row)

    def tasks(self, *, cohort: str | None = None) -> list[Task]:
        if cohort is None:
            rows = self._connection.execute(
                "SELECT * FROM tasks ORDER BY created_at, id"
            )
        else:
            rows = self._connection.execute(
                "SELECT * FROM tasks WHERE cohort = ? ORDER BY created_at, id",
                (cohort,),
            )
        return [self._task(row) for row in rows]

    def record_attempt(
        self,
        *,
        task: Task,
        run_id: str,
        response: str,
        verification: Verification,
        route: RouteDecision,
    ) -> str:
        self.save_task(task)
        attempt_id = f"attempt_{uuid.uuid4().hex[:16]}"
        self._connection.execute(
            """
            INSERT INTO attempts
                (id, run_id, task_id, response, passed, score, verification_json,
                 expert_id, route_score, route_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                run_id,
                task.id,
                response,
                int(verification.passed),
                verification.score,
                _dump({"reason": verification.reason, "details": verification.details}),
                route.expert_id,
                route.score,
                route.reason,
                utc_now(),
            ),
        )
        if not verification.passed:
            failure_id = f"failure_{uuid.uuid4().hex[:16]}"
            fingerprint = str(
                task.metadata.get("failure_type") or " ".join(task.tags) or task.prompt
            )
            self._connection.execute(
                """
                INSERT INTO failures
                    (id, attempt_id, correction, fingerprint, resolved_by, created_at)
                VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (failure_id, attempt_id, task.expected, fingerprint, utc_now()),
            )
            self._append_ledger(
                "failure.captured",
                failure_id,
                {
                    "attempt_id": attempt_id,
                    "task_id": task.id,
                    "fingerprint": fingerprint,
                },
            )
        self._connection.commit()
        return attempt_id

    def unresolved_failures(self) -> list[Failure]:
        rows = self._connection.execute(
            """
            SELECT f.id AS failure_id, f.attempt_id, f.correction, f.fingerprint,
                   f.resolved_by, a.response, t.*
            FROM failures f
            JOIN attempts a ON a.id = f.attempt_id
            JOIN tasks t ON t.id = a.task_id
            WHERE f.resolved_by IS NULL
            ORDER BY f.created_at, f.id
            """
        )
        return [
            Failure(
                id=row["failure_id"],
                attempt_id=row["attempt_id"],
                task=self._task(row),
                response=row["response"],
                correction=row["correction"],
                fingerprint=row["fingerprint"],
                resolved_by=row["resolved_by"],
            )
            for row in rows
        ]

    def record_correction(
        self,
        *,
        failure_id: str,
        response: str,
        source: str,
        verification: Verification,
        provenance: dict[str, Any] | None = None,
    ) -> Correction:
        correction = Correction(
            id=f"correction_{uuid.uuid4().hex[:16]}",
            failure_id=failure_id,
            response=response,
            source=source,
            verification=verification,
            accepted=verification.passed,
            provenance=provenance or {},
        )
        self._connection.execute(
            """
            INSERT INTO corrections
                (id, failure_id, response, source, verification_json, accepted,
                 provenance_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                correction.id,
                correction.failure_id,
                correction.response,
                correction.source,
                _dump(
                    {
                        "passed": correction.verification.passed,
                        "score": correction.verification.score,
                        "reason": correction.verification.reason,
                        "details": correction.verification.details,
                    }
                ),
                int(correction.accepted),
                _dump(correction.provenance),
                correction.created_at,
            ),
        )
        self._append_ledger(
            "correction.accepted" if correction.accepted else "correction.rejected",
            correction.id,
            {
                "failure_id": failure_id,
                "source": source,
                "score": verification.score,
            },
        )
        if correction.accepted:
            self._connection.execute(
                "UPDATE failures SET correction = ? WHERE id = ?",
                (correction.response, correction.failure_id),
            )
        self._connection.commit()
        return correction

    def corrections(self, *, accepted: bool | None = None) -> list[Correction]:
        if accepted is None:
            rows = self._connection.execute(
                "SELECT * FROM corrections ORDER BY created_at, id"
            )
        else:
            rows = self._connection.execute(
                "SELECT * FROM corrections WHERE accepted = ? ORDER BY created_at, id",
                (int(accepted),),
            )
        values: list[Correction] = []
        for row in rows:
            result = _load(row["verification_json"])
            values.append(
                Correction(
                    id=row["id"],
                    failure_id=row["failure_id"],
                    response=row["response"],
                    source=row["source"],
                    verification=Verification(
                        result["passed"],
                        result["score"],
                        result["reason"],
                        result["details"],
                    ),
                    accepted=bool(row["accepted"]),
                    provenance=_load(row["provenance_json"]),
                    created_at=row["created_at"],
                )
            )
        return values

    def assign_dataset_role(
        self,
        *,
        task_id: str,
        role: DatasetRole,
        content: str,
        failure_id: str | None = None,
        correction_id: str | None = None,
    ) -> DatasetExample:
        if role is DatasetRole.TRAIN:
            if failure_id is None or correction_id is None:
                raise ValueError(
                    "training examples require a verifier-backed correction"
                )
            correction = self._connection.execute(
                "SELECT failure_id, accepted FROM corrections WHERE id = ?",
                (correction_id,),
            ).fetchone()
            if (
                correction is None
                or not correction["accepted"]
                or correction["failure_id"] != failure_id
            ):
                raise ValueError(
                    "training examples require a matching accepted correction"
                )
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing_task = self._connection.execute(
            "SELECT role FROM dataset_examples WHERE task_id = ? LIMIT 1",
            (task_id,),
        ).fetchone()
        if existing_task is not None:
            raise ValueError(
                f"dataset leakage: task {task_id} is already assigned to "
                f"{existing_task['role']}"
            )
        existing = self._connection.execute(
            "SELECT role, task_id FROM dataset_examples WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if existing is not None:
            raise ValueError(
                f"dataset leakage: content already assigned to {existing['role']} "
                f"for task {existing['task_id']}"
            )
        example = DatasetExample(
            id=f"example_{uuid.uuid4().hex[:16]}",
            task_id=task_id,
            role=role,
            content_hash=content_hash,
            failure_id=failure_id,
            correction_id=correction_id,
        )
        self._connection.execute(
            """
            INSERT INTO dataset_examples
                (id, task_id, role, content_hash, failure_id, correction_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                example.id,
                example.task_id,
                example.role.value,
                example.content_hash,
                example.failure_id,
                example.correction_id,
                example.created_at,
            ),
        )
        self._append_ledger(
            "dataset.assigned",
            example.id,
            {"task_id": task_id, "role": role.value, "content_hash": content_hash},
        )
        self._connection.commit()
        return example

    def dataset_examples(self, role: DatasetRole | None = None) -> list[DatasetExample]:
        if role is None:
            rows = self._connection.execute(
                "SELECT * FROM dataset_examples ORDER BY created_at, id"
            )
        else:
            rows = self._connection.execute(
                "SELECT * FROM dataset_examples WHERE role = ? ORDER BY created_at, id",
                (role.value,),
            )
        return [
            DatasetExample(
                id=row["id"],
                task_id=row["task_id"],
                role=DatasetRole(row["role"]),
                content_hash=row["content_hash"],
                failure_id=row["failure_id"],
                correction_id=row["correction_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def save_artifact(self, artifact: Artifact) -> None:
        self._connection.execute(
            """
            INSERT INTO artifacts
                (id, kind, path, sha256, size_bytes, provenance_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.kind,
                artifact.path,
                artifact.sha256,
                artifact.size_bytes,
                _dump(artifact.provenance),
                artifact.created_at,
            ),
        )
        self._append_ledger(
            "artifact.saved",
            artifact.id,
            {
                "kind": artifact.kind,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            },
        )
        self._connection.commit()

    def successful_tasks(
        self,
        *,
        limit: int | None = None,
        exclude_fingerprints: Iterable[str] | None = None,
    ) -> list[Task]:
        sql = """
            SELECT t.* FROM tasks t
            JOIN attempts a ON a.task_id = t.id
            WHERE a.passed = 1
              AND NOT EXISTS (
                  SELECT 1 FROM attempts newer
                  WHERE newer.task_id = a.task_id
                    AND (
                        newer.created_at > a.created_at
                        OR (newer.created_at = a.created_at AND newer.id > a.id)
                    )
              )
            ORDER BY t.created_at, t.id
        """
        excluded = set(exclude_fingerprints or ())
        tasks = [self._task(row) for row in self._connection.execute(sql)]
        if excluded:
            tasks = [
                task for task in tasks if self._task_fingerprint(task) not in excluded
            ]
        return tasks[:limit] if limit is not None else tasks

    @staticmethod
    def _task_fingerprint(task: Task) -> str:
        return str(
            task.metadata.get("failure_type") or " ".join(task.tags) or task.prompt
        )

    def save_expert(self, expert: Expert, *, event: str = "expert.saved") -> None:
        self._connection.execute(
            """
            INSERT INTO experts
                (id, name, status, artifact_json, routing_profile_json, born_from_json,
                 metrics_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, status=excluded.status,
                artifact_json=excluded.artifact_json,
                routing_profile_json=excluded.routing_profile_json,
                born_from_json=excluded.born_from_json,
                metrics_json=excluded.metrics_json, updated_at=excluded.updated_at
            """,
            (
                expert.id,
                expert.name,
                expert.status.value,
                _dump(expert.artifact),
                _dump(expert.routing_profile),
                _dump(expert.born_from),
                _dump(expert.metrics),
                expert.created_at,
                expert.updated_at,
            ),
        )
        self._append_ledger(
            event,
            expert.id,
            {
                "status": expert.status.value,
                "metrics": expert.metrics,
                "born_from": expert.born_from,
            },
        )
        self._connection.commit()

    def experts(self, status: ExpertStatus | None = None) -> list[Expert]:
        if status is None:
            rows = self._connection.execute(
                "SELECT * FROM experts ORDER BY created_at, id"
            )
        else:
            rows = self._connection.execute(
                "SELECT * FROM experts WHERE status = ? ORDER BY created_at, id",
                (status.value,),
            )
        return [self._expert(row) for row in rows]

    def routable_experts(self) -> list[Expert]:
        manifest = self.current_deployment()
        if manifest is None:
            return self.experts(ExpertStatus.ACTIVE)
        allowed = set(manifest.expert_ids)
        return [
            expert
            for expert in self.experts(ExpertStatus.ACTIVE)
            if expert.id in allowed
        ]

    def get_expert(self, expert_id: str) -> Expert:
        row = self._connection.execute(
            "SELECT * FROM experts WHERE id = ?", (expert_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown expert: {expert_id}")
        return self._expert(row)

    def mark_failures_resolved(
        self, failure_ids: Iterable[str], expert_id: str
    ) -> None:
        ids = tuple(failure_ids)
        self._connection.executemany(
            "UPDATE failures SET resolved_by = ? WHERE id = ?",
            ((expert_id, failure_id) for failure_id in ids),
        )
        self._append_ledger(
            "failures.resolved", expert_id, {"failure_ids": ids, "count": len(ids)}
        )
        self._connection.commit()

    def start_cycle(self) -> str:
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        self._connection.execute(
            "INSERT INTO cycles (id, status, started_at) VALUES (?, 'running', ?)",
            (cycle_id, now),
        )
        self._append_ledger("cycle.started", cycle_id, {})
        self._connection.commit()
        return cycle_id

    def finish_cycle(self, cycle_id: str, report: dict[str, Any]) -> None:
        self._connection.execute(
            "UPDATE cycles SET status = 'finished', report_json = ?, finished_at = ? WHERE id = ?",
            (_dump(report), utc_now(), cycle_id),
        )
        self._append_ledger("cycle.finished", cycle_id, report)
        self._connection.commit()

    def publish_deployment(
        self,
        *,
        base_model_revision: str,
        expert_ids: Iterable[str],
        router_version: str,
        verifier_suite_version: str,
        decoding_config: dict[str, Any],
        reason: str,
    ) -> DeploymentManifest:
        sequence = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM deployments"
        ).fetchone()[0]
        manifest = DeploymentManifest(
            id=f"deployment_{uuid.uuid4().hex[:12]}",
            sequence=sequence,
            base_model_revision=base_model_revision,
            expert_ids=tuple(sorted(set(expert_ids))),
            router_version=router_version,
            verifier_suite_version=verifier_suite_version,
            decoding_config=decoding_config,
            reason=reason,
        )
        with self._connection:
            self._connection.execute(
                "UPDATE deployments SET current = 0 WHERE current = 1"
            )
            self._connection.execute(
                """
                INSERT INTO deployments
                    (id, sequence, base_model_revision, expert_ids_json, router_version,
                     verifier_suite_version, decoding_config_json, reason, current, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    manifest.id,
                    manifest.sequence,
                    manifest.base_model_revision,
                    _dump(manifest.expert_ids),
                    manifest.router_version,
                    manifest.verifier_suite_version,
                    _dump(manifest.decoding_config),
                    manifest.reason,
                    manifest.created_at,
                ),
            )
            self._append_ledger(
                "deployment.published",
                manifest.id,
                {
                    "sequence": sequence,
                    "expert_ids": manifest.expert_ids,
                    "reason": reason,
                },
            )
        return manifest

    def current_deployment(self) -> DeploymentManifest | None:
        row = self._connection.execute(
            "SELECT * FROM deployments WHERE current = 1"
        ).fetchone()
        return self._deployment(row) if row is not None else None

    def deployments(self) -> list[DeploymentManifest]:
        rows = self._connection.execute("SELECT * FROM deployments ORDER BY sequence")
        return [self._deployment(row) for row in rows]

    def rollback_to(self, deployment_id: str, reason: str) -> DeploymentManifest:
        row = self._connection.execute(
            "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown deployment: {deployment_id}")
        target = self._deployment(row)
        return self.publish_deployment(
            base_model_revision=target.base_model_revision,
            expert_ids=target.expert_ids,
            router_version=target.router_version,
            verifier_suite_version=target.verifier_suite_version,
            decoding_config=target.decoding_config,
            reason=f"rollback to {deployment_id}: {reason}",
        )

    def record_evaluation(
        self, *, run_id: str, label: str, cohort: str, metrics: dict[str, Any]
    ) -> str:
        evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"
        payload = _dump(metrics)
        created_at = utc_now()
        self._connection.execute(
            """
            INSERT INTO evaluations
                (id, run_id, label, cohort, metrics_json, metrics_sha256,
                 evaluation_digest_schema, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_id,
                run_id,
                label,
                cohort,
                payload,
                _evaluation_digest(
                    evaluation_id=evaluation_id,
                    run_id=run_id,
                    label=label,
                    cohort=cohort,
                    metrics_json=payload,
                    created_at=created_at,
                ),
                EVALUATION_DIGEST_SCHEMA,
                created_at,
            ),
        )
        self._append_ledger("evaluation.recorded", evaluation_id, metrics)
        self._connection.commit()
        return evaluation_id

    def evaluations(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM evaluations ORDER BY created_at, id"
        )
        return [
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "label": row["label"],
                "cohort": row["cohort"],
                "metrics": _load(row["metrics_json"]),
                "metrics_sha256": row["metrics_sha256"],
                "evaluation_digest_schema": row["evaluation_digest_schema"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def verify_evaluations(self) -> dict[str, Any]:
        """Recompute every stored evaluation digest.

        The digest commits to the row identity and selector fields as well as
        its metrics. Rows written before this digest schema existed remain
        readable, but their old metrics-only (or absent) digest is unverified.
        A digest is never backfilled because that cannot prove historical
        contents.
        """
        rows = self._connection.execute(
            "SELECT id, run_id, label, cohort, metrics_json, metrics_sha256, "
            "evaluation_digest_schema, created_at FROM evaluations ORDER BY id"
        )
        tampered: list[str] = []
        unhashed: list[str] = []
        checked = 0
        for row in rows:
            if (
                not row["metrics_sha256"]
                or row["evaluation_digest_schema"] != EVALUATION_DIGEST_SCHEMA
            ):
                unhashed.append(row["id"])
                continue
            checked += 1
            expected = _evaluation_digest(
                evaluation_id=row["id"],
                run_id=row["run_id"],
                label=row["label"],
                cohort=row["cohort"],
                metrics_json=row["metrics_json"],
                created_at=row["created_at"],
            )
            if expected != row["metrics_sha256"]:
                tampered.append(row["id"])
        return _evidence_verdict(
            checked=checked,
            tampered=tampered,
            unhashed=unhashed,
            tampered_key="tampered_evaluation_ids",
            unhashed_key="unhashed_evaluation_ids",
            unhashed_reason="evaluation rows carry no current-schema digest",
        )

    def ledger(self) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM ledger ORDER BY sequence")
        return [
            {
                "sequence": row["sequence"],
                "event_type": row["event_type"],
                "entity_id": row["entity_id"],
                "payload": _load(row["payload_json"]),
                "payload_sha256": row["payload_sha256"],
                "previous_sha256": row["previous_sha256"],
                "entry_sha256": row["entry_sha256"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def verify_ledger(self) -> dict[str, Any]:
        """Walk the ledger hash chain and name the first link that breaks.

        Each entry commits to its payload and to the previous entry's digest,
        so rewriting one row invalidates it and everything after it. This is a
        local chain, not an external anchor: someone who can rewrite the whole
        file can recompute the whole chain. It detects edits, not an attacker
        with full write access.

        An unhashed row breaks continuity rather than being skipped over. The
        walk used to reset ``previous`` to an empty string and carry on as if
        the chain resumed cleanly, so a stretch of unhashed rows -- exactly what
        a legacy database is -- verified as intact. A chain with a hole in it is
        ``unverified``, and every row after the hole is unverified with it:
        nothing local establishes where the chain restarted.
        """
        rows = list(self._connection.execute("SELECT * FROM ledger ORDER BY sequence"))
        broken: list[dict[str, Any]] = []
        unhashed: list[int] = []
        unanchored: list[int] = []
        previous = ""
        chain_broken_at: int | None = None
        checked = 0
        for row in rows:
            if not row["entry_sha256"]:
                unhashed.append(row["sequence"])
                if chain_broken_at is None:
                    chain_broken_at = row["sequence"]
                continue
            if chain_broken_at is not None:
                # Downstream of a hole. The row may hash correctly against its
                # own recorded predecessor, but nothing ties that predecessor
                # to the start of the chain.
                unanchored.append(row["sequence"])
                continue
            payload_digest = _digest(row["payload_json"])
            expected = _ledger_entry_digest(
                event_type=row["event_type"],
                entity_id=row["entity_id"],
                payload_sha256=payload_digest,
                created_at=row["created_at"],
                previous_sha256=previous,
            )
            if payload_digest != row["payload_sha256"]:
                broken.append({"sequence": row["sequence"], "reason": "payload edited"})
            elif row["previous_sha256"] != previous:
                broken.append({"sequence": row["sequence"], "reason": "chain reordered"})
            elif expected != row["entry_sha256"]:
                broken.append({"sequence": row["sequence"], "reason": "entry edited"})
            checked += 1
            # The recomputed digest, not the recorded one: a rewritten row must
            # invalidate every link after it, which is what a chain is for.
            previous = expected
        verdict = _evidence_verdict(
            checked=checked,
            tampered=broken,
            unhashed=[*unhashed, *unanchored],
            tampered_key="broken",
            unhashed_key="unhashed_sequences",
            unhashed_reason="the ledger chain has an unhashed link",
        )
        verdict["unanchored_sequences"] = unanchored
        return verdict

    def summary(self) -> dict[str, Any]:
        scalar = lambda sql: self._connection.execute(sql).fetchone()[0]
        return {
            "tasks": scalar("SELECT COUNT(*) FROM tasks"),
            "attempts": scalar("SELECT COUNT(*) FROM attempts"),
            "unresolved_failures": scalar(
                "SELECT COUNT(*) FROM failures WHERE resolved_by IS NULL"
            ),
            "active_experts": scalar(
                "SELECT COUNT(*) FROM experts WHERE status = 'active'"
            ),
            "rejected_experts": scalar(
                "SELECT COUNT(*) FROM experts WHERE status = 'rejected'"
            ),
            "removed_experts": scalar(
                "SELECT COUNT(*) FROM experts WHERE status = 'removed'"
            ),
            "cycles": scalar("SELECT COUNT(*) FROM cycles WHERE status = 'finished'"),
            "accepted_corrections": scalar(
                "SELECT COUNT(*) FROM corrections WHERE accepted = 1"
            ),
            "deployments": scalar("SELECT COUNT(*) FROM deployments"),
        }

    def _append_ledger(self, event_type: str, entity_id: str, payload: Any) -> None:
        encoded = _dump(payload)
        payload_sha256 = _digest(encoded)
        created_at = utc_now()
        previous = self._connection.execute(
            "SELECT entry_sha256 FROM ledger ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_sha256 = previous["entry_sha256"] if previous is not None else ""
        entry_sha256 = _ledger_entry_digest(
            event_type=event_type,
            entity_id=entity_id,
            payload_sha256=payload_sha256,
            created_at=created_at,
            previous_sha256=previous_sha256,
        )
        self._connection.execute(
            "INSERT INTO ledger (event_type, entity_id, payload_json, payload_sha256,"
            " previous_sha256, entry_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_type,
                entity_id,
                encoded,
                payload_sha256,
                previous_sha256,
                entry_sha256,
                created_at,
            ),
        )

    @staticmethod
    def _task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            prompt=row["prompt"],
            expected=row["expected"],
            verifier=row["verifier"],
            cohort=row["cohort"],
            tags=tuple(_load(row["tags_json"])),
            metadata=_load(row["metadata_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _expert(row: sqlite3.Row) -> Expert:
        return Expert(
            id=row["id"],
            name=row["name"],
            status=ExpertStatus(row["status"]),
            artifact=_load(row["artifact_json"]),
            routing_profile=_load(row["routing_profile_json"]),
            born_from=tuple(_load(row["born_from_json"])),
            metrics=_load(row["metrics_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _deployment(row: sqlite3.Row) -> DeploymentManifest:
        return DeploymentManifest(
            id=row["id"],
            sequence=row["sequence"],
            base_model_revision=row["base_model_revision"],
            expert_ids=tuple(_load(row["expert_ids_json"])),
            router_version=row["router_version"],
            verifier_suite_version=row["verifier_suite_version"],
            decoding_config=_load(row["decoding_config_json"]),
            reason=row["reason"],
            current=bool(row["current"]),
            created_at=row["created_at"],
        )
