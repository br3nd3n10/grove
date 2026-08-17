from __future__ import annotations

import json
import sqlite3

import pytest

from grove.benchmark import LongitudinalBenchmark
from grove.demo import DemoMathBackend
from grove.models import (
    DatasetRole,
    Expert,
    ExpertStatus,
    RouteDecision,
    Task,
    Verification,
)
from grove.runtime import GroveRuntime
from grove.sandbox import SandboxResult
from grove.store import GroveStore
from grove.verifiers import PythonCase, PythonSuite, SandboxedPythonVerifier


class FakeSandbox:
    def __init__(self, output):
        self.output = output
        self.payload = None

    def run_python(self, source, payload=None):
        self.payload = payload
        return SandboxResult(
            exit_code=0,
            stdout=json.dumps(self.output),
            stderr="",
            duration_seconds=0.01,
            metadata={"network_attached": False},
        )


def test_hidden_python_verifier_keeps_expected_values_on_host():
    task = Task("code_1", "Implement solve(payload)", verifier="sandboxed_python")
    sandbox = FakeSandbox([3, 7])
    verifier = SandboxedPythonVerifier(
        sandbox,  # type: ignore[arg-type]
        [
            PythonSuite(
                task.id,
                (PythonCase({"value": 2}, 3), PythonCase({"value": 6}, 7)),
            )
        ],
    )

    result = verifier.verify(task, "def solve(payload): return payload['value'] + 1")

    assert result.passed
    assert result.score == 1.0
    assert sandbox.payload == [{"value": 2}, {"value": 6}]
    assert "expected_hash" in result.details


def test_partial_hidden_failure_is_not_accepted():
    task = Task("code_2", "Implement solve(payload)", verifier="sandboxed_python")
    verifier = SandboxedPythonVerifier(
        FakeSandbox([3, 99]),  # type: ignore[arg-type]
        [
            PythonSuite(
                task.id,
                (PythonCase({"value": 2}, 3), PythonCase({"value": 6}, 7)),
            )
        ],
    )

    result = verifier.verify(task, "def solve(payload): return payload['value'] + 1")

    assert not result.passed
    assert result.score == 0.5


def test_verified_correction_and_split_are_auditable(tmp_path):
    with GroveStore(tmp_path / "grove.db") as store:
        task = Task("task_1", "broken", verifier="exact", expected="fixed")
        store.record_attempt(
            task=task,
            run_id="run",
            response="broken",
            verification=Verification(False, 0.0, "failed"),
            route=RouteDecision(None, 0.0, "base"),
        )
        failure = store.unresolved_failures()[0]
        correction = store.record_correction(
            failure_id=failure.id,
            response="fixed",
            source="canonical",
            verification=Verification(True, 1.0, "verified"),
            provenance={"suite": "v1"},
        )
        example = store.assign_dataset_role(
            task_id=task.id,
            role=DatasetRole.TRAIN,
            content='{"prompt":"broken","completion":"fixed"}',
            failure_id=failure.id,
            correction_id=correction.id,
        )

        assert correction.accepted
        assert store.unresolved_failures()[0].correction == "fixed"
        assert example.role is DatasetRole.TRAIN
        with pytest.raises(ValueError, match="dataset leakage"):
            store.assign_dataset_role(
                task_id=task.id,
                role=DatasetRole.TARGET,
                content='{"prompt":"broken","completion":"fixed"}',
            )
        with pytest.raises(ValueError, match="already assigned"):
            store.assign_dataset_role(
                task_id=task.id,
                role=DatasetRole.TARGET,
                content="different serialization of the same task",
            )


def test_deployment_manifests_are_append_only_and_rollback_creates_new_version(
    tmp_path,
):
    with GroveStore(tmp_path / "grove.db") as store:
        first = store.publish_deployment(
            base_model_revision="base@one",
            expert_ids=(),
            router_version="router-1",
            verifier_suite_version="suite-1",
            decoding_config={"temperature": 0},
            reason="baseline",
        )
        second = store.publish_deployment(
            base_model_revision="base@one",
            expert_ids=("expert-a",),
            router_version="router-2",
            verifier_suite_version="suite-1",
            decoding_config={"temperature": 0},
            reason="admit expert-a",
        )
        rollback = store.rollback_to(first.id, "regression detected")

        assert first.sequence == 1
        assert second.sequence == 2
        assert rollback.sequence == 3
        assert rollback.expert_ids == ()
        assert store.current_deployment().id == rollback.id  # type: ignore[union-attr]
        assert len(store.deployments()) == 3


def test_training_role_rejects_unverified_content(tmp_path):
    with GroveStore(tmp_path / "grove.db") as store:
        task = Task("task_unverified", "prompt")
        store.save_task(task)

        with pytest.raises(ValueError, match="verifier-backed correction"):
            store.assign_dataset_role(
                task_id=task.id,
                role=DatasetRole.TRAIN,
                content=task.prompt,
            )


def test_growth_counter_follows_deployment_not_lifecycle_state(tmp_path):
    with GroveStore(tmp_path / "grove.db") as store:
        expert = Expert(
            id="expert_active_but_unplugged",
            name="test",
            status=ExpertStatus.ACTIVE,
            artifact={"parameter_count": 123},
            routing_profile={},
            born_from=(),
        )
        store.save_expert(expert)
        store.publish_deployment(
            base_model_revision="base",
            expert_ids=(),
            router_version="router",
            verifier_suite_version="suite",
            decoding_config={},
            reason="rollback",
        )
        benchmark = LongitudinalBenchmark(
            store, GroveRuntime(store, DemoMathBackend())
        )

        metrics = benchmark.evaluate({}, label="unplugged")

        assert metrics["active_experts"] == 0
        assert metrics["added_parameters"] == 0


# --------------------------------------------------------------------------
# Finding 7: local integrity for the SQLite evidence store
# --------------------------------------------------------------------------


def test_evaluation_rows_carry_a_recomputable_digest(tmp_path):
    with GroveStore(tmp_path / "grove.db") as store:
        store.record_evaluation(
            run_id="run", label="baseline", cohort="all", metrics={"capability": 0.5}
        )

        verdict = store.verify_evaluations()

        assert verdict["status"] == "clean"
        assert verdict["authoritative"] is True
        assert verdict["checked"] == 1
        assert verdict["tampered_evaluation_ids"] == []
        assert verdict["unhashed_evaluation_ids"] == []


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("id", "eval_rebound"),
        ("run_id", "edited-run"),
        ("label", "edited-label"),
        ("cohort", "edited-cohort"),
        ("created_at", "2099-01-01T00:00:00+00:00"),
    ],
)
def test_evaluation_digest_binds_identity_and_selectors(
    tmp_path, column: str, value: str
):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="run", label="rollback", cohort="all", metrics={"capability": 0.5}
        )

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(f"UPDATE evaluations SET {column} = ?", (value,))
    connection.close()

    with GroveStore(database) as store:
        verdict = store.verify_evaluations()

    assert verdict["status"] == "tampered"
    assert verdict["authoritative"] is False
    assert len(verdict["tampered_evaluation_ids"]) == 1


def test_legacy_evaluation_digest_schema_is_unverified(tmp_path):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="run", label="rollback", cohort="all", metrics={"capability": 0.5}
        )

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "ALTER TABLE evaluations DROP COLUMN evaluation_digest_schema"
        )
    connection.close()

    with GroveStore(database) as store:
        verdict = store.verify_evaluations()

    assert verdict["status"] == "unverified"
    assert verdict["authoritative"] is False
    assert verdict["checked"] == 0
    assert len(verdict["unhashed_evaluation_ids"]) == 1


def test_editing_db_metrics_breaks_the_stored_digest(tmp_path):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        evaluation_id = store.record_evaluation(
            run_id="run", label="baseline", cohort="all", metrics={"capability": 0.5}
        )

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "UPDATE evaluations SET metrics_json = ?", ('{"capability":1.0}',)
        )
    connection.close()

    with GroveStore(database) as store:
        verdict = store.verify_evaluations()

    assert verdict["intact"] is False
    assert verdict["tampered_evaluation_ids"] == [evaluation_id]


def test_ledger_hash_chain_is_intact_for_an_untouched_store(tmp_path):
    with GroveStore(tmp_path / "grove.db") as store:
        store.record_evaluation(
            run_id="run", label="baseline", cohort="all", metrics={"capability": 0.5}
        )
        store.record_evaluation(
            run_id="run", label="after", cohort="all", metrics={"capability": 0.9}
        )

        verdict = store.verify_ledger()

    assert verdict["intact"] is True
    assert verdict["checked"] == 2
    assert verdict["broken"] == []


def test_ledger_hash_chain_detects_row_rewrite(tmp_path):
    """An append-only table is only append-only if editing a row is visible."""
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="run", label="baseline", cohort="all", metrics={"capability": 0.5}
        )
        store.record_evaluation(
            run_id="run", label="after", cohort="all", metrics={"capability": 0.9}
        )

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "UPDATE ledger SET payload_json = ? WHERE sequence = 1",
            ('{"capability":1.0}',),
        )
    connection.close()

    with GroveStore(database) as store:
        verdict = store.verify_ledger()

    assert verdict["intact"] is False
    assert verdict["broken"][0] == {"sequence": 1, "reason": "payload edited"}
    # The rewrite invalidates every later link too, which is the point of a chain.
    assert [item["sequence"] for item in verdict["broken"]] == [1, 2]


def test_ledger_hash_chain_detects_a_deleted_row(tmp_path):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        for index in range(3):
            store.record_evaluation(
                run_id="run",
                label=f"checkpoint_{index}",
                cohort="all",
                metrics={"capability": index / 10},
            )

    connection = sqlite3.connect(database)
    with connection:
        connection.execute("DELETE FROM ledger WHERE sequence = 2")
    connection.close()

    with GroveStore(database) as store:
        verdict = store.verify_ledger()

    assert verdict["intact"] is False
    assert verdict["broken"][0]["reason"] == "chain reordered"


# --------------------------------------------------------------------------
# Finding 4: absent evidence is unverified, never clean
# --------------------------------------------------------------------------


def _unhash_evaluations(database) -> None:
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("UPDATE evaluations SET metrics_sha256 = ''")
    connection.close()


def _unhash_ledger(database, sequence: int) -> None:
    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "UPDATE ledger SET entry_sha256 = '', payload_sha256 = '',"
            " previous_sha256 = '' WHERE sequence = ?",
            (sequence,),
        )
    connection.close()


def test_store_reports_unhashed_evaluation_rows_unverified(tmp_path):
    """``intact = not tampered`` called a database with no digests clean.

    The real 2026-07-31 store has four evaluation rows, none of them hashed. It
    verified with ``checked: 0`` and ``intact: true``. Nothing was checked.
    """
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="run", label="baseline", cohort="all", metrics={"capability": 0.5}
        )
    _unhash_evaluations(database)

    with GroveStore(database) as store:
        verdict = store.verify_evaluations()

    assert verdict["status"] == "unverified"
    assert verdict["authoritative"] is False
    assert verdict["intact"] is False
    assert verdict["checked"] == 0
    assert len(verdict["unhashed_evaluation_ids"]) == 1


def test_store_reports_unhashed_ledger_rows_unverified(tmp_path):
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="run", label="baseline", cohort="all", metrics={"capability": 0.5}
        )
    _unhash_ledger(database, 1)

    with GroveStore(database) as store:
        verdict = store.verify_ledger()

    assert verdict["status"] == "unverified"
    assert verdict["authoritative"] is False
    assert verdict["unhashed_sequences"] == [1]


def test_ledger_chain_does_not_cross_an_unhashed_row_as_clean(tmp_path):
    """A hole in a chain does not heal by restarting from an empty digest."""
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        for index in range(3):
            store.record_evaluation(
                run_id="run",
                label=f"checkpoint_{index}",
                cohort="all",
                metrics={"capability": index / 10},
            )
    _unhash_ledger(database, 2)

    with GroveStore(database) as store:
        verdict = store.verify_ledger()

    assert verdict["status"] == "unverified"
    assert verdict["authoritative"] is False
    assert verdict["unhashed_sequences"] == [2, 3]
    # Row 3 hashes against its own recorded predecessor, but nothing ties that
    # predecessor back to the start of the chain.
    assert verdict["unanchored_sequences"] == [3]
    assert verdict["broken"] == []


def test_unverified_evidence_cannot_be_called_clean(tmp_path):
    """The three states never collapse into two."""
    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        store.record_evaluation(
            run_id="run", label="baseline", cohort="all", metrics={"capability": 0.5}
        )
        clean = store.verify_evaluations()
    _unhash_evaluations(database)
    with GroveStore(database) as store:
        unverified = store.verify_evaluations()

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "UPDATE evaluations SET metrics_sha256 = ?", ("f" * 64,)
        )
    connection.close()
    with GroveStore(database) as store:
        tampered = store.verify_evaluations()

    assert [clean["status"], unverified["status"], tampered["status"]] == [
        "clean",
        "unverified",
        "tampered",
    ]
    assert [
        clean["authoritative"],
        unverified["authoritative"],
        tampered["authoritative"],
    ] == [True, False, False]


def test_curve_does_not_consume_unverified_legacy_evidence(tmp_path):
    """A checkpoint built from an unhashable row says so on the checkpoint."""
    from grove.demo import DemoMathBackend
    from grove.runtime import GroveRuntime

    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        runtime = GroveRuntime(store, DemoMathBackend())
        benchmark = LongitudinalBenchmark(store, runtime)
        benchmark.evaluate({}, label="baseline")
        assert [point["evidence"] for point in benchmark.curve()] == ["verified"]

    _unhash_evaluations(database)

    with GroveStore(database) as store:
        runtime = GroveRuntime(store, DemoMathBackend())
        curve = LongitudinalBenchmark(store, runtime).curve()

    assert [point["evidence"] for point in curve] == ["unverified"]


def test_curve_marks_an_edited_row_as_tampered(tmp_path):
    from grove.demo import DemoMathBackend
    from grove.runtime import GroveRuntime

    database = tmp_path / "grove.db"
    with GroveStore(database) as store:
        runtime = GroveRuntime(store, DemoMathBackend())
        LongitudinalBenchmark(store, runtime).evaluate({}, label="baseline")

    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "UPDATE evaluations SET metrics_json = ?",
            ('{"capability":1.0,"active_experts":0,"added_parameters":0,"cohorts":{}}',),
        )
    connection.close()

    with GroveStore(database) as store:
        runtime = GroveRuntime(store, DemoMathBackend())
        curve = LongitudinalBenchmark(store, runtime).curve()

    assert [point["evidence"] for point in curve] == ["tampered"]
