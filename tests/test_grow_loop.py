from __future__ import annotations

from dataclasses import replace

import pytest

from grove.benchmark import LongitudinalBenchmark
from grove.demo import (
    DemoMathBackend,
    DemoMathTrainer,
    demo_benchmark_cohorts,
    demo_live_tasks,
)
from grove.models import Cluster, Expert, ExpertStatus, Failure
from grove.runtime import GroveRuntime
from grove.sleep import SleepCycle
from grove.store import GroveStore


@pytest.fixture
def system(tmp_path):
    store = GroveStore(tmp_path / "grove.db")
    runtime = GroveRuntime(store, DemoMathBackend())
    yield store, runtime
    store.close()


def test_runtime_captures_only_verified_failures(system):
    store, runtime = system
    results = runtime.run(demo_live_tasks(), run_id="capture")

    assert sum(result.verification.passed for result in results) == 4
    failures = store.unresolved_failures()
    assert len(failures) == 4
    assert {failure.task.metadata["operation"] for failure in failures} == {"subtract"}
    assert all(failure.correction is not None for failure in failures)


def test_sleep_cycle_admits_expert_without_regression(system):
    store, runtime = system
    benchmark = LongitudinalBenchmark(store, runtime)
    baseline = benchmark.evaluate(demo_benchmark_cohorts(), label="baseline")
    runtime.run(demo_live_tasks(), run_id="capture")

    report = SleepCycle(store, runtime, DemoMathTrainer()).run()
    grown = benchmark.evaluate(demo_benchmark_cohorts(), label="grown")

    assert len(report.experts_admitted) == 1
    assert report.experts_rejected == ()
    assert baseline["cohorts"]["plasticity_new_skill"]["pass_rate"] == 0.0
    assert grown["cohorts"]["plasticity_new_skill"]["pass_rate"] == 1.0
    assert grown["cohorts"]["regression_known_skills"]["pass_rate"] == 1.0
    assert store.summary()["unresolved_failures"] == 0
    assert benchmark.curve()[-1]["forgetting"] == 0.0


def test_probation_rejects_router_regression(system):
    store, runtime = system
    runtime.run(demo_live_tasks(), run_id="capture")

    class BroadTrainer(DemoMathTrainer):
        def train(self, cluster, candidate_id):
            candidate = super().train(cluster, candidate_id)
            return replace(
                candidate,
                routing_profile={"tags": ["arithmetic"], "keywords": [], "tokens": {}},
            )

    report = SleepCycle(store, runtime, BroadTrainer()).run()

    assert report.experts_admitted == ()
    assert len(report.experts_rejected) == 1
    rejected = store.get_expert(report.experts_rejected[0])
    assert rejected.status is ExpertStatus.REJECTED
    assert rejected.metrics["regression_rate"] > 0
    assert store.summary()["unresolved_failures"] == 4


def test_demand_gate_waits_for_recurring_failure(system):
    store, runtime = system
    subtraction = [
        task for task in demo_live_tasks() if task.metadata["operation"] == "subtract"
    ]
    runtime.run(subtraction[:2], run_id="too_small")

    report = SleepCycle(store, runtime, DemoMathTrainer()).run()

    assert report.candidates_trained == 0
    assert report.skipped[0]["reason"].startswith("needs 3 failures")


def test_removed_expert_is_immediately_unplugged(system):
    store, runtime = system
    runtime.run(demo_live_tasks(), run_id="capture")
    cycle = SleepCycle(store, runtime, DemoMathTrainer())
    report = cycle.run()
    expert_id = report.experts_admitted[0]

    before = runtime.run(demo_benchmark_cohorts()["plasticity_new_skill"], record=False)
    removed = cycle.remove_expert(expert_id, "test rollback")
    after = runtime.run(demo_benchmark_cohorts()["plasticity_new_skill"], record=False)

    assert all(result.verification.passed for result in before)
    assert not any(result.verification.passed for result in after)
    assert removed.status is ExpertStatus.REMOVED
    assert store.summary()["active_experts"] == 0
    assert [event["event_type"] for event in store.ledger()[-2:]] == [
        "expert.removed",
        "deployment.published",
    ]


def _active_expert(expert_id: str) -> Expert:
    return Expert(
        id=expert_id,
        name=f"{expert_id}-expert",
        status=ExpertStatus.ACTIVE,
        artifact={"backend": "demo-math", "operation": "subtract"},
        routing_profile={"tags": [], "keywords": [], "tokens": {}},
        born_from=(),
    )


def test_remove_expert_does_not_redeploy_unplugged_active_expert(system):
    """Finding 10: removal rebuilt the manifest from lifecycle status.

    An expert can be lifecycle-active and deliberately unplugged, which is what
    a rollback manifest does. Deriving the replacement deployment from status
    plugged it straight back in, so removing one expert silently deployed
    another.
    """
    store, runtime = system
    store.save_expert(_active_expert("expert_a"))
    store.save_expert(_active_expert("expert_b"))
    store.publish_deployment(
        base_model_revision="base@1",
        expert_ids=("expert_b",),
        router_version="profile-router-v1",
        verifier_suite_version="suite-v1",
        decoding_config={"temperature": 0.0},
        reason="only b is plugged in; a is active but unplugged",
    )
    cycle = SleepCycle(store, runtime, DemoMathTrainer())

    cycle.remove_expert("expert_b", "unplug the only deployed expert")

    current = store.current_deployment()
    assert current is not None
    assert current.expert_ids == ()
    # expert_a is still lifecycle-active; removal must not deploy it.
    assert {expert.id for expert in store.experts(ExpertStatus.ACTIVE)} == {"expert_a"}


def test_remove_expert_keeps_the_other_deployed_members(system):
    store, runtime = system
    store.save_expert(_active_expert("expert_a"))
    store.save_expert(_active_expert("expert_b"))
    store.publish_deployment(
        base_model_revision="base@1",
        expert_ids=("expert_a", "expert_b"),
        router_version="profile-router-v1",
        verifier_suite_version="suite-v1",
        decoding_config={"temperature": 0.0},
        reason="both deployed",
    )
    cycle = SleepCycle(store, runtime, DemoMathTrainer())

    cycle.remove_expert("expert_b", "retire b")

    current = store.current_deployment()
    assert current is not None
    assert current.expert_ids == ("expert_a",)

def test_cluster_pairing_key_ignores_correction_source():
    from grove.sleep import cluster_pairing_key

    tasks = [
        replace(task, expected=f"answer-{source}")
        for source, task in enumerate(demo_live_tasks()[:2])
    ]
    failures = tuple(
        Failure(
            id=f"failure_{task.id}",
            attempt_id=f"attempt_{task.id}",
            task=task,
            response="wrong",
            correction=task.expected,
            fingerprint="subtract",
        )
        for task in tasks
    )
    canonical = Cluster("cluster", "subtract", failures, {})
    self_repair = Cluster(
        "cluster",
        "subtract",
        tuple(replace(failure, correction="model repair") for failure in failures),
        {},
    )

    assert cluster_pairing_key(canonical) == cluster_pairing_key(self_repair)


def test_cluster_pairing_key_changes_when_training_task_ids_change():
    from grove.sleep import cluster_pairing_key

    tasks = demo_live_tasks()[:3]

    def cluster(selected):
        return Cluster(
            "cluster",
            "subtract",
            tuple(
                Failure(
                    id=f"failure_{task.id}",
                    attempt_id=f"attempt_{task.id}",
                    task=task,
                    response="wrong",
                    correction=task.expected,
                    fingerprint="subtract",
                )
                for task in selected
            ),
            {},
        )

    assert cluster_pairing_key(cluster(tasks[:2])) != cluster_pairing_key(
        cluster(tasks[1:])
    )


def test_remove_expert_without_a_deployment_keeps_other_active_experts(system):
    store, runtime = system
    store.save_expert(_active_expert("expert_a"))
    store.save_expert(_active_expert("expert_b"))
    cycle = SleepCycle(store, runtime, DemoMathTrainer())

    cycle.remove_expert("expert_a", "retire a")

    current = store.current_deployment()
    assert current is not None
    assert current.expert_ids == ("expert_b",)
    assert {expert.id for expert in store.experts(ExpertStatus.ACTIVE)} == {"expert_b"}


def test_remove_expert_preserves_current_deployment_pins(system):
    store, runtime = system
    store.save_expert(_active_expert("expert_a"))
    store.save_expert(_active_expert("expert_b"))
    store.publish_deployment(
        base_model_revision="base@pinned",
        expert_ids=("expert_a", "expert_b"),
        router_version="router@pinned",
        verifier_suite_version="verifier@pinned",
        decoding_config={"temperature": 0.17, "max_tokens": 321},
        reason="pinned deployment",
    )
    cycle = SleepCycle(
        store,
        runtime,
        DemoMathTrainer(),
        base_model_revision="different-base",
        router_version="different-router",
        verifier_suite_version="different-verifier",
        decoding_config={"temperature": 0.99},
    )

    cycle.remove_expert("expert_a", "retire a")

    current = store.current_deployment()
    assert current is not None
    assert current.expert_ids == ("expert_b",)
    assert current.base_model_revision == "base@pinned"
    assert current.router_version == "router@pinned"
    assert current.verifier_suite_version == "verifier@pinned"
    assert current.decoding_config == {"temperature": 0.17, "max_tokens": 321}
