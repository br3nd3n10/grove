"""Gates that separate router-shielded stability from adapter-intrinsic stability.

The 2026-08-06 audit named "no forgetting is router-shielded" the most serious
open risk: routed replay can report zero regression purely because the router
never sends a replay prompt to the candidate. The 2026-08-07 review then found
three ways the first fix could still lie -- net pass-rate cancellation, an empty
cohort reported as stable, and circular route recall. These tests pin all of it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from grove.demo import (
    DemoMathBackend,
    DemoMathTrainer,
    demo_benchmark_cohorts,
    demo_live_tasks,
)
from grove.models import Expert, ExpertStatus, RouteDecision, Task, Verification
from grove.runtime import GroveRuntime
from grove.sleep import SleepCycle, SleepPolicy
from grove.store import GroveStore

PASS = "1"
FAIL = "0"


# --------------------------------------------------------------------------
# Scripted harness: exact control over per-task outcomes before and after.
# --------------------------------------------------------------------------


def scripted_task(task_id: str, *, family: str = "widget", tags=("widget",)) -> Task:
    return Task(
        id=task_id,
        prompt=f"solve {task_id}",
        expected=PASS,
        verifier="exact",
        tags=tags,
        metadata={"failure_type": family},
    )


class ScriptedBackend:
    """Answers keyed by task id, split by whether an expert is switched on.

    ``phase`` lets a replay task pass at capture time and behave differently
    during probation, which is how a real drift actually shows up.
    """

    def __init__(
        self,
        *,
        capture: dict,
        base: dict,
        forced: dict,
        active: dict | None = None,
    ) -> None:
        self.capture = capture
        self.base = base
        self.forced = forced
        self.active = active or {}
        self.phase = "capture"

    def generate(self, task: Task, expert=None) -> str:
        if self.phase == "capture":
            return self.capture.get(task.id, FAIL)
        if expert is None:
            return self.base.get(task.id, FAIL)
        if expert.id.startswith("active"):
            return self.active.get(task.id, self.base.get(task.id, FAIL))
        return self.forced.get(task.id, FAIL)


class ScriptedTrainer:
    """Emits a candidate whose routing profile is fixed by the test.

    ``keywords`` is the only signal the oracle-free route probe can use: tags
    are stripped before routing because they are the gold labels the profile was
    fitted to. A test that wants the router to reach a cohort has to put a real
    prompt token in the profile.
    """

    def __init__(self, *, tags=("widget",), keywords=()) -> None:
        self.tags = list(tags)
        self.keywords = list(keywords)

    def train(self, cluster, candidate_id: str) -> Expert:
        return Expert(
            id=candidate_id,
            name="scripted-expert",
            status=ExpertStatus.CANDIDATE,
            artifact={"backend": "scripted", "parameter_count": 1},
            routing_profile={
                "tags": self.tags,
                "keywords": self.keywords,
                "tokens": {},
            },
            born_from=tuple(failure.id for failure in cluster.failures),
        )


def probation_metrics(
    tmp_path,
    *,
    replay_ids: list[str],
    base: dict,
    forced: dict,
    heldout: list[Task] | None = None,
    policy: SleepPolicy | None = None,
    profile_tags=("widget",),
    profile_keywords=(),
    replay_tags=(),
    replay_family="replay",
    active: dict | None = None,
    active_profile_tags=("replay",),
):
    """Drive one full cycle and hand back the candidate's recorded metrics."""
    train_tasks = [scripted_task(f"train_{index}") for index in range(3)]
    replay_tasks = [
        scripted_task(task_id, family=replay_family, tags=replay_tags)
        for task_id in replay_ids
    ]
    capture = {task.id: FAIL for task in train_tasks}
    capture.update({task.id: PASS for task in replay_tasks})
    backend = ScriptedBackend(capture=capture, base=base, forced=forced, active=active)

    store = GroveStore(tmp_path / "grove.db")
    try:
        if active is not None:
            store.save_expert(
                Expert(
                    id="active_replay_expert",
                    name="active replay expert",
                    status=ExpertStatus.ACTIVE,
                    artifact={"backend": "scripted", "parameter_count": 1},
                    routing_profile={
                        "tags": list(active_profile_tags),
                        "keywords": [],
                        "tokens": {},
                    },
                    born_from=(),
                )
            )
        runtime = GroveRuntime(store, backend)
        runtime.run([*replay_tasks, *train_tasks], run_id="capture")
        backend.phase = "probation"
        cycle = SleepCycle(
            store,
            runtime,
            ScriptedTrainer(tags=profile_tags, keywords=profile_keywords),
            policy=policy,
            heldout_targets={"widget": list(heldout or [])},
        )
        report = cycle.run()
        expert_id = (*report.experts_admitted, *report.experts_rejected)[0]
        expert = store.get_expert(expert_id)
        return report, expert
    finally:
        store.close()


# --------------------------------------------------------------------------
# Finding 1: per-task regression, never a net pass-rate difference.
# --------------------------------------------------------------------------


def test_a_new_pass_cannot_cancel_a_lost_pass(tmp_path):
    """Baseline [pass, fail] against forced [fail, pass] is a full regression.

    The net pass rate is unchanged at 0.5 in both runs. Counting per task, the
    one thing the adapter used to get right is now broken.
    """
    report, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a", "replay_b"],
        base={"replay_a": PASS, "replay_b": FAIL, **_all_train(PASS)},
        forced={"replay_a": FAIL, "replay_b": PASS, **_all_train(PASS)},
        policy=SleepPolicy(max_forced_regression_rate=0.0),
    )

    assert expert.metrics["forced_regression_rate"] == 1.0
    assert expert.metrics["forced_regression_denominator"] == 1
    assert expert.metrics["forced_regression_task_ids"] == ["replay_a"]
    # The net rate that the old arithmetic reported.
    assert expert.metrics["replay_pass_rate_before"] == 0.5
    assert expert.metrics["forced_replay_rate"] == 0.5
    assert report.experts_admitted == ()
    assert (
        "forced-adapter regression budget exceeded"
        in expert.metrics["rejection_reason"]
    )


def test_one_loss_out_of_two_prior_passes_is_a_half_rate(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a", "replay_b"],
        base={"replay_a": PASS, "replay_b": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, "replay_b": FAIL, **_all_train(PASS)},
    )

    assert expert.metrics["forced_regression_rate"] == 0.5
    assert expert.metrics["forced_regression_denominator"] == 2
    assert expert.metrics["forced_regression_task_ids"] == ["replay_b"]
    assert expert.metrics["forgetting_claim"] == "router_shielded"


def test_nothing_prior_passing_means_unmeasured_not_stable(tmp_path):
    """[fail, fail] -> [pass, pass] is an improvement, not proof of stability."""
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a", "replay_b"],
        base={"replay_a": FAIL, "replay_b": FAIL, **_all_train(PASS)},
        forced={"replay_a": PASS, "replay_b": PASS, **_all_train(PASS)},
    )

    assert expert.metrics["forced_regression_denominator"] == 0
    assert expert.metrics["forced_regression_rate"] is None
    assert expert.metrics["forced_replay_measured"] is False
    assert expert.metrics["forgetting_claim"] == "unmeasured"


def test_forgetting_claim_regression_outranks_router_shield_story(tmp_path):
    """Research case: previous routed deployment passed, candidate route fails."""
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": FAIL, **_all_train(PASS)},
        active={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": FAIL, **_all_train(PASS)},
        replay_tags=("replay",),
        profile_tags=("replay",),
    )

    assert expert.metrics["regression_rate"] == 1.0
    assert expert.metrics["forced_regression_rate"] == 1.0
    assert expert.metrics["forced_regression_reference"] == (
        "routed_before_with_active_experts"
    )
    assert expert.metrics["forgetting_claim"] == "regression"
    assert expert.metrics["forgetting_claim"] != "router_shielded"


def test_adapter_intrinsic_claim_requires_a_bare_base_reference(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, **_all_train(PASS)},
    )

    assert expert.metrics["forced_regression_reference"] == "base_no_experts"
    assert expert.metrics["forgetting_claim"] == "adapter_intrinsic"


def test_clean_forced_replay_with_only_routed_reference_is_unverified(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": FAIL, **_all_train(PASS)},
        active={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, **_all_train(PASS)},
        replay_tags=("replay",),
        profile_tags=("widget",),
    )

    assert expert.metrics["regression_rate"] == 0.0
    assert expert.metrics["forced_regression_rate"] == 0.0
    assert expert.metrics["forced_regression_reference"] == (
        "routed_before_with_active_experts"
    )
    assert expert.metrics["forgetting_claim"] == "unverified_reference"


def test_measured_bare_base_reference_with_no_prior_pass_is_unmeasured(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": FAIL, **_all_train(PASS)},
        active={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": FAIL, **_all_train(PASS)},
        policy=SleepPolicy(measure_base_reference=True),
        replay_tags=("replay",),
        profile_tags=("widget", "replay"),
    )

    assert expert.metrics["forced_regression_reference"] == "base_no_experts"
    assert expert.metrics["forced_regression_rate"] is None
    assert expert.metrics["forgetting_claim"] == "unmeasured"


def test_routed_and_forced_regression_are_recorded_against_named_scopes(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": FAIL, **_all_train(PASS)},
    )

    # Routed replay never reaches the candidate, so the deployed system is clean
    # while the adapter is not. Both numbers are kept, neither is conflated.
    assert expert.metrics["regression_rate"] == 0.0
    assert expert.metrics["forced_regression_rate"] == 1.0
    assert expert.metrics["regression_reference_scope"] == "base_no_experts"
    assert expert.metrics["active_experts_at_probation"] == 0
    assert expert.metrics["replay_task_ids"] == ["replay_a"]


def test_base_reference_is_opt_in_when_active_experts_exist(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        active={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, **_all_train(PASS)},
        policy=SleepPolicy(measure_base_reference=True),
        replay_tags=("replay",),
    )

    assert expert.metrics["base_reference_measured"] is True
    assert expert.metrics["base_replay_pass_rate"] == 1.0
    assert expert.metrics["forced_regression_reference"] == "base_no_experts"


# --------------------------------------------------------------------------
# Finding 2: an empty cohort is unmeasured, never stable.
# --------------------------------------------------------------------------


def test_empty_replay_cannot_report_adapter_intrinsic_stability(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=[],
        base=_all_train(PASS),
        forced=_all_train(PASS),
    )

    assert expert.metrics["replay_examples"] == 0
    assert expert.metrics["forced_replay_measured"] is False
    assert expert.metrics["forced_replay_rate"] is None
    assert expert.metrics["forced_regression_rate"] is None
    assert expert.metrics["forgetting_claim"] == "unmeasured"
    # No negative evidence exists, so the negative route metrics are unmeasured.
    assert expert.metrics["route_negatives"] == 0
    assert expert.metrics["route_false_positive_rate"] is None


def test_policy_can_demand_a_measured_replay_cohort(tmp_path):
    report, expert = probation_metrics(
        tmp_path,
        replay_ids=[],
        base=_all_train(PASS),
        forced=_all_train(PASS),
        policy=SleepPolicy(require_measured_replay=True),
    )

    assert report.experts_admitted == ()
    assert (
        "forced-adapter stability is unmeasured" in (expert.metrics["rejection_reason"])
    )


def test_declared_minimum_replay_cohort_is_enforced(tmp_path):
    report, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a", "replay_b"],
        base={"replay_a": PASS, "replay_b": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, "replay_b": PASS, **_all_train(PASS)},
        policy=SleepPolicy(min_replay_examples=50),
    )

    assert report.experts_admitted == ()
    assert (
        "prior-passing replay cohort below the declared minimum"
        in (expert.metrics["rejection_reason"])
    )


def test_forced_replay_measurement_is_explicitly_opt_out(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, **_all_train(PASS)},
        policy=SleepPolicy(measure_forced_replay=False),
    )

    assert expert.metrics["forced_replay_measured"] is False
    assert expert.metrics["forced_replay_rate"] is None
    assert expert.metrics["forgetting_claim"] == "unmeasured"


# --------------------------------------------------------------------------
# Finding 3: route recall must come from independent held-out evidence.
# --------------------------------------------------------------------------


def test_route_recall_uses_heldout_positives_not_the_training_cluster(tmp_path):
    """A profile fitted to the training prompts must not score itself on them."""
    heldout = [
        scripted_task(f"heldout_{index}", tags=("gadget",)) for index in range(2)
    ]

    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={
            "replay_a": PASS,
            **_all_train(PASS),
            **{task.id: PASS for task in heldout},
        },
        heldout=heldout,
        profile_keywords=("train_0", "train_1"),
    )

    assert expert.metrics["route_positive_source"] == "heldout_forced_pass_oracle_free"
    assert expert.metrics["route_probe_metadata"] == "oracle_free"
    # Three training failures exist; the denominator is the two held-outs.
    assert expert.metrics["target_examples"] == 3
    assert expert.metrics["route_positives"] == 2
    # The router matches the training prompts and misses the held-out family.
    assert expert.metrics["route_recall"] == 0.0
    assert "route recall below threshold" in expert.metrics["rejection_reason"]


def test_route_recall_strips_gold_family_tags(tmp_path):
    """The gold family tag is the label the routing profile was fitted to.

    ``ProfileRouter.score`` gives an exact tag overlap the maximum score, so a
    candidate whose only signal is the family tag used to report recall 1.0 on
    held-out tasks that still carried that tag. Nothing about routing was
    measured: two copies of the same label were compared. The prompts here share
    no useful token with the profile, so oracle-free recall is 0.0 and the
    gold-tag diagnostic still shows the 1.0 the old probe reported.
    """
    heldout = [
        scripted_task(f"heldout_{index}", tags=("widget",)) for index in range(2)
    ]

    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={
            "replay_a": PASS,
            **_all_train(PASS),
            **{task.id: PASS for task in heldout},
        },
        heldout=heldout,
        profile_tags=("widget",),
        profile_keywords=(),
    )

    assert expert.metrics["route_positives"] == 2
    assert expert.metrics["route_recall"] == 0.0
    assert expert.metrics["route_recall_gold_tags"] == 1.0
    assert "route recall below threshold" in expert.metrics["rejection_reason"]


def test_gold_tag_route_probe_is_not_independent_evidence(tmp_path):
    """The gold-tag number is reported, and reported as non-independent."""
    heldout = [scripted_task("heldout_0", tags=("widget",))]

    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, **_all_train(PASS), "heldout_0": PASS},
        heldout=heldout,
        profile_tags=("widget",),
    )

    assert expert.metrics["route_recall_gold_tags_independent"] is False
    assert expert.metrics["route_positive_source"].endswith("_oracle_free")


def test_route_recall_rewards_a_router_that_reaches_the_heldout_family(tmp_path):
    heldout = [
        scripted_task(f"heldout_{index}", tags=("gadget",)) for index in range(2)
    ]

    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={
            "replay_a": PASS,
            **_all_train(PASS),
            **{task.id: PASS for task in heldout},
        },
        heldout=heldout,
        # A prompt token, not the gold tag: this is routing the probe can trust.
        profile_keywords=("heldout_0", "heldout_1"),
    )

    assert expert.metrics["route_recall"] == 1.0
    assert expert.metrics["route_precision"] == 1.0
    assert expert.metrics["route_precision_cohort_dependent"] is True
    assert expert.metrics["route_false_positive_rate"] == 0.0


def test_absent_heldout_cohort_leaves_route_recall_unmeasured(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, **_all_train(PASS)},
    )

    assert expert.metrics["route_positives"] == 0
    assert expert.metrics["route_recall"] is None
    assert expert.metrics["route_recall_gold_tags"] is None
    assert expert.metrics["route_precision"] is None
    assert expert.metrics["route_precision_cohort_dependent"] is True


def test_min_route_precision_is_not_silently_discarded():
    """Finding 14: the keyword was accepted, dropped, and never gated anything.

    Route precision's denominator mixes the held-out positive cohort with the
    replay negative cohort, so it cannot be a threshold until a fixed cohort
    defines it. Refusing the keyword is honest; swallowing it let a policy claim
    a gate that did not exist.
    """
    with pytest.raises(TypeError, match="min_route_precision"):
        SleepPolicy(min_route_precision=1.0)

    assert not hasattr(SleepPolicy(), "min_route_precision")


def test_route_precision_is_reported_but_not_used_as_a_gate(tmp_path):
    heldout = [
        scripted_task(f"heldout_{index}", tags=("gadget",)) for index in range(3)
    ]

    report, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a", "replay_b"],
        base={"replay_a": PASS, "replay_b": PASS, **_all_train(PASS)},
        forced={
            "replay_a": PASS,
            "replay_b": PASS,
            **_all_train(PASS),
            **{task.id: PASS for task in heldout},
        },
        heldout=heldout,
        policy=SleepPolicy(
            min_plasticity_gain=0.0,
            max_route_false_positive_rate=1.0,
        ),
        # "solve" opens every scripted prompt, so this router grabs everything.
        profile_keywords=("solve",),
        replay_tags=("replay",),
    )

    assert expert.metrics["route_recall"] == 1.0
    assert expert.metrics["route_false_positive_rate"] == 1.0
    assert expert.metrics["route_precision"] == pytest.approx(3 / 5)
    assert expert.metrics["route_precision_cohort_dependent"] is True
    assert report.experts_admitted == (expert.id,)


def test_policy_can_demand_a_measured_route_recall(tmp_path):
    report, expert = probation_metrics(
        tmp_path,
        replay_ids=["replay_a"],
        base={"replay_a": PASS, **_all_train(PASS)},
        forced={"replay_a": PASS, **_all_train(PASS)},
        policy=SleepPolicy(require_measured_route_recall=True),
    )

    assert report.experts_admitted == ()
    assert "route recall is unmeasured" in expert.metrics["rejection_reason"]


def _all_train(answer: str) -> dict:
    return {f"train_{index}": answer for index in range(3)}


# --------------------------------------------------------------------------
# The deterministic demo, which is itself a router-shielded example.
# --------------------------------------------------------------------------


@pytest.fixture
def demo_system(tmp_path):
    store = GroveStore(tmp_path / "grove.db")
    runtime = GroveRuntime(store, DemoMathBackend())
    runtime.run(demo_live_tasks(), run_id="capture")
    yield store, runtime
    store.close()


def demo_heldout() -> dict:
    return {"subtract": demo_benchmark_cohorts()["plasticity_new_skill"]}


def test_forced_replay_exposes_forgetting_the_router_hides(demo_system):
    store, runtime = demo_system

    report = SleepCycle(
        store, runtime, DemoMathTrainer(), heldout_targets=demo_heldout()
    ).run()
    expert = store.get_expert(report.experts_admitted[0])

    # Deployed configuration: the router shields replay, so routed replay is clean.
    assert expert.metrics["regression_rate"] == 0.0
    assert expert.metrics["regression_denominator"] == 4
    # The adapter itself, measured with the shield off, is not clean at all.
    assert expert.metrics["forced_replay_measured"] is True
    assert expert.metrics["forced_regression_rate"] == 1.0
    assert set(expert.metrics["forced_regression_task_ids"]) == {
        "live_add_1",
        "live_add_2",
        "live_mul_1",
        "live_mul_2",
    }
    assert expert.metrics["forgetting_claim"] == "router_shielded"


def test_strict_forced_regression_policy_can_reject_the_demo_expert(demo_system):
    store, runtime = demo_system

    report = SleepCycle(
        store,
        runtime,
        DemoMathTrainer(),
        policy=SleepPolicy(max_forced_regression_rate=0.0),
        heldout_targets=demo_heldout(),
    ).run()

    assert report.experts_admitted == ()
    rejected = store.get_expert(report.experts_rejected[0])
    assert (
        "forced-adapter regression budget exceeded"
        in rejected.metrics["rejection_reason"]
    )
    assert store.summary()["unresolved_failures"] == 4


def test_demo_route_metrics_are_scored_on_held_out_subtraction(demo_system):
    store, runtime = demo_system

    report = SleepCycle(
        store, runtime, DemoMathTrainer(), heldout_targets=demo_heldout()
    ).run()
    expert = store.get_expert(report.experts_admitted[0])

    assert expert.metrics["route_positives"] == 3
    assert expert.metrics["route_negatives"] == 4
    assert expert.metrics["route_recall"] == 1.0
    assert expert.metrics["route_precision"] == 1.0
    assert expert.metrics["route_precision_cohort_dependent"] is True
    assert expert.metrics["route_false_positive_rate"] == 0.0


def test_router_that_claims_base_passing_replay_traffic_is_rejected(demo_system):
    store, runtime = demo_system

    class BroadTrainer(DemoMathTrainer):
        def train(self, cluster, candidate_id):
            candidate = super().train(cluster, candidate_id)
            return replace(
                candidate,
                # Prompt keywords wide enough to catch the add and multiply
                # replay cohort as well as the subtraction family. Gold tags
                # would not do it: the probe strips them before routing.
                routing_profile={
                    "tags": ["arithmetic"],
                    "keywords": [
                        "plus",
                        "times",
                        "minus",
                        "subtract",
                        "difference",
                        "what",
                    ],
                    "tokens": {},
                },
            )

    report = SleepCycle(
        store, runtime, BroadTrainer(), heldout_targets=demo_heldout()
    ).run()

    assert report.experts_admitted == ()
    rejected = store.get_expert(report.experts_rejected[0])
    assert rejected.metrics["route_false_positive_rate"] == 1.0
    assert rejected.metrics["route_precision"] == pytest.approx(3 / 7)
    assert rejected.metrics["route_precision_cohort_dependent"] is True
    assert (
        "router claims base-passing replay traffic"
        in rejected.metrics["rejection_reason"]
    )


# --------------------------------------------------------------------------
# Finding 10: replay negatives are latest-passing and family-independent.
# --------------------------------------------------------------------------


def test_successful_tasks_excludes_pass_then_fail_latest_attempt(tmp_path):
    store = GroveStore(tmp_path / "grove.db")
    try:
        task = scripted_task("replay", family="replay")
        store.record_attempt(
            task=task,
            run_id="first",
            response=PASS,
            verification=Verification(True, 1.0, "passed"),
            route=RouteDecision(None, 0.0, "base"),
        )
        store.record_attempt(
            task=task,
            run_id="second",
            response=FAIL,
            verification=Verification(False, 0.0, "failed"),
            route=RouteDecision(None, 0.0, "base"),
        )

        assert store.successful_tasks() == []
    finally:
        store.close()


def test_successful_tasks_includes_fail_then_pass_latest_attempt(tmp_path):
    store = GroveStore(tmp_path / "grove.db")
    try:
        task = scripted_task("replay", family="replay")
        store.record_attempt(
            task=task,
            run_id="first",
            response=FAIL,
            verification=Verification(False, 0.0, "failed"),
            route=RouteDecision(None, 0.0, "base"),
        )
        store.record_attempt(
            task=task,
            run_id="second",
            response=PASS,
            verification=Verification(True, 1.0, "passed"),
            route=RouteDecision(None, 0.0, "base"),
        )

        assert [task.id for task in store.successful_tasks()] == ["replay"]
    finally:
        store.close()


def test_successful_tasks_can_exclude_candidate_family(tmp_path):
    store = GroveStore(tmp_path / "grove.db")
    try:
        in_family = scripted_task("same_family", family="widget")
        out_family = scripted_task("replay", family="replay")
        for task in (in_family, out_family):
            store.record_attempt(
                task=task,
                run_id="capture",
                response=PASS,
                verification=Verification(True, 1.0, "passed"),
                route=RouteDecision(None, 0.0, "base"),
            )

        assert [
            task.id for task in store.successful_tasks(exclude_fingerprints={"widget"})
        ] == ["replay"]
    finally:
        store.close()


def test_sleep_cycle_excludes_in_family_successes_from_replay_negatives(tmp_path):
    _, expert = probation_metrics(
        tmp_path,
        replay_ids=["same_family"],
        base={"same_family": PASS, **_all_train(PASS)},
        forced={"same_family": FAIL, **_all_train(PASS)},
        replay_tags=("widget",),
        replay_family="widget",
    )

    assert expert.metrics["replay_examples"] == 0
    assert expert.metrics["replay_task_ids"] == []
    assert expert.metrics["route_negatives"] == 0
