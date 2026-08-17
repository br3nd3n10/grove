"""EXP-005: multi-cycle growth, coexistence measurement, and independent grading.

The second-cycle coexistence experiment is the first plural measurement in
this repository: every earlier routing and stability number was single-expert.
These tests pin four things before the run exists:

1. the second failure family is real -- verified canonical solutions, fresh
   held-outs, and content-hash disjointness from every cycle-1 prompt;
2. the multi-cycle configuration is spec-driven and refused when it cannot be
   honoured, before any cost;
3. the coexistence probe measures a two-expert deployment honestly -- per
   family, per expert, with denominators, and with ``None`` never read as zero;
4. the sealed spec grades every rule independently, so a measurable zero
   records as a falsification (exit 1), never as unusable (exit 2), which is
   the structural lesson of the EXP-004 research note.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from grove import experiment
from grove.coding_tasks import coding_catalog, second_cycle_catalog
from grove.models import Expert, ExpertStatus, Task
from grove.runtime import GroveRuntime
from grove.store import GroveStore
from grove.verifiers import VerifierRegistry

_ROOT = Path(__file__).parents[1]
_SPEC = _ROOT / "experiments" / "EXP-005-second-cycle-coexistence.json"
_SCRIPT = _ROOT / "scripts" / "run_exp005.sh"

_checker_spec = importlib.util.spec_from_file_location(
    "check_experiment_spec_exp005", _ROOT / "scripts" / "check_experiment_spec.py"
)
assert _checker_spec is not None and _checker_spec.loader is not None
checker = importlib.util.module_from_spec(_checker_spec)
_checker_spec.loader.exec_module(checker)


def _spec() -> dict:
    return json.loads(_SPEC.read_text())


def _machine(spec: dict) -> dict:
    return experiment.normalize_required_setup(spec["required_setup"])["machine"]


# ---------------------------------------------------------------------------
# The second failure family
# ---------------------------------------------------------------------------


def test_second_family_has_governed_disjoint_splits():
    second = second_cycle_catalog()
    by_role = {}
    for item in second:
        by_role.setdefault(item.role.value, []).append(item)

    assert len(by_role["train"]) == 20
    assert len(by_role["target"]) == 4
    for item in second:
        assert item.task.metadata["failure_type"] == "path_restructure"
        assert "path_restructure" in item.task.tags
        assert item.suite.version == "path-restructure-v1"
        assert item.task.expected is None
        assert item.suite.cases
    combined = coding_catalog() + second
    assert len({item.task.id for item in combined}) == len(combined)
    assert len({item.task.prompt for item in combined}) == len(combined)


def test_second_family_is_content_hash_disjoint_from_cycle_one():
    """The store rejects content-hash collisions; assert none can occur."""
    cycle_one_hashes = {
        hashlib.sha256(item.task.prompt.encode()).hexdigest()
        for item in coding_catalog()
    }
    for item in second_cycle_catalog():
        digest = hashlib.sha256(item.task.prompt.encode()).hexdigest()
        assert digest not in cycle_one_hashes, item.task.id


def test_second_family_canonical_solutions_pass_their_own_suites():
    for item in second_cycle_catalog():
        namespace: dict[str, object] = {}
        exec(item.reference_solution, namespace)  # noqa: S102
        solve = namespace["solve"]
        for case in item.suite.cases:
            actual = solve(copy.deepcopy(case.payload))
            assert actual == case.expected, (item.task.id, case.payload, actual)


def test_archived_transfer_targets_stay_in_the_future_role():
    """path_rename / path_project are probed, never trained (open question 4)."""
    future_ids = {
        item.task.id
        for item in coding_catalog()
        if item.role.value == "future"
    }
    assert future_ids == {"path_rename", "path_project"}
    second_ids = {item.task.id for item in second_cycle_catalog()}
    assert not (future_ids & second_ids)


# ---------------------------------------------------------------------------
# Spec-driven multi-cycle configuration
# ---------------------------------------------------------------------------


def test_exp005_declares_the_multi_cycle_configuration():
    machine = _machine(_spec())

    assert machine["growth_cycles"] == 2
    assert machine["second_family"] == "path_restructure"
    assert machine["correction_source"] == "canonical"
    assert machine["compare_corrections"] is False
    assert machine["min_replay_examples"] == 50


def test_exp005_growth_cycles_resolve_from_the_spec_not_defaults():
    assert experiment.resolve_growth_cycles(_machine(_spec())) == 2
    assert experiment.resolve_growth_cycles({}) == 1
    assert experiment.resolve_growth_cycles({}, growth_cycles=2) == 2


@pytest.mark.parametrize("bad", [0, 3, -1, True, "2", 2.0, None])
def test_invalid_growth_cycles_are_rejected_before_any_cost(bad):
    problems = experiment.growth_cycle_problems(bad)

    assert problems, bad


def test_exp005_preflight_accepts_the_sealed_configuration():
    spec = _spec()
    machine = _machine(spec)
    attempts, decoding = experiment.resolve_self_repair_configuration(machine)

    result = experiment.preflight_experiment(
        spec,
        correction_source="canonical",
        self_repair_attempts=attempts,
        compare_corrections=False,
        arm="primary",
        self_repair_decoding=decoding,
        growth_cycles=experiment.resolve_growth_cycles(machine),
    )
    assert result["growth_cycles"] == 2
    # Single-arm by design: independent grading needs no control pairing.
    assert result["control_requirement"]["required"] is False


def test_exp005_run_setup_satisfies_the_sealed_declaration():
    spec = _spec()
    machine = _machine(spec)
    attempts, decoding = experiment.resolve_self_repair_configuration(machine)
    setup = experiment.run_setup_manifest(
        coding_catalog() + second_cycle_catalog(),
        experiment.REAL_CYCLE_POLICY,
        correction_source="canonical",
        self_repair_attempts=attempts,
        compare_corrections=False,
        database=Path("/tmp/exp005-test.db"),
        reset=True,
        arm="primary",
        self_repair_decoding=decoding,
        growth_cycles=2,
        second_family="path_restructure",
        verifier_suite_version=experiment.MULTI_CYCLE_VERIFIER_SUITE_VERSION,
    )

    check = experiment.validate_required_setup(spec, setup)
    assert check["satisfied"], check["mismatches"]
    assert setup["growth_cycles"] == 2
    assert setup["second_family"] == "path_restructure"
    assert setup["verifier_suite_version"].endswith("path-restructure-v1")
    for purpose in ("heldout_evaluation", "replay_evaluation"):
        assert setup["decoding_by_purpose"][purpose]["temperature"] == 0.0


def test_a_single_cycle_launch_under_exp005_refuses_before_any_cost(tmp_path):
    """The sealed spec demands two cycles; a one-cycle run contradicts it."""
    database = tmp_path / "exp005.db"
    report = tmp_path / "exp005.json"

    with pytest.raises(
        experiment.ExperimentSetupError, match="growth_cycles"
    ):
        experiment.run_first_real_cycle(
            database,
            report,
            reset=True,
            correction_source="canonical",
            spec_path=_SPEC,
            arm="primary",
            growth_cycles=1,
        )
    assert not database.exists()
    assert not report.exists()


def test_run_setup_records_second_cycle_capture_and_trained_sets():
    setup: dict[str, object] = {}
    experiment.record_second_cycle_failures(
        setup, attempted=["b", "a"], trained=["a"]
    )

    assert setup["second_cycle_attempted_training_failure_ids"] == ["a", "b"]
    assert setup["second_cycle_trained_failure_ids"] == ["a"]
    assert setup["second_cycle_attempted_training_failure_set_sha256"] == (
        checker.canonical_hash(["a", "b"])
    )
    assert setup["second_cycle_trained_failure_set_sha256"] == (
        checker.canonical_hash(["a"])
    )


# ---------------------------------------------------------------------------
# The coexistence probe
# ---------------------------------------------------------------------------


class _ScriptedBackend:
    """Deterministic backend: which expert answers decides what comes back."""

    def __init__(self, breaks_when_forced: dict[str, set[str]]) -> None:
        # expert_id -> task ids that expert answers wrongly.
        self.breaks_when_forced = breaks_when_forced

    def generate(self, task: Task, expert: Expert | None) -> str:
        family = task.metadata.get("failure_type")
        if family == "core":
            if expert is not None and task.id in self.breaks_when_forced.get(
                expert.id, set()
            ):
                return "wrong"
            return task.expected or ""
        owner = task.metadata.get("owner_expert")
        if expert is not None and expert.id == owner:
            return task.expected or ""
        return "wrong"


def _expert(expert_id: str, family: str, keywords: list[str]) -> Expert:
    return Expert(
        id=expert_id,
        name=expert_id,
        status=ExpertStatus.ACTIVE,
        artifact={"parameter_count": 2_640_000},
        routing_profile={"tags": ["python", family], "keywords": keywords},
        born_from=(),
    )


def _task(
    task_id: str, prompt: str, family: str, owner: str | None = None
) -> Task:
    metadata = {"failure_type": family}
    if owner is not None:
        metadata["owner_expert"] = owner
    return Task(
        id=task_id,
        prompt=prompt,
        expected="ok",
        verifier="exact",
        tags=("python", family),
        metadata=metadata,
    )


def _two_expert_world(tmp_path, *, expert_b_keywords: list[str]):
    store = GroveStore(tmp_path / "probe.db")
    expert_a = _expert("expert_a", "fam_a", ["glimmer", "lattice"])
    expert_b = _expert("expert_b", "fam_b", expert_b_keywords)
    store.save_expert(expert_a)
    store.save_expert(expert_b)
    store.publish_deployment(
        base_model_revision="base@test",
        expert_ids=(expert_a.id, expert_b.id),
        router_version="profile-router-v1",
        verifier_suite_version="test-v1",
        decoding_config={"temperature": 0.0},
        reason="two-expert probe world",
    )
    backend = _ScriptedBackend({"expert_a": {"core_1"}})
    runtime = GroveRuntime(store, backend, verifiers=VerifierRegistry())
    heldout_by_family = {
        "fam_a": [
            _task("hold_a1", "glimmer lattice request one", "fam_a", "expert_a"),
            _task("hold_a2", "glimmer lattice request two", "fam_a", "expert_a"),
        ],
        "fam_b": [
            _task("hold_b1", "quasar prism request one", "fam_b", "expert_b"),
            _task("hold_b2", "quasar prism request two", "fam_b", "expert_b"),
        ],
    }
    replay = [
        _task("core_1", "plain widget job one", "core"),
        _task("core_2", "plain widget job two", "core"),
        _task("core_3", "plain widget job three", "core"),
        _task("core_4", "plain widget job four", "core"),
        _task("banked_a1", "glimmer lattice banked one", "fam_a", "expert_a"),
        _task("banked_a2", "glimmer lattice banked two", "fam_a", "expert_a"),
    ]
    return store, runtime, heldout_by_family, replay


def test_coexistence_probe_measures_two_experts_with_denominators(tmp_path):
    store, runtime, heldout_by_family, replay = _two_expert_world(
        tmp_path, expert_b_keywords=["quasar", "prism"]
    )
    try:
        checkpoint = experiment.coexistence_checkpoint(
            runtime,
            store,
            capability={
                "capability": 0.9,
                "added_parameters": 5_280_000,
                "evaluation_id": "eval_x",
            },
            replay=replay,
            heldout_by_family=heldout_by_family,
            cycle_one_training_ids=["banked_a1", "banked_a2", "banked_a3"],
        )
    finally:
        store.close()

    assert checkpoint["active_experts"] == 2
    assert checkpoint["replay_examples"] == 6
    # Routed replay: the router keeps every prior-passing task passing.
    assert checkpoint["replay_regression_rate"] == 0.0
    assert checkpoint["cycle_1_training_targets_in_replay"] == 2
    # Route metrics span both experts on oracle-free copies.
    assert checkpoint["route_positives"] == 4
    assert checkpoint["route_recall"] == 1.0
    assert checkpoint["route_precision"] == 1.0
    assert checkpoint["route_false_positive_rate"] == 0.0
    assert checkpoint["route_negatives"] == 4
    assert checkpoint["route_positive_source"] == "heldout_forced_pass_oracle_free"
    fam_a = checkpoint["per_family"]["fam_a"]
    fam_b = checkpoint["per_family"]["fam_b"]
    # The interference probe re-measures each family routed and forced-on.
    assert fam_a["heldout_routed_rate"] == 1.0
    assert fam_a["heldout_forced_rate"] == 1.0
    assert fam_b["heldout_routed_rate"] == 1.0
    # Forced-on replay is measured per expert against the bare base, so each
    # expert's forgetting claim is its own.
    assert fam_a["forced_replay_denominator"] == 4
    assert fam_a["forced_regression_rate"] == 0.25
    assert fam_a["forgetting_claim"] == "router_shielded"
    assert fam_b["forced_regression_rate"] == 0.0
    assert fam_b["forgetting_claim"] == "adapter_intrinsic"
    assert checkpoint["added_parameters"] == 5_280_000


def test_coexistence_probe_records_router_confusion_as_a_drop(tmp_path):
    """An expert whose family cannot reach it measures recall 0, not silence."""
    store, runtime, heldout_by_family, replay = _two_expert_world(
        tmp_path, expert_b_keywords=["unrelated", "vocabulary"]
    )
    try:
        checkpoint = experiment.coexistence_checkpoint(
            runtime,
            store,
            capability={"capability": 0.5, "added_parameters": 0},
            replay=replay,
            heldout_by_family=heldout_by_family,
            cycle_one_training_ids=[],
        )
    finally:
        store.close()
    fam_b = checkpoint["per_family"]["fam_b"]
    # Expert B still solves its family forced-on, and the deployed router
    # still finds it through the gold family tag -- but the oracle-free probe,
    # which sees only the prompt, records the confusion where a rule can read
    # it. That asymmetry is exactly why route_recall is measured tag-stripped.
    assert fam_b["heldout_forced_rate"] == 1.0
    assert fam_b["heldout_routed_rate"] == 1.0
    assert fam_b["route_recall"] == 0.0
    assert checkpoint["route_recall"] == 0.5
    assert checkpoint["route_false_positive_rate"] == 0.0


def test_coexistence_probe_reports_empty_baseline_as_unmeasured(tmp_path):
    store = GroveStore(tmp_path / "empty.db")
    backend = _ScriptedBackend({})
    runtime = GroveRuntime(store, backend, verifiers=VerifierRegistry())
    try:
        checkpoint = experiment.coexistence_checkpoint(
            runtime,
            store,
            capability={"capability": 0.6, "added_parameters": 0},
            replay=[],
            heldout_by_family={"fam_a": [], "fam_b": []},
            cycle_one_training_ids=[],
        )
    finally:
        store.close()

    assert checkpoint["active_experts"] == 0
    assert checkpoint["replay_examples"] == 0
    assert checkpoint["replay_regression_rate"] is None
    assert checkpoint["route_recall"] is None
    assert checkpoint["route_false_positive_rate"] is None
    for entry in checkpoint["per_family"].values():
        assert entry["forgetting_claim"] == "unmeasured"
        assert entry["forced_regression_rate"] is None


# ---------------------------------------------------------------------------
# Independent grading: falsified is exit 1, unusable stays exit 2
# ---------------------------------------------------------------------------


def _passing_coexistence() -> dict:
    def family(expert_id: str) -> dict:
        return {
            "expert_id": expert_id,
            "heldout_examples": 4,
            "heldout_forced_rate": 0.75,
            "heldout_routed_rate": 0.75,
            "route_positives": 3,
            "route_positives_routed_to_own_expert": 3,
            "route_recall": 1.0,
            "forced_replay_denominator": 94,
            "forced_regression_rate": 0.489,
            "forced_regression_task_ids": [],
            "forced_regression_reference": "base_no_experts",
            "forgetting_claim": "router_shielded",
        }

    checkpoint = {
        "active_experts": 2,
        "active_expert_ids": ["expert_a", "expert_b"],
        "added_parameters": 5_280_000,
        "capability": 0.93,
        "evaluation_id": "eval_2",
        "replay_examples": 112,
        "replay_pass_rate": 1.0,
        "replay_regression_rate": 0.0,
        "replay_regressed_task_ids": [],
        "cycle_1_training_targets_in_replay": 18,
        "route_positive_source": "heldout_forced_pass_oracle_free",
        "route_probe_metadata": "oracle_free",
        "route_positives": 6,
        "route_negatives": 94,
        "route_negatives_routed": 0,
        "route_recall": 1.0,
        "route_precision": 1.0,
        "route_precision_cohort_dependent": True,
        "route_false_positive_rate": 0.0,
        "per_family": {
            "escaped_path": family("expert_a"),
            "path_restructure": family("expert_b"),
        },
    }
    return {
        "baseline": {"capability": 0.878, "active_experts": 0},
        "after_cycle_1": {"capability": 0.9, "active_experts": 1},
        "after_cycle_2": checkpoint,
        "capability": {
            "baseline": 0.878,
            "after_cycle_1": 0.9,
            "after_cycle_2": 0.93,
            "delta_cycle_1": 0.022,
            "delta_cycle_2": 0.03,
            "monotonic_non_decreasing": True,
        },
    }


def _provenance() -> dict:
    provenance = {
        "base_model": "model@abc",
        "decoding_config": {"temperature": 0.0, "max_tokens": 768},
        "worker": {"model_manifest_sha256": "3" * 64},
        "extra": {"correction_source": "canonical", "growth_cycles": 2},
    }
    provenance["provenance_sha256"] = checker.canonical_hash(provenance)
    return provenance


def _exp005_report(spec: dict) -> dict:
    machine = _machine(spec)
    document = {
        "arm": "primary",
        "correction_source": "canonical",
        "experiment_spec": {
            "spec_id": spec["spec_id"],
            "spec_sha256": spec["spec_sha256"],
        },
        "run_setup": {
            **machine,
            "decoding_by_purpose": {
                "heldout_evaluation": {"temperature": 0.0},
                "replay_evaluation": {"temperature": 0.0},
            },
        },
        "provenance": _provenance(),
        "provenance_gaps": ["models.base.aggregate_sha256"],
        "growth_cycles": 2,
        "cycle": {"experts_admitted": ["expert_a"]},
        "second_cycle": {"experts_admitted": ["expert_b"]},
        "coexistence": _passing_coexistence(),
    }
    return experiment.seal_report(document, spec=spec)


def test_exp005_grades_a_conforming_report_as_all_rules_satisfied():
    spec = _spec()
    verdict = checker.check(spec, _exp005_report(spec))

    assert verdict["unusable"] is False, verdict["blockers"]
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []


def test_a_zero_yield_second_cycle_is_falsified_not_unusable():
    """The EXP-004 structural lesson: a measurable zero grades as a result."""
    spec = _spec()
    machine = _machine(spec)
    document = {
        "arm": "primary",
        "correction_source": "canonical",
        "experiment_spec": {
            "spec_id": spec["spec_id"],
            "spec_sha256": spec["spec_sha256"],
        },
        "run_setup": {
            **machine,
            "decoding_by_purpose": {
                "heldout_evaluation": {"temperature": 0.0},
                "replay_evaluation": {"temperature": 0.0},
            },
        },
        "provenance": _provenance(),
        "provenance_gaps": ["models.base.aggregate_sha256"],
        "growth_cycles": 2,
        "cycle": {"experts_admitted": ["expert_a"]},
        # The second cycle admitted nothing: its metrics are honest absences.
        "second_cycle": {"experts_admitted": []},
        "coexistence": _passing_coexistence(),
    }
    document["coexistence"]["after_cycle_2"]["active_experts"] = 1
    restructure = document["coexistence"]["after_cycle_2"]["per_family"][
        "path_restructure"
    ]
    restructure.update(
        {
            "expert_id": None,
            "heldout_forced_rate": None,
            "heldout_routed_rate": None,
            "route_recall": None,
            "forced_regression_rate": None,
            "forgetting_claim": "unmeasured",
        }
    )
    report = experiment.seal_report(document, spec=spec)

    verdict = checker.check(spec, report)

    assert verdict["unusable"] is False, verdict["blockers"]
    assert verdict["rules_failed"] > 0
    assert "H1" in verdict["falsified_hypotheses"]
    failed_ids = {
        rule["id"]
        for rule in verdict["rules"]
        if not rule["passed"] and not rule["unevaluable"]
    }
    assert {"D5", "D6", "D7", "D8b", "D19b"} <= failed_ids


def test_an_edited_coexistence_metric_is_unusable_not_a_result():
    spec = _spec()
    report = _exp005_report(spec)
    report["coexistence"]["after_cycle_2"]["route_false_positive_rate"] = 0.5

    verdict = checker.check(spec, report)

    assert verdict["unusable"] is True
    assert verdict["falsified_hypotheses"] == []
    assert verdict["rules_failed"] == 0


# ---------------------------------------------------------------------------
# Sealed-spec shape and the runnable procedure
# ---------------------------------------------------------------------------


def test_exp005_is_sealed_single_arm_and_independently_graded():
    spec = _spec()

    assert spec["spec_sha256"] == checker.spec_digest(spec)
    assert spec["requires_report_integrity"] is True
    assert spec["seal_self_consistent"] is True
    assert spec["timing_attestation"] is None
    assert spec["permitted_provenance_gaps"] == ["models.base.aggregate_sha256"]
    assert spec["required_resolved_identity"] == [
        "provenance.worker.model_manifest_sha256"
    ]
    # Independent grading by construction: no rule needs a second arm, so a
    # zero-yield outcome can never collapse the run to unusable via pairing.
    for rule in spec["decision_rules"]:
        assert not str(rule["comparison"]).startswith("delta"), rule["id"]
        assert "control_path" not in rule, rule["id"]
        assert "pair_on" not in rule, rule["id"]
        assert rule.get("arm") != "control", rule["id"]
    assert checker.control_requirement(spec)["required"] is False
    # The unusable-versus-falsified semantics are stated in the sealed text.
    assert any(
        "UNUSABLE VERSUS FALSIFIED" in item
        for item in spec["preregistered_limitations"]
    )
    assert "THRESHOLD JUSTIFICATIONS" in spec["background"]


def test_exp005_thresholds_match_the_sealed_admission_policy():
    spec = _spec()
    rules = {rule["id"]: rule for rule in spec["decision_rules"]}
    policy = experiment.REAL_CYCLE_POLICY

    assert rules["D7"]["value"] == policy.min_heldout_fix_rate == 0.75
    assert rules["D14"]["value"] == policy.min_heldout_fix_rate
    assert rules["D8"]["value"] == policy.min_route_recall == 0.5
    assert rules["D10"]["value"] == policy.max_route_false_positive_rate == 0.0
    assert rules["D11"]["value"] == policy.min_replay_examples == 50
    assert rules["D12"]["value"] == int(policy.min_target_fix_rate * 20) == 16
    assert rules["D13"]["value"] == policy.max_regression_rate == 0.0


def test_run_script_matches_the_sealed_command():
    """The reviewer diffs the script against the sealed spec; so do we."""
    script = _SCRIPT.read_text()
    spec = _spec()

    assert "real-cycle --reset --cycles 2" in script
    assert "--spec" in script and "--arm primary" in script
    assert "--correction-source canonical" in script
    assert 'SPEC="experiments/EXP-005-second-cycle-coexistence.json"' in script
    # The real-cycle exit code -- exit 2 included -- propagates unchanged.
    assert 'exit "$run_status"' in script
    for fragment in (
        "scripts/run_exp005.sh <db> <report>",
        "real-cycle --reset --cycles 2",
        "--arm primary",
        "--correction-source canonical",
        "check_experiment_spec.py",
    ):
        assert fragment in spec["command"], fragment
