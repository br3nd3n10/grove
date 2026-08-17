"""EXP-004: the fair self-repair regime is spec-driven and sealed.

EXP-003's self-repair arm was configured unfairly by accident: the global
greedy decoding made all three attempts per failure near-identical, and the
repair prompt carried one generic sentence. These tests pin that EXP-004's
regime -- 8 sampled, seeded attempts with honest feedback -- is configured by
the sealed spec's ``required_setup``, validated before any cost, and recorded
in the run setup where the sealed decision rules can bind it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grove import experiment

_SPEC = Path(__file__).parents[1] / "experiments" / "EXP-004-fair-self-repair-ab.json"


def _spec() -> dict:
    return json.loads(_SPEC.read_text())


def _machine(spec: dict, arm: str) -> dict:
    declaration = (
        spec["required_setup"] if arm == "primary" else spec["control_required_setup"]
    )
    return experiment.normalize_required_setup(declaration)["machine"]


def test_exp004_declares_the_fair_regime_in_machine_checkable_form():
    spec = _spec()

    for arm in ("primary", "control"):
        machine = _machine(spec, arm)
        assert machine["self_repair_attempts"] == 8, arm
        assert machine["self_repair_decoding"] == {
            "base_seed": 20260809,
            "max_tokens": 768,
            "temperature": 0.8,
        }, arm
    # The declared H1 threshold is 0.25, with a written justification.
    d3 = next(rule for rule in spec["decision_rules"] if rule["id"] == "D3")
    assert d3["value"] == 0.25
    assert "THRESHOLD JUSTIFICATION" in spec["background"]
    assert "min_cluster_size 3" in spec["background"]


def test_exp004_attempts_and_decoding_resolve_from_the_spec_not_defaults():
    """Requirement: raise attempts via configuration, never policy edits."""
    spec = _spec()

    attempts, decoding = experiment.resolve_self_repair_configuration(
        _machine(spec, "primary")
    )
    assert attempts == 8
    assert decoding == {"base_seed": 20260809, "max_tokens": 768, "temperature": 0.8}

    # An explicit argument still wins, and no spec means the legacy defaults.
    attempts, decoding = experiment.resolve_self_repair_configuration(
        _machine(spec, "primary"), attempts=2, decoding={"temperature": 0.5}
    )
    assert attempts == 2
    assert decoding == {"temperature": 0.5}
    assert experiment.resolve_self_repair_configuration({}) == (3, None)


def test_exp004_preflight_accepts_both_arms():
    spec = _spec()
    attempts, decoding = experiment.resolve_self_repair_configuration(
        _machine(spec, "primary")
    )

    for arm, source in (("primary", "self-repair"), ("control", "canonical")):
        result = experiment.preflight_experiment(
            spec,
            correction_source=source,
            self_repair_attempts=attempts,
            compare_corrections=True,
            arm=arm,
            self_repair_decoding=decoding,
        )
        assert result["self_repair_attempts"] == 8, arm
        assert result["self_repair_decoding"]["temperature"] == 0.8, arm
        assert result["control_requirement"]["required"] is True, arm


def test_exp004_run_setup_records_per_purpose_decoding_for_both_arms():
    """Evaluation stays greedy; only self-repair samples; the spec binds both."""
    from grove.coding_tasks import coding_catalog

    spec = _spec()
    attempts, decoding = experiment.resolve_self_repair_configuration(
        _machine(spec, "primary")
    )
    catalog = coding_catalog()
    for arm, source in (("primary", "self-repair"), ("control", "canonical")):
        name, declaration = experiment.select_setup_profile(
            spec, correction_source=source, arm=arm
        )
        setup = experiment.run_setup_manifest(
            catalog,
            experiment.REAL_CYCLE_POLICY,
            correction_source=source,
            self_repair_attempts=attempts,
            compare_corrections=True,
            database=Path("/tmp/exp004-test.db"),
            reset=True,
            arm=name,
            self_repair_decoding=decoding,
        )
        check = experiment.validate_required_setup(
            spec, setup, declaration=declaration
        )
        assert check["satisfied"], (arm, check["mismatches"])
        purposes = setup["decoding_by_purpose"]
        for purpose in (
            "baseline_evaluation",
            "heldout_evaluation",
            "replay_evaluation",
        ):
            assert purposes[purpose]["temperature"] == 0.0, (arm, purpose)
        assert purposes["self_repair"]["temperature"] == 0.8, arm
        assert setup["self_repair_decoding"]["base_seed"] == 20260809, arm
        assert setup["self_repair_attempts"] == 8, arm


@pytest.mark.parametrize(
    ("decoding", "expected"),
    [
        ("not-a-mapping", "must be a mapping"),
        ({"base_seed": 1}, "temperature must be a number"),
        ({"temperature": 0.0, "base_seed": 1}, "is not sampled"),
        ({"temperature": 0.8}, "base_seed must be an integer"),
        ({"temperature": 0.8, "base_seed": True}, "base_seed must be an integer"),
        ({"temperature": 0.8, "base_seed": 1, "max_tokens": 0}, "max_tokens"),
        ({"temperature": 0.8, "base_seed": 1, "top_p": 0.9}, "unsupported key"),
    ],
)
def test_invalid_self_repair_decoding_is_rejected_before_any_cost(
    decoding, expected
):
    problems = experiment.self_repair_decoding_problems(decoding)

    assert problems, decoding
    assert any(expected in problem for problem in problems), problems


def test_a_spec_declaring_bad_repair_decoding_fails_the_preflight(tmp_path):
    from grove.provenance import canonical_hash

    spec = _spec()
    spec.pop(experiment.SPEC_HASH_FIELD)
    spec["required_setup"]["machine"]["self_repair_decoding"] = {
        "temperature": 0.0,
        "base_seed": 1,
    }
    spec[experiment.SPEC_HASH_FIELD] = canonical_hash(spec)

    with pytest.raises(experiment.ExperimentSetupError, match="is not sampled"):
        experiment.preflight_experiment(
            spec,
            correction_source="self-repair",
            self_repair_attempts=8,
            compare_corrections=True,
            arm="primary",
        )


def test_exp004_and_exp003_share_the_pairing_and_provenance_design():
    """The fair regime changes the repair budget, never the comparison rules."""
    exp003 = json.loads(
        (_SPEC.parent / "EXP-003-correction-source-ab.json").read_text()
    )
    exp004 = _spec()

    def rule(spec: dict, rule_id: str) -> dict:
        return next(item for item in spec["decision_rules"] if item["id"] == rule_id)

    for rule_id in ("D6d", "D6e"):
        assert rule(exp004, rule_id)["pair_on"] == rule(exp003, rule_id)["pair_on"]
        assert rule(exp004, rule_id)["value"] == rule(exp003, rule_id)["value"]
    assert rule(exp004, "D7")["value"] == rule(exp003, "D7")["value"]
    assert (
        exp004["required_resolved_identity"] == exp003["required_resolved_identity"]
    )
    assert exp004["permitted_provenance_gaps"] == exp003["permitted_provenance_gaps"]
    assert exp004["requires_report_integrity"] is True
    assert exp004["requires_control_report"] is True
    # Same unusable-versus-falsified semantics, stated in the sealed text.
    assert any(
        "unusable run is not a falsified run" in item
        for item in exp004["preregistered_limitations"]
    )
