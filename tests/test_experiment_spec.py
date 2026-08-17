"""The predeclared-spec checker must be able to say "the prediction failed".

A spec that can only pass is decoration. These tests pin the outcomes that
matter: satisfied, falsified, tampered-with, unbound, and -- for a paired
design -- missing or mismatched control arm.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import types
from dataclasses import replace
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_experiment_spec.py"
_SPEC_DIR = Path(__file__).parents[1] / "experiments"
_spec = importlib.util.spec_from_file_location("check_experiment_spec", _SCRIPT)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def sealed(spec: dict) -> dict:
    """Deep copy so a test that tampers with a rule cannot leak into the next."""
    clone = copy.deepcopy(spec)
    return {**clone, checker.SPEC_HASH_FIELD: checker.spec_digest(clone)}


BASE_SPEC = {
    "spec_id": "EXP-TEST",
    "hypotheses": [
        {
            "id": "H1",
            "claim": "the adapter forgets nothing",
            "falsified_if": "any prior-passing replay task fails under the adapter",
        }
    ],
    "decision_rules": [
        {
            "id": "D1",
            "hypothesis": "H1",
            "path": "experts[*].metrics.forced_regression_rate",
            "comparison": "<=",
            "value": 0.0,
        },
        {
            "id": "D2",
            "hypothesis": "H1",
            "path": "experts[*].metrics.forgetting_claim",
            "comparison": "==",
            "value": "adapter_intrinsic",
        },
        {
            "id": "D3",
            "path": "provenance_gaps",
            "comparison": "set==",
            "value": [],
        },
    ],
}

PAIRED_SPEC = {
    "spec_id": "EXP-PAIR",
    "requires_control_report": True,
    "hypotheses": [
        {
            "id": "H2",
            "claim": "self-repair matches the human reference",
            "falsified_if": "held-out rate is more than 0.1 below the control arm",
        }
    ],
    "required_setup": {"correction_source": "self-repair"},
    "control_required_setup": {"correction_source": "canonical"},
    "decision_rules": [
        {
            "id": "D1",
            "hypothesis": "H2",
            "path": "experts[*].metrics.heldout_target_rate",
            "control_path": "experts[*].metrics.heldout_target_rate",
            "pair_on": "experts[*].pairing_key",
            "comparison": "delta>=",
            "value": -0.1,
        },
        {
            "id": "D2",
            "arm": "control",
            "path": "correction_source",
            "comparison": "==",
            "value": "canonical",
        },
    ],
}


def _provenance() -> dict:
    """A fully resolved provenance block: nothing null, nothing unavailable."""
    return {
        "base_model": "model@abc",
        "verifiers": {"suites_sha256": "5" * 64},
        "training_config_sha256": "6" * 64,
        "decoding_config": {"temperature": 0.0},
        "source": {
            "revision": "a" * 40,
            "tree": "b" * 40,
            "dirty": False,
            "status_sha256": "d" * 64,
            "worktree_sha256": "e" * 64,
        },
        "worker": {
            "host": "worker-host",
            "framework_versions_sha256": "f" * 64,
            "checkout": {
                "revision": "9" * 40,
                "tree": "8" * 40,
                "dirty": False,
                "status_sha256": "1" * 64,
                "worktree_sha256": "2" * 64,
            },
            "model_manifest_sha256": "3" * 64,
        },
        "sandbox_image": {"image": "grove-python-base", "fingerprint": "7" * 64},
        "models": {"base": {"aggregate_sha256": "c" * 64}},
    }


def _run_setup(source: str) -> dict:
    return {
        "arm": "primary" if source == "self-repair" else "control",
        "correction_source": source,
        "compare_corrections": True,
        "self_repair_attempts": 3,
        "min_replay_examples": 50,
        "verifier_suite_version": "suite-v1",
        "admission_policy_sha256": "a1" * 32,
        "cohort_manifest_sha256": "b2" * 32,
        "actual_training_failure_ids": ["train_a", "train_b"],
        "actual_training_failure_set_sha256": "c3" * 32,
        "attempted_training_failure_set_sha256": "c3" * 32,
    }


def test_training_failure_manifest_distinguishes_attempted_from_trained():
    from grove.experiment import (
        build_run_manifest,
        record_actual_training_failures,
        record_trained_failures,
    )

    setup: dict[str, object] = {}
    record_actual_training_failures(setup, ["train_b", "train_a"])
    record_trained_failures(setup, ["train_a"])

    assert setup["actual_training_failure_ids"] == ["train_a", "train_b"]
    assert setup["attempted_training_failure_ids"] == ["train_a", "train_b"]
    assert setup["trained_failure_ids"] == ["train_a"]
    assert setup["attempted_training_failure_set_sha256"] == checker.canonical_hash(
        ["train_a", "train_b"]
    )
    assert setup["trained_failure_set_sha256"] == checker.canonical_hash(["train_a"])
    assert setup["trained_failure_ids"] != setup["attempted_training_failure_ids"]

    manifest = build_run_manifest({"run_setup": setup})
    assert manifest["attempted_training_failure_set_sha256"] == setup[
        "attempted_training_failure_set_sha256"
    ]
    assert manifest["trained_failure_set_sha256"] == setup[
        "trained_failure_set_sha256"
    ]


def bind(document: dict, spec: dict | None = None) -> dict:
    """Attach the same local run manifest the runner writes."""
    from grove.experiment import build_run_manifest

    manifest = build_run_manifest(
        document,
        rule_paths=checker.decision_rule_input_paths(spec or {}),
    )
    document["run_manifest"] = manifest
    document["run_manifest_sha256"] = checker.canonical_hash(manifest)
    return document


def report(
    *,
    forced_rate: float,
    claim: str,
    gaps: list[str] | None = None,
    spec: dict | None = None,
) -> dict:
    document = {
        "provenance_gaps": gaps or [],
        "experts": [
            {
                "id": "expert_a",
                "metrics": {
                    "forced_regression_rate": forced_rate,
                    "forgetting_claim": claim,
                },
            }
        ],
    }
    provenance = _provenance()
    provenance["provenance_sha256"] = checker.canonical_hash(provenance)
    document["provenance"] = provenance
    if spec is not None:
        document["experiment_spec"] = {
            "spec_id": spec["spec_id"],
            checker.SPEC_HASH_FIELD: spec[checker.SPEC_HASH_FIELD],
        }
    return bind(document, spec)


def arm(
    *,
    spec: dict,
    source: str,
    heldout: float,
    forced: float = 0.0,
    expert_ids: tuple[str, ...] = ("expert_a",),
    pairing_keys: tuple[str, ...] | None = None,
) -> dict:
    """One arm of a paired report.

    ``expert_ids`` are deliberately free to differ between arms: real expert
    ids are fresh UUIDs. ``pairing_keys`` is what the arms must share.
    """
    keys = pairing_keys or tuple(f"key_{index}" for index in range(len(expert_ids)))
    provenance = _provenance()
    provenance["provenance_sha256"] = checker.canonical_hash(provenance)
    return bind(
        {
            "correction_source": source,
            "provenance": provenance,
            "provenance_gaps": [],
            "run_setup": _run_setup(source),
            "experiment_spec": {
                "spec_id": spec["spec_id"],
                checker.SPEC_HASH_FIELD: spec[checker.SPEC_HASH_FIELD],
            },
            "experts": [
                {
                    "id": expert_id,
                    "pairing_key": pairing_key,
                    "metrics": {
                        "pairing_key": pairing_key,
                        "heldout_target_rate": heldout,
                        "forced_regression_rate": forced,
                    },
                }
                for expert_id, pairing_key in zip(expert_ids, keys, strict=True)
            ],
        },
        spec,
    )


# --------------------------------------------------------------------------
# Core verdicts
# --------------------------------------------------------------------------


def test_satisfied_run_reports_no_falsified_hypotheses():
    spec = sealed(BASE_SPEC)

    verdict = checker.check(
        spec, report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    )

    assert verdict["spec_intact"] is True
    assert verdict["spec_binding"]["bound"] is True
    assert verdict["unusable"] is False
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert verdict["verdict"] == "all predeclared rules satisfied"


def test_router_shielded_result_falsifies_the_intrinsic_hypothesis():
    spec = sealed(BASE_SPEC)

    verdict = checker.check(
        spec, report(forced_rate=0.5, claim="router_shielded", spec=spec)
    )

    assert verdict["unusable"] is False
    assert verdict["rules_failed"] == 2
    assert verdict["falsified_hypotheses"] == ["H1"]
    assert verdict["verdict"] == "predeclared rule(s) failed"


def test_editing_a_threshold_after_declaration_is_detected():
    spec = sealed(BASE_SPEC)
    bound = report(forced_rate=0.5, claim="router_shielded", spec=spec)
    spec["decision_rules"][0]["value"] = 0.9

    verdict = checker.check(spec, bound)

    assert verdict["spec_intact"] is False
    assert verdict["unusable"] is True
    assert verdict["verdict"] == "spec altered after declaration"
    assert verdict["recorded_spec_sha256"] != verdict["computed_spec_sha256"]


def test_a_missing_metric_fails_rather_than_passing_silently():
    spec = sealed(BASE_SPEC)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    document["experts"] = [{"metrics": {}}]
    bind(document, spec)

    verdict = checker.check(spec, document)

    assert verdict["rules_failed"] == 2
    assert all(
        "absent" in rule["detail"] for rule in verdict["rules"] if not rule["passed"]
    )


def test_fanout_requires_every_expert_to_comply():
    spec = sealed(BASE_SPEC)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    document["experts"].append(
        {
            "metrics": {
                "forced_regression_rate": 0.25,
                "forgetting_claim": "router_shielded",
            }
        }
    )
    bind(document, spec)

    verdict = checker.check(spec, document)

    assert verdict["rules_failed"] == 2
    assert "0.25" in verdict["rules"][0]["detail"]


# --------------------------------------------------------------------------
# Finding 5: the report must be bound to the spec version that produced it.
# --------------------------------------------------------------------------


def test_a_report_with_no_spec_binding_is_unusable():
    spec = sealed(BASE_SPEC)

    verdict = checker.check(spec, report(forced_rate=0.0, claim="adapter_intrinsic"))

    assert verdict["spec_intact"] is True
    assert verdict["unusable"] is True
    assert "not bound" in verdict["verdict"]
    assert verdict["spec_binding"]["reason"] == (
        "report records no experiment_spec.spec_sha256"
    )


def test_resealing_an_edited_spec_invalidates_the_earlier_report():
    """The whole point of binding: re-sealing cannot rescue an old result."""
    original = sealed(BASE_SPEC)
    old_report = report(forced_rate=0.0, claim="adapter_intrinsic", spec=original)
    assert checker.check(original, old_report)["unusable"] is False

    edited = copy.deepcopy(BASE_SPEC)
    edited["decision_rules"][0]["value"] = 0.9
    resealed = sealed(edited)

    verdict = checker.check(resealed, old_report)

    # The resealed file is internally consistent again...
    assert verdict["spec_intact"] is True
    # ...but the report names the digest it actually ran under.
    assert verdict["unusable"] is True
    assert verdict["spec_binding"]["reason"] == (
        "report was produced under a different version of this spec"
    )


def test_a_report_bound_to_another_spec_is_rejected():
    spec = sealed(BASE_SPEC)
    other = sealed(PAIRED_SPEC)

    verdict = checker.check(spec, report(forced_rate=0.0, claim="ok", spec=other))

    assert verdict["unusable"] is True
    assert "EXP-PAIR" in verdict["spec_binding"]["reason"]


def test_seal_refuses_to_overwrite_without_reseal(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(BASE_SPEC))

    assert checker.main(["--spec", str(path), "--seal"]) == 0
    first = json.loads(path.read_text())[checker.SPEC_HASH_FIELD]
    assert checker.main(["--spec", str(path), "--seal"]) == 2
    assert json.loads(path.read_text())[checker.SPEC_HASH_FIELD] == first

    edited = json.loads(path.read_text())
    edited["decision_rules"][0]["value"] = 0.9
    path.write_text(json.dumps(edited))
    assert checker.main(["--spec", str(path), "--seal", "--reseal"]) == 0
    assert json.loads(path.read_text())[checker.SPEC_HASH_FIELD] != first


# --------------------------------------------------------------------------
# Finding 4: a paired design must actually compare its control arm.
# --------------------------------------------------------------------------


def test_paired_spec_without_a_control_report_is_unusable():
    spec = sealed(PAIRED_SPEC)

    verdict = checker.check(spec, arm(spec=spec, source="self-repair", heldout=0.75))

    assert verdict["control_required"] is True
    assert verdict["control_supplied"] is False
    assert verdict["unusable"] is True
    assert "need a control report and none was supplied" in verdict["verdict"]
    # The requirement comes from the rules themselves, not the optional flag.
    assert [entry["rule"] for entry in verdict["control_requirement"]["rules"]] == [
        "D1",
        "D2",
    ]


def test_a_self_repair_arm_far_below_the_control_falsifies_the_hypothesis():
    spec = sealed(PAIRED_SPEC)

    verdict = checker.check(
        spec,
        arm(spec=spec, source="self-repair", heldout=0.75),
        arm(spec=spec, source="canonical", heldout=1.0),
    )

    assert verdict["unusable"] is False
    assert verdict["falsified_hypotheses"] == ["H2"]
    delta_rule = verdict["rules"][0]
    assert delta_rule["passed"] is False
    assert "deltas [-0.25]" in delta_rule["detail"]


def test_a_self_repair_arm_within_the_declared_tolerance_passes():
    spec = sealed(PAIRED_SPEC)

    verdict = checker.check(
        spec,
        arm(spec=spec, source="self-repair", heldout=0.85),
        arm(spec=spec, source="canonical", heldout=0.9),
    )

    assert verdict["rules_failed"] == 0
    assert verdict["verdict"] == "all predeclared rules satisfied"


def test_arms_that_differ_in_more_than_the_variable_under_test_are_rejected():
    spec = sealed(PAIRED_SPEC)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    control["provenance"]["training_config_sha256"] = "different-recipe"
    # Model a second, independently sealed run whose recipe really differs;
    # do not leave the report integrity binding stale while testing pairing.
    from grove.experiment import seal_report
    from grove.provenance import canonical_hash

    control["provenance"]["provenance_sha256"] = canonical_hash(
        {
            key: value
            for key, value in control["provenance"].items()
            if key != "provenance_sha256"
        }
    )
    seal_report(control, spec=spec)

    verdict = checker.check(
        spec, arm(spec=spec, source="self-repair", heldout=0.85), control
    )

    assert verdict["unusable"] is True
    assert verdict["verdict"] == "control and primary arms are not comparable"
    assert any(
        item["path"] == "provenance.training_config_sha256"
        for item in verdict["arm_pairing"]["mismatches"]
    )


def test_two_identical_arms_are_not_a_comparison():
    spec = sealed(PAIRED_SPEC)

    verdict = checker.check(
        spec,
        arm(spec=spec, source="canonical", heldout=0.9),
        arm(spec=spec, source="canonical", heldout=0.9),
    )

    assert verdict["unusable"] is True
    assert any(
        item["path"] == "correction_source"
        for item in verdict["arm_pairing"]["mismatches"]
    )


def test_control_arm_rules_read_the_control_report():
    spec = sealed(PAIRED_SPEC)

    verdict = checker.check(
        spec,
        arm(spec=spec, source="self-repair", heldout=0.85),
        arm(spec=spec, source="canonical", heldout=0.9),
    )
    control_rule = next(rule for rule in verdict["rules"] if rule["arm"] == "control")

    assert control_rule["passed"] is True
    assert "canonical" in control_rule["detail"]


# --------------------------------------------------------------------------
# Comparison semantics and shipped-spec hygiene
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("comparison", "observed", "expected", "passes"),
    [
        ("count>=", [1, 2], 1, True),
        ("count>=", [], 1, False),
        ("exists", "anything", None, True),
        ("in", "canonical", ["canonical", "self-repair"], True),
        ("not_in", "canonical", ["self-repair"], True),
        (">=", True, 0.5, False),
        ("set==", ["b", "a"], ["a", "b"], True),
        ("delta>=", -0.05, -0.1, True),
        ("delta>=", -0.25, -0.1, False),
    ],
)
def test_comparison_semantics(comparison, observed, expected, passes):
    result, _ = checker.evaluate(comparison, observed, expected)

    assert result is passes


def test_booleans_are_not_treated_as_numbers():
    """`True >= 0.5` is true in Python; a rate rule must not accept a flag."""
    passed, detail = checker.evaluate(">=", True, 0.5)

    assert passed is False
    assert "True" in detail


def test_null_metrics_never_satisfy_a_rate_rule():
    """An unmeasured metric is serialised as null and must not pass a bound."""
    passed, _ = checker.evaluate("<=", None, 0.0)

    assert passed is False


def test_shipped_specs_are_sealed_and_parse():
    specs = sorted(_SPEC_DIR.glob("*.json"))

    assert specs, "no predeclared experiment specs found"
    for path in specs:
        spec = json.loads(path.read_text())
        assert spec[checker.SPEC_HASH_FIELD] == checker.spec_digest(spec), path.name
        assert spec["decision_rules"], path.name
        assert spec["hypotheses"], path.name
        # Every shipped spec must name what would refute it.
        for hypothesis in spec["hypotheses"]:
            assert hypothesis["falsified_if"].strip(), path.name
        # Every shipped spec must bind its run command to itself.
        assert "--spec" in spec["command"], path.name


def test_shipped_specs_use_the_same_digest_as_the_runner(tmp_path):
    """The script and the library must agree, or binding silently breaks."""
    from grove.experiment import SPEC_HASH_FIELD, load_sealed_spec
    from grove.provenance import canonical_hash

    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        payload = {key: value for key, value in spec.items() if key != SPEC_HASH_FIELD}
        assert canonical_hash(payload) == checker.spec_digest(spec), path.name
        assert load_sealed_spec(path)[SPEC_HASH_FIELD] == spec[SPEC_HASH_FIELD]


def test_runner_refuses_an_unsealed_or_edited_spec(tmp_path):
    from grove.experiment import load_sealed_spec

    unsealed = tmp_path / "unsealed.json"
    unsealed.write_text(json.dumps(BASE_SPEC))
    with pytest.raises(ValueError, match="not sealed"):
        load_sealed_spec(unsealed)

    tampered = tmp_path / "tampered.json"
    payload = sealed(BASE_SPEC)
    payload["decision_rules"][0]["value"] = 0.9
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="altered after sealing"):
        load_sealed_spec(tampered)


def test_the_seal_is_verified_before_any_sandbox_or_model_work(tmp_path, monkeypatch):
    """Binding is worthless if the run has already spent money reaching it.

    The finding required the seal check to happen *before* real-cycle work, so
    this pins the ordering directly: an unsealed spec must abort without
    creating a database, a report, or a sandbox, and a sealed one must get
    exactly as far as sandbox construction.
    """
    from grove import experiment

    class ReachedSandbox(Exception):
        pass

    def explode(*_args, **_kwargs):
        raise ReachedSandbox("sandbox construction reached")

    monkeypatch.setattr(experiment, "LxdSandbox", explode)
    database = tmp_path / "cycle.db"
    report = tmp_path / "report.json"

    unsealed = tmp_path / "unsealed.json"
    unsealed.write_text(json.dumps(BASE_SPEC))
    with pytest.raises(ValueError, match="not sealed"):
        experiment.run_first_real_cycle(
            database, report, reset=True, spec_path=unsealed
        )
    assert not database.exists()
    assert not report.exists()

    tampered = tmp_path / "tampered.json"
    payload = sealed(BASE_SPEC)
    payload["decision_rules"][0]["value"] = 0.9
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="altered after sealing"):
        experiment.run_first_real_cycle(
            database, report, reset=True, spec_path=tampered
        )
    assert not database.exists()

    # A sealed spec is not simply always rejected. Before the EXP-002 replay
    # cohort was authored, the shipped spec stopped one step later at the
    # capacity preflight -- still before any sandbox. That ordered gate still
    # exists; a deliberately thinned catalog pins it.
    shipped = _SPEC_DIR / "EXP-002-forced-replay-and-route-precision.json"
    from grove.coding_tasks import coding_catalog

    with pytest.MonkeyPatch.context() as thin:
        thin.setattr(experiment, "coding_catalog", lambda: coding_catalog()[:4])
        with pytest.raises(experiment.ExperimentSetupError, match="impossible"):
            experiment.run_first_real_cycle(
                database, report, reset=True, spec_path=shipped
            )
    assert not database.exists()

    # With the authored catalog the shipped spec passes capacity and gets
    # exactly as far as sandbox construction.
    with pytest.raises(ReachedSandbox):
        experiment.run_first_real_cycle(database, report, reset=True, spec_path=shipped)
    assert not database.exists()

    # With a feasible declared cohort the same sealed spec reaches the sandbox,
    # which proves the preflights are ordered gates and not a blanket refusal.
    monkeypatch.setattr(
        experiment,
        "REAL_CYCLE_POLICY",
        replace(experiment.REAL_CYCLE_POLICY, min_replay_examples=1),
    )
    feasible = json.loads(shipped.read_text())
    feasible["required_setup"]["machine"]["min_replay_examples"] = 1
    feasible_path = tmp_path / "feasible.json"
    feasible_path.write_text(json.dumps(sealed(feasible)))
    with pytest.raises(ReachedSandbox):
        experiment.run_first_real_cycle(
            database, report, reset=True, spec_path=feasible_path
        )
    assert not database.exists()


def test_sandbox_preflight_failure_preserves_database_and_returns_setup_exit(
    tmp_path, monkeypatch, capsys
):
    """An infrastructure refusal must happen before ``--reset`` can delete evidence."""
    from grove import cli, experiment

    class BrokenSandbox:
        def preflight(self):
            raise RuntimeError("sandbox image missing")

    monkeypatch.setattr(
        experiment,
        "REAL_CYCLE_POLICY",
        replace(experiment.REAL_CYCLE_POLICY, min_replay_examples=0),
    )
    monkeypatch.setattr(experiment, "LxdSandbox", BrokenSandbox)
    database = tmp_path / "existing.db"
    database.write_bytes(b"evidence that must survive")

    assert (
        cli.main(
            [
                "--db",
                str(database),
                "real-cycle",
                "--reset",
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )

    assert database.read_bytes() == b"evidence that must survive"
    assert "setup_refused" in capsys.readouterr().err


def test_cli_exit_codes_distinguish_failure_from_unusable(tmp_path):
    spec_path = tmp_path / "spec.json"
    report_path = tmp_path / "report.json"
    spec = sealed(BASE_SPEC)
    spec_path.write_text(json.dumps(spec))

    report_path.write_text(
        json.dumps(report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec))
    )
    assert checker.main(["--spec", str(spec_path), "--report", str(report_path)]) == 0

    report_path.write_text(
        json.dumps(report(forced_rate=0.5, claim="router_shielded", spec=spec))
    )
    assert checker.main(["--spec", str(spec_path), "--report", str(report_path)]) == 1

    report_path.write_text(json.dumps(report(forced_rate=0.5, claim="router_shielded")))
    assert checker.main(["--spec", str(spec_path), "--report", str(report_path)]) == 2


def test_cli_accepts_a_control_report(tmp_path):
    spec = sealed(PAIRED_SPEC)
    spec_path = tmp_path / "spec.json"
    primary = tmp_path / "self.json"
    control = tmp_path / "canonical.json"
    spec_path.write_text(json.dumps(spec))
    primary.write_text(json.dumps(arm(spec=spec, source="self-repair", heldout=0.85)))
    control.write_text(json.dumps(arm(spec=spec, source="canonical", heldout=0.9)))

    assert (
        checker.main(
            [
                "--spec",
                str(spec_path),
                "--report",
                str(primary),
                "--control-report",
                str(control),
            ]
        )
        == 0
    )
    # The same spec without its control arm cannot be judged at all.
    assert checker.main(["--spec", str(spec_path), "--report", str(primary)]) == 2


# --------------------------------------------------------------------------
# Iteration-4 findings: fanout deltas, setup conformance, arm identity,
# a vacuous tolerance, and partial provenance.
# --------------------------------------------------------------------------


def test_a_second_expert_cannot_hide_behind_the_first(tmp_path):
    """Finding 2: `_delta` used only the first entry of each arm.

    The second expert scored 0.0 against a control of 0.8 -- a -0.8 delta --
    and the -0.1 tolerance passed anyway.
    """
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9, expert_ids=("a", "b"))
    primary["experts"][1]["metrics"]["heldout_target_rate"] = 0.0
    bind(primary, spec)
    control = arm(spec=spec, source="canonical", heldout=0.9, expert_ids=("a", "b"))
    control["experts"][1]["metrics"]["heldout_target_rate"] = 0.8
    bind(control, spec)

    verdict = checker.check(spec, primary, control)
    delta_rule = verdict["rules"][0]

    assert delta_rule["passed"] is False
    assert "deltas [0.0, -0.8]" in delta_rule["detail"]
    assert verdict["falsified_hypotheses"] == ["H2"]


def test_delta_pairs_on_the_declared_key_not_on_position():
    spec = sealed(PAIRED_SPEC)
    primary = arm(
        spec=spec,
        source="self-repair",
        heldout=0.9,
        expert_ids=("a", "b"),
        pairing_keys=("key_a", "key_b"),
    )
    primary["experts"][1]["metrics"]["heldout_target_rate"] = 0.5
    bind(primary, spec)
    # Real arms carry different expert ids in a different order. Only the
    # pairing key says which expert is which.
    control = arm(
        spec=spec,
        source="canonical",
        heldout=0.9,
        expert_ids=("q", "r"),
        pairing_keys=("key_b", "key_a"),
    )
    control["experts"][0]["metrics"]["heldout_target_rate"] = 0.55
    bind(control, spec)

    verdict = checker.check(spec, primary, control)

    # key_b:0.5 pairs with key_b:0.55 for -0.05, not with key_a:0.9 for -0.4.
    assert "deltas [0.0, -0.05" in verdict["rules"][0]["detail"]
    assert verdict["rules"][0]["passed"] is True


def test_exp003_random_expert_ids_use_stable_pairing_keys():
    """Finding 1: pairing on ``experts[*].id`` can never match across runs.

    Expert ids are fresh UUIDs, so two arms of the same experiment always carry
    different ones. Pairing on them produced a failed delta rule and a falsified
    H2 -- a broken comparison reported as a scientific result.
    """
    spec = sealed(PAIRED_SPEC)
    primary = arm(
        spec=spec,
        source="self-repair",
        heldout=0.9,
        expert_ids=("expert_5e3ac1f00b21",),
        pairing_keys=("stable_escaped_path",),
    )
    control = arm(
        spec=spec,
        source="canonical",
        heldout=0.95,
        expert_ids=("expert_9b7712de40aa",),
        pairing_keys=("stable_escaped_path",),
    )

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is False
    assert verdict["rules_unevaluable"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert "paired on experts[*].pairing_key" in verdict["rules"][0]["detail"]


def test_unmatched_pairing_keys_are_unusable_not_falsified():
    spec = sealed(PAIRED_SPEC)
    primary = arm(
        spec=spec, source="self-repair", heldout=0.9, pairing_keys=("family_a",)
    )
    control = arm(
        spec=spec, source="canonical", heldout=0.9, pairing_keys=("family_b",)
    )

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    # Every rule, not only the delta: a run that cannot be paired publishes no
    # scientific outcome at all.
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert verdict["rules_failed"] == 0
    # A comparison that could not be made is not a refuted prediction.
    assert verdict["falsified_hypotheses"] == []
    assert any("could not be paired" in item for item in verdict["blockers"])


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ((None, None), "null or empty"),
        (("dup", "dup"), "duplicated across entries"),
    ],
)
def test_absent_or_duplicated_pairing_keys_are_refused(keys, expected):
    spec = sealed(PAIRED_SPEC)
    primary = arm(
        spec=spec,
        source="self-repair",
        heldout=0.9,
        expert_ids=("a", "b"),
        pairing_keys=keys,
    )
    control = arm(
        spec=spec,
        source="canonical",
        heldout=0.9,
        expert_ids=("c", "d"),
        pairing_keys=keys,
    )

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["falsified_hypotheses"] == []
    assert expected in verdict["rules"][0]["detail"]


def test_missing_pairing_key_field_is_refused():
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    del control["experts"][0]["pairing_key"]
    # Resealed on purpose: the run under test never reported the key, so the
    # refusal being checked is the absent pairing key rather than an edit.
    bind(control, spec)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert "absent on at least one entry" in verdict["rules"][0]["detail"]


def test_a_multi_expert_delta_without_a_pairing_key_is_refused():
    naked = copy.deepcopy(PAIRED_SPEC)
    del naked["decision_rules"][0]["pair_on"]
    spec = sealed(naked)
    primary = arm(spec=spec, source="self-repair", heldout=0.9, expert_ids=("a", "b"))
    control = arm(spec=spec, source="canonical", heldout=0.9, expert_ids=("a", "b"))

    verdict = checker.check(spec, primary, control)

    assert verdict["rules"][0]["passed"] is False
    assert "pairing is ambiguous" in verdict["rules"][0]["detail"]


def test_arms_with_different_expert_counts_cannot_be_compared():
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9, expert_ids=("a", "b"))
    control = arm(spec=spec, source="canonical", heldout=0.9, expert_ids=("a",))

    verdict = checker.check(spec, primary, control)

    assert verdict["rules"][0]["passed"] is False
    assert "no pairing is possible" in verdict["rules"][0]["detail"]


def test_a_run_that_ignored_the_declared_setup_is_unusable():
    """Finding 3: a bound report using the wrong correction source passed."""
    spec = sealed(PAIRED_SPEC)
    wrong = arm(spec=spec, source="canonical", heldout=0.9)
    wrong["correction_source"] = "canonical"
    control = arm(spec=spec, source="self-repair", heldout=0.9)

    verdict = checker.check(spec, wrong, control)

    assert verdict["unusable"] is True
    assert verdict["setup_conformance"]["conformant"] is False
    assert verdict["setup_conformance"]["mismatches"] == [
        {"key": "correction_source", "declared": "self-repair", "observed": "canonical"}
    ]


def test_a_spec_with_required_setup_needs_the_report_to_record_it():
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    del primary["run_setup"]
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert "records no run_setup" in verdict["setup_conformance"]["reason"]


def test_prose_setup_is_reported_separately(tmp_path):
    """Finding 5: prose must never count as machine conformance."""
    prose = copy.deepcopy(PAIRED_SPEC)
    prose["required_setup"] = {
        "machine": {"correction_source": "self-repair"},
        "prose": {"replay_authoring_rule": "write them blind"},
    }
    spec = sealed(prose)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["setup_conformance"]["conformant"] is True
    assert verdict["setup_conformance"]["checked"] == {
        "correction_source": "self-repair"
    }
    # Prose is never silently treated as enforced, and never as verified.
    assert verdict["setup_conformance"]["unchecked_prose_keys"] == [
        "replay_authoring_rule"
    ]
    assert verdict["setup_conformance"]["missing_machine_keys"] == []


def test_missing_machine_setup_key_is_unusable():
    """Finding 5: a declared key the run does not record used to pass.

    ``unchecked_keys`` plus ``conformant: true`` meant deleting
    ``correction_source`` from the run setup silently disabled the strongest
    binding in the protocol.
    """
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    del primary["run_setup"]["correction_source"]
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["setup_conformance"]["conformant"] is False
    assert verdict["setup_conformance"]["missing_machine_keys"] == [
        "correction_source"
    ]


def test_missing_multiple_machine_setup_keys_is_unusable():
    strict = copy.deepcopy(PAIRED_SPEC)
    strict["required_setup"] = {
        "machine": {
            "correction_source": "self-repair",
            "self_repair_attempts": 3,
            "min_replay_examples": 50,
        }
    }
    spec = sealed(strict)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    del primary["run_setup"]["self_repair_attempts"]
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["setup_conformance"]["missing_machine_keys"] == [
        "self_repair_attempts",
    ]


def test_a_spec_declaring_min_route_precision_is_unusable():
    """Finding 14: the threshold was accepted and discarded."""
    bogus = copy.deepcopy(PAIRED_SPEC)
    bogus["required_setup"] = {
        "machine": {"correction_source": "self-repair", "min_route_precision": 1.0}
    }
    spec = sealed(bogus)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["setup_conformance"]["unsupported_keys"] == ["min_route_precision"]


def test_the_control_arm_setup_is_validated_too():
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    control["run_setup"]["correction_source"] = "something-else"

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["control_setup_conformance"]["conformant"] is False


def test_a_sealed_spec_with_nothing_in_it_cannot_pass(tmp_path):
    """Finding 3: an empty sealed spec returned exit 0."""
    empty = sealed({"spec_id": "EXP-EMPTY", "decision_rules": [], "hypotheses": []})
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=empty)

    verdict = checker.check(empty, document)

    assert verdict["unusable"] is True
    assert verdict["spec_substance"]["substantive"] is False
    assert "nothing falsifiable" in verdict["verdict"]

    spec_path = tmp_path / "empty.json"
    report_path = tmp_path / "report.json"
    spec_path.write_text(json.dumps(empty))
    report_path.write_text(json.dumps(document))
    assert checker.main(["--spec", str(spec_path), "--report", str(report_path)]) == 2


def test_a_hypothesis_without_a_falsification_condition_is_rejected():
    hollow = sealed(
        {
            "spec_id": "EXP-HOLLOW",
            "hypotheses": [{"id": "H1", "claim": "it works", "falsified_if": "  "}],
            "decision_rules": [
                {
                    "id": "D1",
                    "path": "provenance_gaps",
                    "comparison": "set==",
                    "value": [],
                }
            ],
        }
    )

    verdict = checker.check(hollow, report(forced_rate=0.0, claim="x", spec=hollow))

    assert verdict["unusable"] is True
    assert "hypotheses[].falsified_if" in verdict["spec_substance"]["missing"]


@pytest.mark.parametrize(
    "path",
    [
        "provenance.source.revision",
        "provenance.source.dirty",
        "provenance.worker.host",
        "provenance.models.base.aggregate_sha256",
        "run_setup.admission_policy_sha256",
        "run_setup.cohort_manifest_sha256",
        "run_setup.self_repair_attempts",
    ],
)
def test_every_identity_field_must_match_across_arms(path):
    """Finding 4: arms differing in revision, policy or cohort paired anyway."""
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    section, _, leaf = path.rpartition(".")
    cursor = control
    for part in section.split("."):
        cursor = cursor[part]
    cursor[leaf] = "diverged"

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(item["path"] == path for item in verdict["arm_pairing"]["mismatches"])


@pytest.mark.parametrize("absent", [None, "unavailable: not collected"])
def test_a_null_or_unresolved_identity_field_is_not_a_match(absent):
    """Two nulls agreeing proves nothing about whether the arms match."""
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    primary["provenance"]["models"]["base"]["aggregate_sha256"] = absent
    control["provenance"]["models"]["base"]["aggregate_sha256"] = absent

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(
        item["path"] == "provenance.models.base.aggregate_sha256"
        for item in verdict["arm_pairing"]["mismatches"]
    )


def test_delta_less_than_or_equal_catches_extra_forgetting():
    """Finding 5: `delta>= -1.0` was vacuous because rates lie in [0, 1]."""
    rule_value = 0.0
    worse, _ = checker.evaluate("delta<=", [1.0], rule_value)
    same, _ = checker.evaluate("delta<=", [0.0], rule_value)
    better, _ = checker.evaluate("delta<=", [-0.5], rule_value)
    vacuous, _ = checker.evaluate("delta>=", [1.0], -1.0)

    assert worse is False
    assert same is True
    assert better is True
    # The operator the spec used to declare, for contrast.
    assert vacuous is True


def test_partial_provenance_permits_only_the_declared_gaps():
    """Finding A: `provenance_gaps == []` is unreachable on the control host."""
    permitted = ["models.base.aggregate_sha256", "worker.model_manifest_sha256"]

    exact, _ = checker.evaluate("subset_of", permitted, permitted)
    fewer, _ = checker.evaluate("subset_of", permitted[:1], permitted)
    none_at_all, _ = checker.evaluate("subset_of", [], permitted)
    surprise, detail = checker.evaluate(
        "subset_of", [*permitted, "source.dirty"], permitted
    )

    assert exact is True
    assert fewer is True
    assert none_at_all is True
    assert surprise is False
    assert "undeclared ['source.dirty']" in detail


def test_shipped_specs_declare_their_provenance_gaps_rather_than_demanding_none():
    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        gap_rules = [
            rule for rule in spec["decision_rules"] if rule["path"] == "provenance_gaps"
        ]
        assert gap_rules, path.name
        for rule in gap_rules:
            # An empty-set demand would be unsatisfiable on this host.
            assert rule["comparison"] == "subset_of", path.name
            assert rule["value"], path.name


# --------------------------------------------------------------------------
# Finding 3: arm identity must cover actual run state, not just declared intent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "provenance.source.status_sha256",
        "provenance.source.worktree_sha256",
        "provenance.worker.framework_versions_sha256",
        "provenance.worker.checkout.revision",
        "provenance.worker.checkout.tree",
        "provenance.worker.checkout.dirty",
        "provenance.sandbox_image.fingerprint",
        "run_setup.actual_training_failure_ids",
        "run_setup.actual_training_failure_set_sha256",
        "run_setup.attempted_training_failure_set_sha256",
    ],
)
def test_actual_arm_identity_mismatch_blocks_pairing(path):
    """Two arms that agree on every declared field can still differ in fact.

    The cohort hash names the catalog, fixed before either arm runs. Uncommitted
    edits, worker framework drift, a different sandbox image and a different
    actual failure set are all invisible to it.
    """
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    section, _, leaf = path.rpartition(".")
    cursor = control
    for part in section.split("."):
        cursor = cursor[part]
    cursor[leaf] = "diverged"

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(item["path"] == path for item in verdict["arm_pairing"]["mismatches"])


# --------------------------------------------------------------------------
# Finding 4: a declared partial-provenance gap must not break pairing
# --------------------------------------------------------------------------


GAP = "models.base.aggregate_sha256"


def _gap_spec() -> dict:
    permitted = copy.deepcopy(PAIRED_SPEC)
    permitted["permitted_provenance_gaps"] = [GAP]
    return sealed(permitted)


def test_declared_partial_gap_does_not_block_arm_pairing():
    """The base model lives on the worker; neither arm can hash it.

    Treating that as a mismatch made every honest EXP-003 pair unusable. It is
    permitted only when the spec declared it and *both* arms report it.
    """
    spec = _gap_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    for document in (primary, control):
        document["provenance"]["models"]["base"]["aggregate_sha256"] = (
            "unavailable: no model path supplied"
        )
        document["provenance_gaps"] = [GAP]

    verdict = checker.check(spec, primary, control)

    assert verdict["arm_pairing"]["paired"] is True
    assert verdict["arm_pairing"]["permitted_gaps_used"] == [GAP]
    # Pairing through a gap is partial provenance and says so.
    assert verdict["arm_pairing"]["provenance_completeness"] == "partial"


def test_undeclared_identity_gap_blocks_arm_pairing():
    spec = _gap_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    for document in (primary, control):
        document["provenance"]["worker"]["checkout"]["revision"] = (
            "unavailable: worker does not report checkout revision"
        )
        document["provenance_gaps"] = ["worker.checkout.revision"]

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(
        item["path"] == "provenance.worker.checkout.revision"
        for item in verdict["arm_pairing"]["mismatches"]
    )


def test_one_resolved_one_unresolved_gap_blocks_pairing():
    spec = _gap_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    control["provenance"]["models"]["base"]["aggregate_sha256"] = (
        "unavailable: no model path supplied"
    )
    control["provenance_gaps"] = [GAP]

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    mismatch = next(
        item
        for item in verdict["arm_pairing"]["mismatches"]
        if item["path"] == "provenance." + GAP
    )
    assert mismatch["reason"] == "resolved on one arm and unresolved on the other"


def test_a_permitted_gap_one_arm_does_not_report_blocks_pairing():
    """A spec may permit a gap; an arm still has to admit it has one."""
    spec = _gap_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    for document in (primary, control):
        document["provenance"]["models"]["base"]["aggregate_sha256"] = None
    primary["provenance_gaps"] = [GAP]
    control["provenance_gaps"] = []

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True


# --------------------------------------------------------------------------
# Finding 6: declared_before_run proved no timing
# --------------------------------------------------------------------------


def test_declared_before_run_without_attestation_is_unverified():
    spec = sealed(BASE_SPEC)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)

    verdict = checker.check(spec, document)

    assert verdict["timing"]["claim"] == "seal_self_consistent"
    assert verdict["timing"]["status"] == "unverified"
    assert verdict["timing"]["usable_for_preregistration_claim"] is False
    assert verdict["unusable"] is False


def test_preregistration_claim_without_timestamp_refuses_grading():
    claiming = copy.deepcopy(BASE_SPEC)
    claiming["declared_before_run"] = True
    spec = sealed(claiming)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)

    verdict = checker.check(spec, document)

    assert verdict["unusable"] is True
    assert any("timing claim" in item for item in verdict["blockers"])


def test_fabricated_rfc3161_map_is_not_verified():
    """Finding 3, the reviewer's exact input.

    ``{"type": "rfc3161", "timestamp": "0000-not-a-time"}`` used to return
    ``verified: true``. Nothing parsed the timestamp, read a token, checked a
    signature or compared against a real run start. A field anyone can type is
    not evidence.
    """
    verdict = checker.timing_claim(
        {
            "timing_attestation": {
                "type": "rfc3161",
                "timestamp": "0000-not-a-time",
            }
        },
        {},
    )

    assert verdict["verified"] is False
    assert verdict["usable_for_preregistration_claim"] is False
    assert verdict["status"] == "attestation_verifier_not_implemented"
    assert verdict["blocking"] is True


def test_invalid_timestamp_is_not_compared_lexically():
    """Two timestamps are never ordered as strings, so nothing can be smuggled.

    A garbage timestamp and a plausible one reach the same refusal, because the
    timestamp is not consulted at all until a verifier can derive it from a
    signed artifact.
    """
    garbage = checker.timing_claim(
        {"timing_attestation": {"type": "rfc3161", "timestamp": "0000-not-a-time"}},
        {"run_started_at": "2026-08-07T09:00:00+00:00"},
    )
    plausible = checker.timing_claim(
        {
            "timing_attestation": {
                "type": "rfc3161",
                "timestamp": "2026-08-01T00:00:00+00:00",
                "token_sha256": "a" * 64,
            }
        },
        {"run_started_at": "2026-08-07T09:00:00+00:00"},
    )

    assert garbage["verified"] is False
    assert plausible["verified"] is False
    assert garbage["status"] == plausible["status"]


def test_preregistered_claim_without_verifiable_artifact_refuses():
    """A self-reported attestation map cannot make a run gradable."""
    attested = copy.deepcopy(BASE_SPEC)
    attested["declared_before_run"] = True
    attested["timing_attestation"] = {
        "type": "rfc3161",
        "timestamp": "2026-08-01T00:00:00+00:00",
        "token_sha256": "a" * 64,
    }
    spec = sealed(attested)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    document["run_started_at"] = "2026-08-07T09:00:00+00:00"

    verdict = checker.check(spec, document)

    assert verdict["unusable"] is True
    assert verdict["timing"]["verified"] is False
    assert verdict["timing"]["usable_for_preregistration_claim"] is False
    assert "no verifier in this repository" in verdict["timing"]["reason"]


def test_an_unknown_attestation_type_is_refused():
    attested = copy.deepcopy(BASE_SPEC)
    attested["timing_attestation"] = {"type": "a-friend-told-me"}
    spec = sealed(attested)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)

    verdict = checker.check(spec, document)

    assert verdict["unusable"] is True
    assert verdict["timing"]["status"] == "unsupported_attestation"


@pytest.mark.parametrize(
    "attestation",
    [
        {"type": "rfc3161", "timestamp": "0000-not-a-time"},
        {"type": "signed_tag", "tag": "v1"},
        {"type": "transparency_log", "log_index": 7},
    ],
)
def test_preflight_rejects_unverifiable_preregistration_before_cost(
    tmp_path, no_sandbox, attestation
):
    from grove import experiment

    claiming = copy.deepcopy(BASE_SPEC)
    claiming["declared_before_run"] = True
    claiming["timing_attestation"] = attestation
    path = tmp_path / "claiming.json"
    path.write_text(json.dumps(sealed(claiming)))
    database = tmp_path / "cycle.db"

    with pytest.raises(experiment.ExperimentSetupError, match="no verifier"):
        experiment.run_first_real_cycle(
            database, tmp_path / "cycle.json", reset=True, spec_path=path
        )

    assert not database.exists()


def test_shipped_specs_remain_seal_self_consistent_and_untimed():
    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        assert "declared_before_run" not in spec, path.name
        assert spec.get("preregistered") is None, path.name
        assert spec["seal_self_consistent"] is True, path.name
        assert spec["timing_attestation"] is None, path.name



# --------------------------------------------------------------------------
# Finding 7: report metrics must be bound to the run that produced them
# --------------------------------------------------------------------------


def _sealed_arm(spec: dict, source: str, **kwargs) -> dict:
    """An arm report carrying a valid run manifest, as the runner writes it."""
    from grove.experiment import seal_report

    document = arm(spec=spec, source=source, **kwargs)
    return seal_report(document, spec=spec)


def _integrity_spec() -> dict:
    strict = copy.deepcopy(PAIRED_SPEC)
    strict["requires_report_integrity"] = True
    return sealed(strict)


def test_editing_report_metrics_breaks_integrity_binding():
    """Finding 7: editing a metric turned a falsified run into a passing one."""
    spec = _integrity_spec()
    primary = _sealed_arm(spec, "self-repair", heldout=0.5)
    control = _sealed_arm(spec, "canonical", heldout=0.9)

    honest = checker.check(spec, primary, control)
    assert honest["unusable"] is False
    assert honest["rules_failed"] == 1

    primary["experts"][0]["metrics"]["heldout_target_rate"] = 0.9
    tampered = checker.check(spec, primary, control)

    assert tampered["unusable"] is True
    assert tampered["report_integrity"]["status"] == "tampered"
    assert any("metrics were edited" in item for item in tampered["blockers"])


def test_editing_provenance_breaks_provenance_hash():
    spec = _integrity_spec()
    primary = _sealed_arm(spec, "self-repair", heldout=0.9)
    control = _sealed_arm(spec, "canonical", heldout=0.9)
    primary["provenance"]["source"]["revision"] = "f" * 40

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert (
        verdict["report_integrity"]["checks"]["provenance_sha256"]["intact"] is False
    )


def test_editing_the_run_setup_breaks_the_manifest_binding():
    spec = _integrity_spec()
    primary = _sealed_arm(spec, "self-repair", heldout=0.9)
    control = _sealed_arm(spec, "canonical", heldout=0.9)
    primary["run_setup"]["actual_training_failure_ids"] = ["train_a"]

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert "run_setup_sha256" in verdict["report_integrity"]["reason"]


def test_editing_trained_failure_set_breaks_explicit_manifest_binding():
    spec = _integrity_spec()
    primary = _sealed_arm(spec, "self-repair", heldout=0.9)
    control = _sealed_arm(spec, "canonical", heldout=0.9)
    primary["run_setup"]["trained_failure_ids"] = ["train_a"]
    primary["run_setup"]["trained_failure_set_sha256"] = checker.canonical_hash(
        ["train_a"]
    )

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(
        "trained_failure_set_sha256" in problem
        for problem in verdict["report_integrity"]["problems"]
    )

def test_unbound_report_is_not_authoritative():
    """A report with no manifest fails a spec that demands integrity."""
    spec = _integrity_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = _sealed_arm(spec, "canonical", heldout=0.9)
    primary.pop("run_manifest", None)
    primary.pop("run_manifest_sha256", None)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "unbound"


def test_an_intact_sealed_report_passes_integrity():
    spec = _integrity_spec()
    primary = _sealed_arm(spec, "self-repair", heldout=0.9)
    control = _sealed_arm(spec, "canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["report_integrity"]["status"] == "intact"
    assert verdict["control_report_integrity"]["status"] == "intact"
    assert verdict["unusable"] is False


def test_shipped_specs_require_report_integrity():
    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        assert spec["requires_report_integrity"] is True, path.name


# --------------------------------------------------------------------------
# The checker and the library must agree, or a binding silently breaks
# --------------------------------------------------------------------------


def test_checker_and_library_canonical_hashes_agree():
    from grove.provenance import canonical_hash

    payloads = [
        {},
        {"b": 1, "a": [1, 2, {"c": None}]},
        {"float": 0.25, "bool": True, "null": None},
    ]
    for payload in payloads:
        assert checker.canonical_hash(payload) == canonical_hash(payload)


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        {},
        {"correction_source": "self-repair"},
        {"machine": {"a": 1}, "prose": {"b": "text"}},
        {"machine": {"a": 1}},
        {"prose": {"b": "text"}},
    ],
)
def test_checker_and_runner_split_required_setup_the_same_way(declaration):
    from grove.experiment import normalize_required_setup

    assert checker.normalize_required_setup(declaration) == normalize_required_setup(
        declaration
    )


# --------------------------------------------------------------------------
# Findings 2, 5 and 11: refuse before the run spends anything
# --------------------------------------------------------------------------


@pytest.fixture
def no_sandbox(monkeypatch):
    """Make any sandbox construction an unmistakable, loud failure."""
    from grove import experiment

    class ReachedSandbox(Exception):
        pass

    def explode(*_args, **_kwargs):
        raise ReachedSandbox("sandbox construction reached")

    monkeypatch.setattr(experiment, "LxdSandbox", explode)
    return ReachedSandbox


@pytest.fixture
def thin_catalog(monkeypatch):
    """Shrink the catalog below the declared cohort so capacity still refuses.

    The authored EXP-002 replay cohort made the shipped catalog feasible, so
    these ordering tests pin the capacity gate against a deliberately thinned
    catalog instead of live worker or sandbox infrastructure.
    """
    from grove import experiment
    from grove.coding_tasks import coding_catalog

    catalog = coding_catalog()[:4]
    monkeypatch.setattr(experiment, "coding_catalog", lambda: catalog)


def _exp003() -> Path:
    return _SPEC_DIR / "EXP-003-correction-source-ab.json"


def test_exp003_canonical_control_uses_control_setup(
    tmp_path, no_sandbox, thin_catalog
):
    """Finding 2: the canonical control could not launch under its own spec.

    ``required_setup`` declares self-repair, so validating the control arm
    against it produced a contradiction before anything ran. With the control
    profile selected, the canonical arm reaches the *next* gate -- the replay
    capacity refusal -- which is the honest blocker.
    """
    from grove import experiment

    database = tmp_path / "control.db"
    report_path = tmp_path / "control.json"

    with pytest.raises(experiment.ExperimentSetupError) as error:
        experiment.run_first_real_cycle(
            database,
            report_path,
            reset=True,
            correction_source="canonical",
            compare_corrections=True,
            spec_path=_exp003(),
            arm="control",
        )

    assert "replay cohort is impossible" in str(error.value)
    assert "correction_source declared" not in str(error.value)
    assert not database.exists()
    assert not report_path.exists()


def test_exp003_control_arm_is_inferred_from_the_correction_source(
    tmp_path, no_sandbox, thin_catalog
):
    from grove import experiment

    database = tmp_path / "control.db"
    with pytest.raises(experiment.ExperimentSetupError, match="replay cohort"):
        experiment.run_first_real_cycle(
            database,
            tmp_path / "control.json",
            reset=True,
            correction_source="canonical",
            compare_corrections=True,
            spec_path=_exp003(),
        )
    assert not database.exists()


def test_exp003_wrong_arm_setup_fails_before_costly_work(tmp_path, no_sandbox):
    """Asking for the control arm while running self-repair must be refused."""
    from grove import experiment

    database = tmp_path / "wrong.db"
    with pytest.raises(experiment.ExperimentSetupError) as error:
        experiment.run_first_real_cycle(
            database,
            tmp_path / "wrong.json",
            reset=True,
            correction_source="self-repair",
            compare_corrections=True,
            spec_path=_exp003(),
            arm="control",
        )

    assert "correction_source declared 'canonical'" in str(error.value)
    assert not database.exists()


def test_an_unknown_arm_name_is_refused(tmp_path, no_sandbox):
    from grove import experiment

    with pytest.raises(experiment.ExperimentSetupError, match="no 'sidecar' arm"):
        experiment.run_first_real_cycle(
            tmp_path / "x.db",
            tmp_path / "x.json",
            reset=True,
            correction_source="canonical",
            spec_path=_exp003(),
            arm="sidecar",
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"correction_source": "telepathy"}, "unknown correction source"),
        ({"self_repair_attempts": 0}, "at least 1"),
        ({"self_repair_attempts": -3}, "at least 1"),
        ({"self_repair_attempts": "three"}, "must be an integer"),
        ({"self_repair_attempts": True}, "must be an integer"),
    ],
)
def test_invalid_correction_settings_fail_before_sandbox(
    tmp_path, no_sandbox, kwargs, expected
):
    """Finding 11: these were caught only after a full live capture pass."""
    from grove import experiment

    database = tmp_path / "cycle.db"
    report_path = tmp_path / "cycle.json"
    arguments = {"correction_source": "canonical", **kwargs}

    with pytest.raises(experiment.ExperimentSetupError, match=expected):
        experiment.run_first_real_cycle(
            database, report_path, reset=True, **arguments
        )

    assert not database.exists()
    assert not report_path.exists()


def test_hollow_sealed_spec_fails_before_sandbox(tmp_path, no_sandbox):
    from grove import experiment

    hollow = tmp_path / "hollow.json"
    hollow.write_text(
        json.dumps(sealed({"spec_id": "EXP-HOLLOW", "decision_rules": [], "hypotheses": []}))
    )
    database = tmp_path / "cycle.db"

    with pytest.raises(experiment.ExperimentSetupError) as error:
        experiment.run_first_real_cycle(
            database, tmp_path / "cycle.json", reset=True, spec_path=hollow
        )

    assert "no decision_rules" in str(error.value)
    assert "no hypotheses" in str(error.value)
    assert not database.exists()


def test_a_spec_claiming_untimed_preregistration_fails_before_sandbox(
    tmp_path, no_sandbox
):
    from grove import experiment

    claiming = copy.deepcopy(BASE_SPEC)
    claiming["declared_before_run"] = True
    path = tmp_path / "claiming.json"
    path.write_text(json.dumps(sealed(claiming)))

    with pytest.raises(
        experiment.ExperimentSetupError, match="preregistration timing"
    ):
        experiment.run_first_real_cycle(
            tmp_path / "cycle.db", tmp_path / "cycle.json", reset=True, spec_path=path
        )


def test_a_spec_declaring_min_route_precision_fails_before_sandbox(
    tmp_path, no_sandbox
):
    from grove import experiment

    bogus = copy.deepcopy(BASE_SPEC)
    bogus["required_setup"] = {"machine": {"min_route_precision": 1.0}}
    path = tmp_path / "bogus.json"
    path.write_text(json.dumps(sealed(bogus)))

    with pytest.raises(experiment.ExperimentSetupError, match="min_route_precision"):
        experiment.run_first_real_cycle(
            tmp_path / "cycle.db", tmp_path / "cycle.json", reset=True, spec_path=path
        )


def test_missing_machine_setup_key_refuses_before_sandbox(tmp_path, no_sandbox):
    """Finding 5, runner side: an unrecordable declaration must not run."""
    from grove import experiment

    strict = copy.deepcopy(BASE_SPEC)
    strict["required_setup"] = {"machine": {"a_key_the_runner_never_records": 1}}
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(sealed(strict)))
    database = tmp_path / "cycle.db"

    with pytest.raises(experiment.ExperimentSetupError) as error:
        experiment.run_first_real_cycle(
            database, tmp_path / "cycle.json", reset=True, spec_path=path
        )

    assert "does not record" in str(error.value)
    assert not database.exists()


def test_preflight_accepts_a_valid_run():
    """The preflight is a gate, not a blanket refusal."""
    from grove.experiment import preflight_experiment

    result = preflight_experiment(
        json.loads(_exp003().read_text()),
        correction_source="self-repair",
        self_repair_attempts=3,
        compare_corrections=True,
        arm="primary",
    )

    assert result["available_arms"] == ["control", "primary"]


# --------------------------------------------------------------------------
# Finding 1: every value a decision rule reads must be bound
# --------------------------------------------------------------------------


def _exp003_spec() -> dict:
    return json.loads(_exp003().read_text())


def _exp003_arm(spec: dict, *, source: str, arm_name: str, heldout: float) -> dict:
    """An EXP-003-shaped report, sealed the way the runner seals one."""
    from grove.experiment import seal_report
    from grove.provenance import canonical_hash

    provenance = _provenance()
    provenance["extra"] = {"correction_source": source}
    provenance["models"]["base"]["aggregate_sha256"] = (
        "unavailable: no model path supplied"
    )
    provenance["provenance_sha256"] = canonical_hash(provenance)
    setup = _run_setup(source)
    setup["arm"] = arm_name
    document = {
        "arm": arm_name,
        "correction_source": source,
        "provenance": provenance,
        "provenance_gaps": ["models.base.aggregate_sha256"],
        "run_setup": setup,
        "experiment_spec": {
            "spec_id": spec["spec_id"],
            checker.SPEC_HASH_FIELD: spec[checker.SPEC_HASH_FIELD],
        },
        "correction_comparison": {
            "sources": ["canonical-reference-v1", "self-repair-v1"],
            "per_source": {"self-repair-v1": {"verified_rate": 0.75}},
            "training_proposal_reuse": {"enabled": True},
        },
        "cycle": {"experts_admitted": [f"expert_{arm_name}"]},
        "training_proposals": [],
        "experts": [
            {
                "id": f"expert_{arm_name}",
                "pairing_key": "stable_escaped_path",
                "metrics": {
                    "pairing_key": "stable_escaped_path",
                    "heldout_target_rate": heldout,
                    "forced_regression_rate": 0.0,
                },
            }
        ],
    }
    return seal_report(document, spec=spec)


def _exp003_pair(*, heldout: float = 0.9) -> tuple[dict, dict, dict]:
    spec = _exp003_spec()
    primary = _exp003_arm(
        spec, source="self-repair", arm_name="primary", heldout=heldout
    )
    control = _exp003_arm(spec, source="canonical", arm_name="control", heldout=0.9)
    return spec, primary, control


def test_manifest_binds_every_spec_rule_path():
    """The bound set is derived from the rules, so it cannot fall behind them."""
    from grove.experiment import decision_rule_input_paths

    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        expected = decision_rule_input_paths(spec)
        assert expected, path.name
        assert expected == checker.decision_rule_input_paths(spec), path.name
        arm_name = "primary"
        document = _exp003_arm(
            spec, source="self-repair", arm_name=arm_name, heldout=0.9
        )
        bound = set(document["run_manifest"]["decision_inputs"])
        assert bound == set(expected), path.name


def test_an_intact_exp003_pair_is_usable():
    """The guardrail must still let an honest pair through."""
    spec, primary, control = _exp003_pair()

    verdict = checker.check(spec, primary, control)

    assert verdict["report_integrity"]["status"] == "intact"
    assert verdict["control_report_integrity"]["status"] == "intact"
    assert verdict["arm_pairing"]["paired"] is True
    assert verdict["unusable"] is False


def test_editing_exp003_correction_comparison_is_tampered():
    """Finding 1: a failing self-repair yield was edited into a passing one."""
    spec = _exp003_spec()
    primary = _exp003_arm(spec, source="self-repair", arm_name="primary", heldout=0.9)
    primary["correction_comparison"]["per_source"]["self-repair-v1"][
        "verified_rate"
    ] = 0.10
    seal_honestly = copy.deepcopy(primary)
    control = _exp003_arm(spec, source="canonical", arm_name="control", heldout=0.9)

    # Resealed at 0.10 the run is a recorded, honest failure of H1.
    from grove.experiment import seal_report

    seal_report(seal_honestly, spec=spec)
    honest = checker.check(spec, seal_honestly, control)
    assert honest["unusable"] is False
    assert honest["falsified_hypotheses"] == ["H1"]

    # Editing the same value without resealing must not become a pass.
    tampered = copy.deepcopy(seal_honestly)
    tampered["correction_comparison"]["per_source"]["self-repair-v1"][
        "verified_rate"
    ] = 0.95
    verdict = checker.check(spec, tampered, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "tampered"
    assert verdict["falsified_hypotheses"] == []
    assert any(
        "correction_comparison.per_source.self-repair-v1.verified_rate" in problem
        for problem in verdict["report_integrity"]["problems"]
    )


def test_editing_exp003_cycle_admission_is_tampered():
    """Finding 1: fabricated admitted-expert ids satisfied D5."""
    spec, primary, control = _exp003_pair()
    primary["cycle"]["experts_admitted"] = ["fabricated", "fabricated2"]

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "tampered"
    assert any(
        "cycle.experts_admitted" in problem
        for problem in verdict["report_integrity"]["problems"]
    )


def test_emptying_exp003_cycle_admission_is_tampered():
    spec, primary, control = _exp003_pair()
    primary["cycle"]["experts_admitted"] = []

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "tampered"


def test_editing_exp003_top_level_pairing_key_is_tampered():
    """``pair_on`` dereferences this path, so it has to be bound too."""
    spec, primary, control = _exp003_pair()
    primary["experts"][0]["pairing_key"] = "some_other_family"

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "tampered"
    assert any(
        "experts[*].pairing_key" in problem
        for problem in verdict["report_integrity"]["problems"]
    )


def test_editing_a_control_arm_rule_input_is_tampered():
    """A control arm is graded too, so its inputs are bound the same way."""
    spec, primary, control = _exp003_pair()
    control["cycle"]["experts_admitted"] = ["fabricated"]

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["control_report_integrity"]["status"] == "tampered"


def test_missing_decision_input_binding_is_unusable():
    """A manifest that simply omits a rule input must not be accepted."""
    spec, primary, control = _exp003_pair()
    del primary["run_manifest"]["decision_inputs"]["cycle.experts_admitted"]
    primary["run_manifest_sha256"] = checker.canonical_hash(primary["run_manifest"])

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(
        "cycle.experts_admitted is not bound" in problem
        for problem in verdict["report_integrity"]["problems"]
    )


def test_a_manifest_without_any_decision_inputs_is_unusable():
    spec, primary, control = _exp003_pair()
    del primary["run_manifest"]["decision_inputs"]
    primary["run_manifest_sha256"] = checker.canonical_hash(primary["run_manifest"])

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(
        "binds no decision inputs" in problem
        for problem in verdict["report_integrity"]["problems"]
    )


def test_reordering_experts_breaks_the_decision_input_binding():
    """Array order is part of the value a delta rule pairs on."""
    spec = _exp003_spec()
    primary = _exp003_arm(spec, source="self-repair", arm_name="primary", heldout=0.9)
    primary["experts"].append(
        {
            "id": "expert_second",
            "pairing_key": "second_family",
            "metrics": {
                "pairing_key": "second_family",
                "heldout_target_rate": 0.4,
                "forced_regression_rate": 0.0,
            },
        }
    )
    from grove.experiment import seal_report

    seal_report(primary, spec=spec)
    control = _exp003_arm(spec, source="canonical", arm_name="control", heldout=0.9)

    reordered = copy.deepcopy(primary)
    reordered["experts"].reverse()
    verdict = checker.check(spec, reordered, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "tampered"


# --------------------------------------------------------------------------
# One strict canonical encoder for every commitment
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"b": 1, "a": [1, 2, {"c": None}]},
        {"float": 0.25, "bool": True, "null": None},
        {"unicode": "café — ünïcodé"},
        {"nested": [[1, [2, [3]]]]},
    ],
)
def test_checker_and_library_canonical_encoders_agree(payload):
    from grove.provenance import canonical_hash, canonical_json

    assert checker.canonical_json(payload) == canonical_json(payload)
    assert checker.canonical_hash(payload) == canonical_hash(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"path": Path("/tmp/x")},
        {"when": object()},
        {"nan": float("nan")},
        {"inf": float("inf")},
        {1: "int key"},
    ],
)
def test_unrepresentable_values_are_rejected_not_stringified(payload):
    """``default=str`` hashed a repr as if it were data. Both sides refuse now."""
    from grove.provenance import UnrepresentableValue, canonical_hash

    with pytest.raises((UnrepresentableValue, TypeError)):
        canonical_hash(payload)
    with pytest.raises((checker.UnrepresentableValue, TypeError)):
        checker.canonical_hash(payload)


def test_spec_digest_uses_the_shared_canonical_encoder():
    from grove.provenance import canonical_hash

    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        payload = {
            key: value
            for key, value in spec.items()
            if key != checker.SPEC_HASH_FIELD
        }
        assert canonical_hash(payload) == spec[checker.SPEC_HASH_FIELD], path.name


@pytest.mark.parametrize(
    "path",
    [
        "experts[*].metrics.heldout_target_rate",
        "experts[*].pairing_key",
        "cycle.experts_admitted",
        "correction_comparison.per_source.self-repair-v1.verified_rate",
        "absent.path",
        "experts[*].absent",
    ],
)
def test_checker_and_library_resolvers_agree(path):
    """A sealing resolver that disagrees would bind one value and check another.

    The two carry different absence sentinels, so both are normalised to the
    same marker before comparison: what has to match is the shape, the values
    and *where* the holes are.
    """
    from grove.experiment import MISSING_INPUT, resolve_report_path

    def normalise(value, missing):
        if value is missing:
            return "<absent>"
        if isinstance(value, list):
            return [normalise(item, missing) for item in value]
        return value

    _, primary, _ = _exp003_pair()
    library = normalise(resolve_report_path(primary, path), MISSING_INPUT)
    script = normalise(checker.resolve(primary, path), checker.MISSING)

    assert library == script
    # And the binding both sides derive from it agrees too.
    from grove.experiment import decision_input_bindings

    assert decision_input_bindings(primary, [path])[path] == (
        checker.decision_input_binding(primary, path)
    )


# --------------------------------------------------------------------------
# Finding 2: worker content and model identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "provenance.worker.checkout.status_sha256",
        "provenance.worker.checkout.worktree_sha256",
        "provenance.worker.model_manifest_sha256",
    ],
)
def test_worker_content_and_model_mismatch_blocks_pairing(path):
    """A worker can report the same revision while holding different bytes."""
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    section, _, leaf = path.rpartition(".")
    cursor = control
    for part in section.split("."):
        cursor = cursor[part]
    cursor[leaf] = "diverged"

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(item["path"] == path for item in verdict["arm_pairing"]["mismatches"])


def test_unresolved_worker_model_manifest_blocks_exp003_even_if_declared_a_gap():
    """A required identity path can never be waived by a permitted gap."""
    lenient = copy.deepcopy(PAIRED_SPEC)
    lenient["permitted_provenance_gaps"] = [
        "models.base.aggregate_sha256",
        "worker.model_manifest_sha256",
    ]
    lenient["required_resolved_identity"] = [
        "provenance.worker.model_manifest_sha256"
    ]
    spec = sealed(lenient)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    for document in (primary, control):
        document["provenance"]["worker"]["model_manifest_sha256"] = (
            "unavailable: worker does not return model file digests"
        )
        document["provenance_gaps"] = ["worker.model_manifest_sha256"]

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    mismatch = next(
        item
        for item in verdict["arm_pairing"]["mismatches"]
        if item["path"] == "provenance.worker.model_manifest_sha256"
    )
    assert mismatch["reason"] == (
        "worker.model_manifest_sha256 is required for paired identity but is "
        "unavailable"
    )


def test_an_unresolved_required_identity_withdraws_every_other_waiver():
    """A base-model gap is tolerable only while some model identity resolves."""
    lenient = copy.deepcopy(PAIRED_SPEC)
    lenient["permitted_provenance_gaps"] = ["models.base.aggregate_sha256"]
    lenient["required_resolved_identity"] = [
        "provenance.worker.model_manifest_sha256"
    ]
    spec = sealed(lenient)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    for document in (primary, control):
        document["provenance"]["worker"]["model_manifest_sha256"] = (
            "unavailable: worker does not return model file digests"
        )
        document["provenance"]["models"]["base"]["aggregate_sha256"] = (
            "unavailable: no model path supplied"
        )
        document["provenance_gaps"] = ["models.base.aggregate_sha256"]

    verdict = checker.check(spec, primary, control)

    paths = {item["path"] for item in verdict["arm_pairing"]["mismatches"]}
    assert "provenance.worker.model_manifest_sha256" in paths
    assert "provenance.models.base.aggregate_sha256" in paths
    assert verdict["arm_pairing"]["permitted_gaps_used"] == []


def test_permitted_base_model_gap_still_pairs_when_worker_manifest_matches():
    """The closed partial-provenance rule survives the stricter identity."""
    strict = copy.deepcopy(PAIRED_SPEC)
    strict["permitted_provenance_gaps"] = ["models.base.aggregate_sha256"]
    strict["required_resolved_identity"] = [
        "provenance.worker.model_manifest_sha256"
    ]
    spec = sealed(strict)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    for document in (primary, control):
        document["provenance"]["models"]["base"]["aggregate_sha256"] = (
            "unavailable: no model path supplied"
        )
        document["provenance_gaps"] = ["models.base.aggregate_sha256"]

    verdict = checker.check(spec, primary, control)

    assert verdict["arm_pairing"]["paired"] is True
    assert verdict["arm_pairing"]["permitted_gaps_used"] == [
        "models.base.aggregate_sha256"
    ]
    assert verdict["arm_pairing"]["provenance_completeness"] == "partial"


def test_exp003_no_longer_permits_the_worker_model_manifest_gap():
    spec = _exp003_spec()

    assert spec["permitted_provenance_gaps"] == ["models.base.aggregate_sha256"]
    assert spec["required_resolved_identity"] == [
        "provenance.worker.model_manifest_sha256"
    ]
    assert checker.required_resolved_identity(spec) == [
        "provenance.worker.model_manifest_sha256"
    ]


def test_exp002_still_reports_the_worker_manifest_as_a_partial_gap():
    """A single-arm report may still declare the gap it cannot close."""
    spec = json.loads(
        (_SPEC_DIR / "EXP-002-forced-replay-and-route-precision.json").read_text()
    )

    assert "worker.model_manifest_sha256" in spec["permitted_provenance_gaps"]
    assert checker.required_resolved_identity(spec) == []


def test_exp003_refuses_an_unresolved_worker_model_manifest_before_cost(
    tmp_path, no_sandbox, monkeypatch
):
    """Finding 2, runner side: refuse before the database, sandbox and training."""
    from dataclasses import replace as replace_dataclass

    from grove import experiment

    class BareWorker:
        config = type("Config", (), {"host": "fake-worker"})()

        def preflight(self, model_path=None):
            return {
                "status": "ok",
                "python": "3.14.3",
                "mlx": "0.32.0",
                "mlx_lm": "0.31.3",
                "checkout": {
                    "revision": "1" * 40,
                    "tree": "2" * 40,
                    "dirty": False,
                    "status_sha256": "3" * 64,
                    "worktree_sha256": "4" * 64,
                },
                # No model manifest: exactly the worker Grove has today.
            }

    class FakeTrainer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.worker = BareWorker()

    monkeypatch.setattr(experiment, "MlxLoraTrainer", FakeTrainer)
    # A feasible cohort, so the run reaches the identity gate instead of the
    # capacity gate. Nothing about identity depends on cohort size.
    monkeypatch.setattr(
        experiment,
        "REAL_CYCLE_POLICY",
        replace_dataclass(experiment.REAL_CYCLE_POLICY, min_replay_examples=1),
    )
    feasible = _exp003_spec()
    feasible["required_setup"]["machine"]["min_replay_examples"] = 1
    path = tmp_path / "exp003-feasible.json"
    path.write_text(json.dumps(sealed(feasible)))
    database = tmp_path / "cycle.db"
    report_path = tmp_path / "cycle.json"

    with pytest.raises(experiment.ExperimentSetupError) as error:
        experiment.run_first_real_cycle(
            database,
            report_path,
            reset=True,
            correction_source="self-repair",
            compare_corrections=True,
            spec_path=path,
            arm="primary",
        )

    assert "model_manifest_sha256 is required for paired identity" in str(error.value)
    assert not database.exists()
    assert not report_path.exists()


def test_a_worker_that_supplies_a_model_manifest_passes_the_identity_gate(
    tmp_path, no_sandbox, monkeypatch
):
    """The gate is a gate, not a blanket refusal."""
    from dataclasses import replace as replace_dataclass

    from grove import experiment

    class ManifestWorker:
        config = type("Config", (), {"host": "fake-worker"})()

        def preflight(self, model_path=None):
            return {
                "status": "ok",
                "python": "3.14.3",
                "mlx": "0.32.0",
                "mlx_lm": "0.31.3",
                "checkout": {
                    "revision": "1" * 40,
                    "tree": "2" * 40,
                    "dirty": False,
                    "status_sha256": "3" * 64,
                    "worktree_sha256": "4" * 64,
                },
                "model_manifest_sha256": "5" * 64,
            }

    class FakeTrainer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.worker = ManifestWorker()

    monkeypatch.setattr(experiment, "MlxLoraTrainer", FakeTrainer)
    monkeypatch.setattr(
        experiment,
        "REAL_CYCLE_POLICY",
        replace_dataclass(experiment.REAL_CYCLE_POLICY, min_replay_examples=1),
    )
    feasible = _exp003_spec()
    feasible["required_setup"]["machine"]["min_replay_examples"] = 1
    path = tmp_path / "exp003-feasible.json"
    path.write_text(json.dumps(sealed(feasible)))
    database = tmp_path / "cycle.db"

    with pytest.raises(no_sandbox):
        experiment.run_first_real_cycle(
            database,
            tmp_path / "cycle.json",
            reset=True,
            correction_source="self-repair",
            compare_corrections=True,
            spec_path=path,
            arm="primary",
        )

    assert not database.exists()


# --------------------------------------------------------------------------
# A refused setup is a refusal, not a crash
# --------------------------------------------------------------------------


def test_cli_reports_a_setup_refusal_without_a_traceback(
    tmp_path, capsys, thin_catalog
):
    """The refusal was correct; the traceback and exit 1 were not.

    Exit 1 is the code for a falsified prediction. Nothing ran here, so this is
    exit 2 -- the same code the checker uses for "cannot be judged".
    """
    from grove.cli import main

    database = tmp_path / "cycle.db"
    report_path = tmp_path / "cycle.json"

    code = main(
        [
            "--db",
            str(database),
            "real-cycle",
            "--reset",
            "--spec",
            str(_SPEC_DIR / "EXP-002-forced-replay-and-route-precision.json"),
            "--arm",
            "primary",
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    payload = json.loads(captured.err)
    assert payload["status"] == "setup_refused"
    assert "replay cohort is impossible" in payload["error"]
    assert not database.exists()
    assert not report_path.exists()


def test_cli_setup_refusal_is_a_single_deterministic_document(
    tmp_path, capsys, thin_catalog
):
    from grove.cli import main

    arguments = [
        "--db",
        str(tmp_path / "cycle.db"),
        "real-cycle",
        "--reset",
        "--spec",
        str(_exp003()),
        "--arm",
        "control",
        "--correction-source",
        "canonical",
        "--compare-corrections",
        "--report",
        str(tmp_path / "cycle.json"),
    ]

    assert main(arguments) == 2
    first = capsys.readouterr().err
    assert main(arguments) == 2
    second = capsys.readouterr().err

    assert first == second
    assert json.loads(first)["status"] == "setup_refused"


def test_a_tampered_report_falsifies_no_hypothesis():
    """An edit is not a result. Rule outcomes over edited values grade nothing."""
    spec, primary, control = _exp003_pair()
    # D5 demands at least one admitted expert; emptying the list would fail it.
    primary["cycle"]["experts_admitted"] = []

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert verdict["rules_unevaluable"] == verdict["rules_total"]


# --------------------------------------------------------------------------
# Finding 1: control requirements come from the rules, not an optional flag
# --------------------------------------------------------------------------


def _flagless_paired_spec(**overrides) -> dict:
    """PAIRED_SPEC with the optional flag removed, plus overrides."""
    spec = copy.deepcopy(PAIRED_SPEC)
    spec.pop("requires_control_report", None)
    spec.update(overrides)
    return sealed(spec)


def test_delta_rules_require_control_even_when_flag_is_false():
    """Finding 1: the flag was the only switch, so removing it disabled pairing.

    A control with a different base model and a different source revision
    returned exit 0, ``arm_pairing: null`` and "all predeclared rules
    satisfied". The delta rules ran against an arm nobody had identity-checked.
    """
    spec = _flagless_paired_spec(requires_control_report=False)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    control["provenance"]["base_model"] = "some-other-model@zzz"
    control["provenance"]["source"]["revision"] = "f" * 40

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["control_required"] is True
    assert verdict["arm_pairing"] is not None
    assert verdict["arm_pairing"]["paired"] is False
    assert verdict["falsified_hypotheses"] == []
    mismatched = {item["path"] for item in verdict["arm_pairing"]["mismatches"]}
    assert "provenance.base_model" in mismatched
    assert "provenance.source.revision" in mismatched


def test_a_flag_that_contradicts_the_rules_is_itself_a_blocker():
    spec = _flagless_paired_spec(requires_control_report=False)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["control_requirement"]["flag_contradicts_rules"] is True
    assert any("requires_control_report false" in item for item in verdict["blockers"])


def test_delta_rules_without_control_are_unevaluable_not_falsified():
    """Finding 1: exit 1 with H2 falsified, though no comparison happened."""
    spec = _flagless_paired_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)

    verdict = checker.check(spec, primary)

    assert verdict["unusable"] is True
    assert verdict["falsified_hypotheses"] == []
    assert verdict["rules_failed"] == 0
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert any("none was supplied" in item for item in verdict["blockers"])


def test_control_arm_rule_requires_control_report():
    """A rule reading ``arm: control`` needs a control even with no deltas."""
    only_control = {
        "spec_id": "EXP-CONTROL-ONLY",
        "hypotheses": [
            {"id": "H1", "claim": "the control ran", "falsified_if": "it did not"}
        ],
        "decision_rules": [
            {
                "id": "D1",
                "hypothesis": "H1",
                "arm": "control",
                "path": "correction_source",
                "comparison": "==",
                "value": "canonical",
            }
        ],
    }
    spec = sealed(only_control)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)

    need = checker.control_requirement(spec)
    verdict = checker.check(spec, document)

    assert need["required"] is True
    assert need["rules"] == [{"rule": "D1", "because": ["reads the control arm"]}]
    assert verdict["unusable"] is True
    assert verdict["falsified_hypotheses"] == []


def test_rule_derived_control_pairing_checks_arm_identity():
    """Every identity path runs, flag or no flag."""
    spec = _flagless_paired_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    control["provenance"]["worker"]["checkout"]["worktree_sha256"] = "d" * 64

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(
        item["path"] == "provenance.worker.checkout.worktree_sha256"
        for item in verdict["arm_pairing"]["mismatches"]
    )


def test_a_spec_with_no_paired_rules_needs_no_control():
    """The derivation is a derivation, not a blanket demand."""
    spec = sealed(BASE_SPEC)

    need = checker.control_requirement(spec)
    verdict = checker.check(
        spec, report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    )

    assert need["required"] is False
    assert verdict["control_required"] is False
    assert verdict["unusable"] is False


def test_checker_and_runner_derive_the_same_control_requirement():
    from grove.experiment import control_requirement

    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        assert control_requirement(spec) == checker.control_requirement(spec), path.name


# --------------------------------------------------------------------------
# Finding 2: a paired spec must declare its control arm's setup
# --------------------------------------------------------------------------


def test_missing_control_setup_is_not_conformant():
    """Finding 2: deleting control_required_setup made the control unchecked.

    A control arm running with ``compare_corrections: false`` passed, because
    an absent declaration collapsed to "spec declares no required_setup".
    """
    spec = _flagless_paired_spec()
    del spec["control_required_setup"]
    spec = sealed(spec)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    control["run_setup"]["compare_corrections"] = False

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["control_setup_conformance"]["conformant"] is False
    assert "declares no machine-checkable required_setup" in (
        verdict["control_setup_conformance"]["reason"]
    )


def test_empty_control_setup_is_not_conformant():
    spec = _flagless_paired_spec(control_required_setup={})
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["control_setup_conformance"]["conformant"] is False


def test_control_prose_setup_is_not_machine_conformance():
    """Prose is listed, and listing it is not the same as checking it."""
    spec = _flagless_paired_spec(
        control_required_setup={"prose": {"authoring_rule": "written blind"}}
    )
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["control_setup_conformance"]["conformant"] is False
    assert verdict["control_setup_conformance"]["unchecked_prose_keys"] == [
        "authoring_rule"
    ]


def test_control_setup_missing_refuses_before_cost(tmp_path, no_sandbox):
    from grove import experiment

    hollow = _exp003_spec()
    del hollow["control_required_setup"]
    path = tmp_path / "no-control-setup.json"
    path.write_text(json.dumps(sealed(hollow)))
    database = tmp_path / "cycle.db"

    with pytest.raises(experiment.ExperimentSetupError) as error:
        experiment.run_first_real_cycle(
            database,
            tmp_path / "cycle.json",
            reset=True,
            correction_source="self-repair",
            compare_corrections=True,
            spec_path=path,
            arm="primary",
        )

    assert "no machine-checkable control_required_setup" in str(error.value)
    assert not database.exists()


def test_a_contradicting_control_flag_refuses_before_cost(tmp_path, no_sandbox):
    from grove import experiment

    contradicting = _exp003_spec()
    contradicting["requires_control_report"] = False
    path = tmp_path / "contradicting.json"
    path.write_text(json.dumps(sealed(contradicting)))
    database = tmp_path / "cycle.db"

    with pytest.raises(experiment.ExperimentSetupError, match="requires_control_report"):
        experiment.run_first_real_cycle(
            database,
            tmp_path / "cycle.json",
            reset=True,
            correction_source="self-repair",
            compare_corrections=True,
            spec_path=path,
            arm="primary",
        )

    assert not database.exists()


# --------------------------------------------------------------------------
# Finding 3: an absent provenance digest is not neutral
# --------------------------------------------------------------------------


def _unhashed_arm(spec: dict, source: str, arm_name: str) -> dict:
    """An arm whose provenance carries no digest, sealed anyway."""
    from grove.experiment import build_run_manifest, decision_rule_input_paths
    from grove.provenance import canonical_hash, canonical_json

    document = _exp003_arm(spec, source=source, arm_name=arm_name, heldout=0.9)
    del document["provenance"]["provenance_sha256"]
    document.pop("run_manifest", None)
    document.pop("run_manifest_sha256", None)
    serialized = json.loads(canonical_json(document))
    manifest = build_run_manifest(
        serialized, rule_paths=decision_rule_input_paths(spec)
    )
    document["run_manifest"] = manifest
    document["run_manifest_sha256"] = canonical_hash(manifest)
    return document


def test_missing_provenance_digest_blocks_strict_report():
    """Finding 3: absence recorded ``intact: null`` and raised no problem."""
    spec = _exp003_spec()
    primary = _unhashed_arm(spec, "self-repair", "primary")
    control = _unhashed_arm(spec, "canonical", "control")

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "unverified"
    assert verdict["report_integrity"]["authoritative"] is False
    assert any("report integrity" in item for item in verdict["blockers"])


def test_missing_provenance_digest_marks_rules_unevaluable():
    spec = _exp003_spec()
    primary = _unhashed_arm(spec, "self-repair", "primary")
    control = _unhashed_arm(spec, "canonical", "control")

    verdict = checker.check(spec, primary, control)

    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []


def test_editing_both_arm_sources_without_provenance_digest_is_unusable():
    """The reviewer's exact path: no digest, then both source revisions edited."""
    spec = _exp003_spec()
    primary = _unhashed_arm(spec, "self-repair", "primary")
    control = _unhashed_arm(spec, "canonical", "control")
    for document in (primary, control):
        document["provenance"]["source"]["revision"] = "f" * 40

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "unverified"


def test_non_strict_missing_digest_is_unverified_not_intact():
    """A legacy report says so plainly instead of passing as intact."""
    from grove.experiment import build_run_manifest, decision_rule_input_paths
    from grove.provenance import canonical_hash, canonical_json

    spec = sealed(BASE_SPEC)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    document["provenance"] = {"source": {"revision": "a" * 40}}
    # Sealed the way a legacy report was: a self-consistent manifest, but with
    # nothing behind provenance_sha256 for it to commit to.
    serialized = json.loads(canonical_json(document))
    manifest = build_run_manifest(
        serialized, rule_paths=decision_rule_input_paths(spec)
    )
    document["run_manifest"] = manifest
    document["run_manifest_sha256"] = canonical_hash(manifest)

    integrity = checker.report_integrity(spec, document, strict=False)

    assert integrity["status"] == "unverified"
    assert integrity["authoritative"] is False
    # Non-strict, so it does not by itself make the run unusable.
    assert integrity["bound"] is True


def test_sealing_a_strict_report_without_a_provenance_digest_is_refused():
    """The checker is the last defence, not the only one."""
    from grove import experiment

    spec = _exp003_spec()
    document = _exp003_arm(spec, source="self-repair", arm_name="primary", heldout=0.9)
    del document["provenance"]["provenance_sha256"]

    with pytest.raises(experiment.ExperimentSetupError, match="provenance_sha256"):
        experiment.seal_report(document, spec=spec)


# --------------------------------------------------------------------------
# Finding 7: the stored adapter digest is checked too
# --------------------------------------------------------------------------


def test_editing_adapter_digest_is_tampered():
    """Finding 7: forging it returned exit 0 and integrity ``intact``."""
    spec = _exp003_spec()
    primary = _exp003_arm(spec, source="self-repair", arm_name="primary", heldout=0.9)
    primary["experts"][0]["artifact"] = {"adapter_sha256": "a" * 64}
    from grove.experiment import seal_report

    seal_report(primary, spec=spec)
    control = _exp003_arm(spec, source="canonical", arm_name="control", heldout=0.9)

    intact = checker.check(spec, primary, control)
    assert intact["report_integrity"]["status"] == "intact"

    primary["experts"][0]["artifact"]["adapter_sha256"] = "FORGED"
    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "tampered"
    assert any(
        "adapter digest does not match" in problem
        for problem in verdict["report_integrity"]["problems"]
    )


def test_missing_adapter_binding_is_not_reported_intact():
    """Deleting the artifact after sealing is a mismatch, not an absence."""
    spec = _exp003_spec()
    primary = _exp003_arm(spec, source="self-repair", arm_name="primary", heldout=0.9)
    primary["experts"][0]["artifact"] = {"adapter_sha256": "a" * 64}
    from grove.experiment import seal_report

    seal_report(primary, spec=spec)
    control = _exp003_arm(spec, source="canonical", arm_name="control", heldout=0.9)
    del primary["experts"][0]["artifact"]

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["report_integrity"]["status"] == "tampered"


# --------------------------------------------------------------------------
# Finding 6: the sealed EXP-003 text must match the sealed EXP-003 fields
# --------------------------------------------------------------------------


def test_exp003_permitted_gap_matches_required_identity():
    spec = _exp003_spec()

    permitted = set(spec["permitted_provenance_gaps"])
    required = {
        checker._gap_name(path) for path in spec["required_resolved_identity"]
    }

    assert permitted == {"models.base.aggregate_sha256"}
    assert required == {"worker.model_manifest_sha256"}
    # A path cannot be both waivable and required.
    assert not permitted & required


def test_exp003_limitation_does_not_permit_worker_manifest_gap():
    """Finding 6: the sealed prose said D7 permitted a gap the fields forbid."""
    spec = _exp003_spec()
    gap_rule = next(
        rule for rule in spec["decision_rules"] if rule["path"] == "provenance_gaps"
    )
    limitations = " ".join(spec["preregistered_limitations"])

    assert gap_rule["value"] == ["models.base.aggregate_sha256"]
    assert "D7 permits exactly one gap" in limitations
    assert "worker.model_manifest_sha256 is NOT permitted" in limitations


def test_all_shipped_specs_recompute_their_seals():
    from grove.experiment import load_sealed_spec

    for path in sorted(_SPEC_DIR.glob("*.json")):
        spec = json.loads(path.read_text())
        assert spec[checker.SPEC_HASH_FIELD] == checker.spec_digest(spec), path.name
        assert load_sealed_spec(path)[checker.SPEC_HASH_FIELD] == (
            spec[checker.SPEC_HASH_FIELD]
        )


def test_resealing_exp003_invalidates_old_report_binding(tmp_path):
    """Resealing has a cost, and the cost is the point."""
    spec = _exp003_spec()
    bound = _exp003_arm(spec, source="self-repair", arm_name="primary", heldout=0.9)
    assert checker.spec_binding(spec, bound)["bound"] is True

    edited = copy.deepcopy(spec)
    edited["preregistered_limitations"] = [
        *edited["preregistered_limitations"],
        "An additional preregistered limitation.",
    ]
    path = tmp_path / "resealed.json"
    path.write_text(json.dumps(edited, indent=2, sort_keys=True) + "\n")
    assert checker.main(["--spec", str(path), "--seal", "--reseal"]) == 0

    resealed = json.loads(path.read_text())
    assert resealed[checker.SPEC_HASH_FIELD] != spec[checker.SPEC_HASH_FIELD]
    assert resealed[checker.SPEC_HASH_FIELD] == checker.spec_digest(resealed)
    # The report bound to the old digest is refused under the new one.
    binding = checker.spec_binding(resealed, bound)
    assert binding["bound"] is False
    assert "different version" in binding["reason"]


# --------------------------------------------------------------------------
# Current review blockers: blank identity, unbound evidence, rollback binding,
# malformed rules and every declared arm profile.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_identity_values_are_not_a_pairing_match(blank):
    spec = sealed(PAIRED_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    primary["provenance"]["base_model"] = blank
    control["provenance"]["base_model"] = blank

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert any(
        item["path"] == "provenance.base_model"
        for item in verdict["arm_pairing"]["mismatches"]
    )


@pytest.mark.parametrize("blank", ["", "   "])
def test_runner_treats_blank_required_identity_as_unresolved(monkeypatch, blank):
    from grove import experiment

    monkeypatch.setattr(
        experiment,
        "git_revision",
        lambda _root: {"revision": blank},
    )
    result = experiment.preflight_required_identity(
        {"required_resolved_identity": ["provenance.source.revision"]}
    )

    assert result["unresolved"] == [
        {"path": "provenance.source.revision", "value": blank}
    ]


def test_unbound_rule_report_is_unusable_even_without_strict_integrity_flag():
    spec = sealed(BASE_SPEC)
    document = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    document.pop("run_manifest", None)
    document.pop("run_manifest_sha256", None)

    verdict = checker.check(spec, document)

    assert verdict["report_integrity"]["status"] == "unbound"
    assert verdict["unusable"] is True
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert verdict["falsified_hypotheses"] == []


def test_rollback_metrics_and_exact_evaluation_selector_are_manifest_bound():
    from grove.experiment import seal_report

    spec = _integrity_spec()
    primary = _sealed_arm(spec, "self-repair", heldout=0.9)
    primary["rollback"] = {"active_experts": 0, "capability": 0.25}
    primary["rollback_audit"] = {
        "evaluation_id": "eval_rollback_exact",
        "run_id": "rollback_run_exact",
    }
    primary["evaluation_ids"] = ["eval_baseline", "eval_rollback_exact"]
    seal_report(primary, spec=spec)
    control = _sealed_arm(spec, "canonical", heldout=0.9)

    assert primary["run_manifest"]["rollback_sha256"]
    assert primary["run_manifest"]["rollback_audit_sha256"]
    assert primary["run_manifest"]["rollback_evaluation_selector"] == {
        "evaluation_id": "eval_rollback_exact",
        "run_id": "rollback_run_exact",
    }
    assert checker.check(spec, primary, control)["unusable"] is False

    primary["rollback"]["active_experts"] = 99
    tampered = checker.check(spec, primary, control)
    assert tampered["unusable"] is True
    assert tampered["report_integrity"]["status"] == "tampered"
    assert any("rollback_sha256" in item for item in tampered["report_integrity"]["problems"])

    primary["rollback"]["active_experts"] = 0
    primary["rollback_audit"]["evaluation_id"] = "eval_other"
    tampered_selector = checker.check(spec, primary, control)
    assert tampered_selector["unusable"] is True
    assert any(
        "rollback_audit_sha256" in item
        or "rollback_evaluation_selector" in item
        for item in tampered_selector["report_integrity"]["problems"]
    )


@pytest.mark.parametrize("declaration", [False, [], "", 0])
def test_falsey_malformed_required_setup_is_not_collapsed(tmp_path, declaration):
    from grove import experiment

    spec = copy.deepcopy(BASE_SPEC)
    spec["required_setup"] = declaration

    with pytest.raises(
        experiment.ExperimentSetupError, match="primary arm setup schema is invalid"
    ):
        experiment.preflight_experiment(
            sealed(spec),
            correction_source="canonical",
            self_repair_attempts=3,
            compare_corrections=False,
        )


@pytest.mark.parametrize(
    "rule",
    [
        None,
        {"id": "D-BLANK", "path": "   ", "comparison": "==", "value": 1},
        {"id": "D-OP", "path": "value", "comparison": "???", "value": 1},
    ],
)
def test_malformed_decision_rule_returns_exit_two_not_traceback(tmp_path, rule):
    spec = copy.deepcopy(BASE_SPEC)
    spec["decision_rules"] = [rule]
    spec = sealed(spec)
    spec_path = tmp_path / "malformed.json"
    report_path = tmp_path / "report.json"
    spec_path.write_text(json.dumps(spec))
    report_path.write_text(
        json.dumps(report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec))
    )

    assert checker.main(["--spec", str(spec_path), "--report", str(report_path)]) == 2
    verdict = checker.check(spec, json.loads(report_path.read_text()))
    assert verdict["decision_rule_validation"]["valid"] is False
    assert verdict["unusable"] is True


def test_diagnostic_gold_tag_route_metric_cannot_gate_a_sealed_rule():
    spec = copy.deepcopy(BASE_SPEC)
    spec["decision_rules"][0]["path"] = (
        "experts[*].metrics.route_recall_gold_tags"
    )
    spec = sealed(spec)

    verdict = checker.check(
        spec, report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    )

    assert verdict["decision_rule_validation"]["valid"] is False
    assert verdict["unusable"] is True
    assert any("diagnostic metric" in error for error in verdict["blockers"])


def test_runner_rejects_diagnostic_gold_tag_rule_before_sandbox(tmp_path, no_sandbox):
    from grove import experiment

    spec = copy.deepcopy(BASE_SPEC)
    spec["decision_rules"][0]["path"] = (
        "experts[*].metrics.route_recall_gold_tags"
    )
    path = tmp_path / "diagnostic-rule.json"
    path.write_text(json.dumps(sealed(spec)))

    with pytest.raises(experiment.ExperimentSetupError, match="diagnostic metric"):
        experiment.run_first_real_cycle(
            tmp_path / "cycle.db",
            tmp_path / "cycle.json",
            reset=True,
            spec_path=path,
            correction_source="canonical",
        )


def test_runner_rejects_malformed_decision_rule_before_sandbox(
    tmp_path, no_sandbox
):
    from grove import experiment

    spec = copy.deepcopy(BASE_SPEC)
    spec["decision_rules"][0]["comparison"] = "unsupported"
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(sealed(spec)))

    with pytest.raises(experiment.ExperimentSetupError, match="decision rule schema"):
        experiment.run_first_real_cycle(
            tmp_path / "cycle.db",
            tmp_path / "cycle.json",
            reset=True,
            spec_path=path,
            correction_source="canonical",
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("correction_source", "telepathy", "unknown correction source"),
        ("self_repair_attempts", 0, "at least 1"),
        ("self_repair_attempts", "three", "must be an integer"),
    ],
)
def test_every_declared_arm_profile_validates_correction_settings(
    key, value, message
):
    from grove import experiment

    spec = _exp003_spec()
    spec["control_required_setup"]["machine"][key] = value

    with pytest.raises(experiment.ExperimentSetupError, match=message):
        experiment.preflight_experiment(
            sealed(spec),
            correction_source="self-repair",
            self_repair_attempts=3,
            compare_corrections=True,
            arm="primary",
        )


def test_exp003_binds_replay_capacity_in_both_arm_profiles():
    spec = _exp003_spec()

    assert spec["required_setup"]["machine"]["min_replay_examples"] == 50
    assert spec["control_required_setup"]["machine"]["min_replay_examples"] == 50
    assert spec[checker.SPEC_HASH_FIELD] == checker.spec_digest(spec)


# --------------------------------------------------------------------------
# Reviewer B round 5 blockers:
#   1. a non-digest is not worker model identity;
#   3. an unknown decision-rule arm is a schema error, not a primary fallback;
#   4. an unusable run publishes no rule or hypothesis outcome;
#   5. absent, unmeasured and non-numeric delta inputs are unevaluable.
# --------------------------------------------------------------------------

NON_DIGEST_IDENTITIES = [
    False,
    0,
    {},
    [],
    "",
    "   ",
    "not-a-digest",
    "A" * 64,
    "a" * 63,
    "a" * 65,
    "sha256:" + "a" * 64,
]


def _strict_identity_spec() -> dict:
    strict = copy.deepcopy(PAIRED_SPEC)
    strict["permitted_provenance_gaps"] = ["models.base.aggregate_sha256"]
    strict["required_resolved_identity"] = [
        "provenance.worker.model_manifest_sha256"
    ]
    return sealed(strict)


@pytest.mark.parametrize("value", NON_DIGEST_IDENTITIES)
def test_equal_malformed_worker_digests_are_not_a_pairing_match(value):
    """Two arms answering the same non-digest have not shown they match."""
    spec = _strict_identity_spec()
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    for document in (primary, control):
        document["provenance"]["worker"]["model_manifest_sha256"] = value

    verdict = checker.check(spec, primary, control)

    assert verdict["arm_pairing"]["paired"] is False
    mismatch = next(
        item
        for item in verdict["arm_pairing"]["mismatches"]
        if item["path"] == "provenance.worker.model_manifest_sha256"
    )
    assert "required for paired identity" in mismatch["reason"]
    # A required path is unresolved, so no other gap may be waived either.
    assert verdict["arm_pairing"]["permitted_gaps_used"] == []
    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []


@pytest.mark.parametrize("value", NON_DIGEST_IDENTITIES)
def test_runner_preflight_refuses_a_malformed_worker_model_digest(
    monkeypatch, value
):
    """Caught before two MLX training runs are spent, not after."""
    from grove import experiment

    monkeypatch.setattr(
        experiment,
        "_worker_provenance",
        lambda _worker: {"model_manifest_sha256": value},
    )
    monkeypatch.setattr(
        experiment, "MlxLoraTrainer", lambda **_kwargs: types.SimpleNamespace(worker=None)
    )

    result = experiment.preflight_required_identity(
        {"required_resolved_identity": ["provenance.worker.model_manifest_sha256"]}
    )

    assert [item["path"] for item in result["unresolved"]] == [
        "provenance.worker.model_manifest_sha256"
    ]


def test_runner_preflight_accepts_a_real_worker_model_digest(monkeypatch):
    from grove import experiment

    monkeypatch.setattr(
        experiment,
        "_worker_provenance",
        lambda _worker: {"model_manifest_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        experiment, "MlxLoraTrainer", lambda **_kwargs: types.SimpleNamespace(worker=None)
    )

    result = experiment.preflight_required_identity(
        {"required_resolved_identity": ["provenance.worker.model_manifest_sha256"]}
    )

    assert result["unresolved"] == []


UNKNOWN_RULE_ARMS = ["control ", " control", "primary ", "unknown", "", False, 0, None]


@pytest.mark.parametrize("bad_arm", UNKNOWN_RULE_ARMS)
def test_an_unknown_rule_arm_is_a_schema_error_not_a_primary_fallback(bad_arm):
    """``"control "`` used to grade the primary arm silently."""
    typo = copy.deepcopy(PAIRED_SPEC)
    typo["decision_rules"][0]["arm"] = bad_arm
    spec = sealed(typo)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["decision_rule_validation"]["valid"] is False
    assert any(
        "decision_rules[0].arm" in error
        for error in verdict["decision_rule_validation"]["errors"]
    )
    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert verdict["falsified_hypotheses"] == []
    assert all(item["unevaluable"] is True for item in verdict["rules"])


@pytest.mark.parametrize("bad_arm", UNKNOWN_RULE_ARMS)
def test_the_runner_rejects_an_unknown_rule_arm_too(bad_arm):
    from grove.experiment import decision_rule_problems

    typo = copy.deepcopy(PAIRED_SPEC)
    typo["decision_rules"][0]["arm"] = bad_arm

    problems = decision_rule_problems(typo)

    assert any("decision_rules[0].arm" in problem for problem in problems)


def test_an_unknown_rule_arm_exits_two(tmp_path):
    typo = copy.deepcopy(PAIRED_SPEC)
    typo["decision_rules"][0]["arm"] = "control "
    spec = sealed(typo)
    spec_path = tmp_path / "spec.json"
    primary = tmp_path / "self.json"
    control = tmp_path / "canonical.json"
    spec_path.write_text(json.dumps(spec))
    primary.write_text(json.dumps(arm(spec=spec, source="self-repair", heldout=0.9)))
    control.write_text(json.dumps(arm(spec=spec, source="canonical", heldout=0.9)))

    assert (
        checker.main(
            [
                "--spec",
                str(spec_path),
                "--report",
                str(primary),
                "--control-report",
                str(control),
            ]
        )
        == 2
    )


def test_omitted_and_exact_rule_arms_remain_valid():
    """Omission still means primary, and exact control still reads the control."""
    assert checker.decision_rule_problems(PAIRED_SPEC) == []
    assert "arm" not in PAIRED_SPEC["decision_rules"][0]
    assert PAIRED_SPEC["decision_rules"][1]["arm"] == "control"

    spec = sealed(PAIRED_SPEC)
    verdict = checker.check(
        spec,
        arm(spec=spec, source="self-repair", heldout=0.9),
        arm(spec=spec, source="canonical", heldout=0.9),
    )

    assert verdict["unusable"] is False


def _paired_documents(spec):
    return (
        arm(spec=spec, source="self-repair", heldout=0.9),
        arm(spec=spec, source="canonical", heldout=0.9),
    )


def _damaged(damage):
    """A sealed spec and its arms, with exactly one blocker introduced."""
    spec = sealed(PAIRED_SPEC)
    return damage(spec, *_paired_documents(spec))


def _break_setup(spec, primary, control):
    primary["run_setup"]["correction_source"] = "canonical"
    return spec, primary, control


def _break_spec_binding(spec, primary, control):
    primary["experiment_spec"]["spec_sha256"] = "0" * 64
    return spec, primary, control


def _break_timing(spec, primary, control):
    # A preregistration claim with no verifiable attestation, sealed honestly
    # so timing is the first blocker rather than an altered seal.
    claiming = copy.deepcopy(PAIRED_SPEC)
    claiming["declared_before_run"] = True
    resealed = sealed(claiming)
    return (resealed, *_paired_documents(resealed))


def _break_integrity(spec, primary, control):
    primary["experts"][0]["metrics"]["heldout_target_rate"] = 0.2
    return spec, primary, control


def _break_arm_identity(spec, primary, control):
    control["provenance"]["source"]["revision"] = "f" * 40
    return spec, primary, control


def _break_control_presence(spec, primary, control):
    return spec, primary, None


def _break_spec_seal(spec, primary, control):
    spec["decision_rules"][0]["value"] = -0.9
    return spec, primary, control


@pytest.mark.parametrize(
    "damage",
    [
        _break_setup,
        _break_spec_binding,
        _break_timing,
        _break_integrity,
        _break_arm_identity,
        _break_control_presence,
        _break_spec_seal,
    ],
    ids=[
        "setup",
        "spec_binding",
        "timing",
        "integrity",
        "arm_identity",
        "missing_control",
        "altered_seal",
    ],
)
def test_an_unusable_run_publishes_no_scientific_outcome(damage):
    """Exit 2 means "not judged". It must never carry a falsified hypothesis."""
    spec, primary, control = _damaged(damage)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["blockers"]
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert all(item["unevaluable"] is True for item in verdict["rules"])
    assert verdict["verdict"] == verdict["blockers"][0]


@pytest.mark.parametrize(
    "damage",
    [
        _break_setup,
        _break_spec_binding,
        _break_integrity,
        _break_arm_identity,
        _break_spec_seal,
    ],
    ids=["setup", "spec_binding", "integrity", "arm_identity", "altered_seal"],
)
def test_an_unusable_run_exits_two_from_the_cli(tmp_path, damage):
    spec, primary, control = _damaged(damage)
    spec_path = tmp_path / "spec.json"
    primary_path = tmp_path / "self.json"
    control_path = tmp_path / "canonical.json"
    spec_path.write_text(json.dumps(spec))
    primary_path.write_text(json.dumps(primary))
    control_path.write_text(json.dumps(control))

    assert (
        checker.main(
            [
                "--spec",
                str(spec_path),
                "--report",
                str(primary_path),
                "--control-report",
                str(control_path),
            ]
        )
        == 2
    )


def _delete_metric(document):
    del document["experts"][0]["metrics"]["heldout_target_rate"]


def _null_metric(document):
    document["experts"][0]["metrics"]["heldout_target_rate"] = None


def _string_metric(document):
    document["experts"][0]["metrics"]["heldout_target_rate"] = "0.9"


def _mapping_metric(document):
    document["experts"][0]["metrics"]["heldout_target_rate"] = {"value": 0.9}


def _bool_metric(document):
    document["experts"][0]["metrics"]["heldout_target_rate"] = True


@pytest.mark.parametrize(
    ("mutate", "state"),
    [
        (_delete_metric, "absent"),
        (_null_metric, "null/unmeasured"),
        (_string_metric, "non-numeric (str)"),
        (_mapping_metric, "non-numeric (dict)"),
        (_bool_metric, "non-numeric (bool)"),
    ],
    ids=["absent", "null", "string", "mapping", "bool"],
)
@pytest.mark.parametrize("target", ["primary", "control"])
def test_an_unmeasured_delta_input_is_unevaluable_not_falsified(
    mutate, state, target
):
    """A metric nobody measured is not a refuted prediction."""
    spec = sealed(PAIRED_SPEC)
    primary, control = _paired_documents(spec)
    document = primary if target == "primary" else control
    mutate(document)
    # Resealed: the run genuinely reported this value, so the refusal under
    # test is the delta input itself rather than an edited report.
    bind(document, spec)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert verdict["falsified_hypotheses"] == []
    # No rule keeps a scientific outcome in either direction: a rule graded
    # before the refusal must not stay `passed: true` in an unusable payload.
    assert all(item["passed"] is False for item in verdict["rules"])
    assert all(item["unevaluable"] is True for item in verdict["rules"])
    detail = verdict["rules"][0]["detail"]
    for fragment in (
        "D1",
        target,
        "experts[*].metrics.heldout_target_rate",
        state,
    ):
        assert fragment in detail or fragment in " ".join(verdict["blockers"])
    assert any(
        "D1" in blocker
        and target in blocker
        and "experts[*].metrics.heldout_target_rate" in blocker
        and state in blocker
        for blocker in verdict["blockers"]
    )


@pytest.mark.parametrize(
    "mutate",
    [_delete_metric, _null_metric, _string_metric, _mapping_metric, _bool_metric],
    ids=["absent", "null", "string", "mapping", "bool"],
)
def test_an_unmeasured_delta_input_exits_two(tmp_path, mutate):
    spec = sealed(PAIRED_SPEC)
    primary, control = _paired_documents(spec)
    mutate(control)
    bind(control, spec)
    spec_path = tmp_path / "spec.json"
    primary_path = tmp_path / "self.json"
    control_path = tmp_path / "canonical.json"
    spec_path.write_text(json.dumps(spec))
    primary_path.write_text(json.dumps(primary))
    control_path.write_text(json.dumps(control))

    assert (
        checker.main(
            [
                "--spec",
                str(spec_path),
                "--report",
                str(primary_path),
                "--control-report",
                str(control_path),
            ]
        )
        == 2
    )


BAD_DELTA_THRESHOLDS = [
    None,
    "not-a-number",
    {},
    [],
    False,
]


def _spec_with_bad_delta_threshold(value=checker.MISSING):
    spec = copy.deepcopy(PAIRED_SPEC)
    if value is checker.MISSING:
        del spec["decision_rules"][0]["value"]
    else:
        spec["decision_rules"][0]["value"] = value
    return sealed(spec)


@pytest.mark.parametrize(
    "value",
    [checker.MISSING, *BAD_DELTA_THRESHOLDS],
    ids=["missing", "null", "string", "mapping", "list", "bool"],
)
def test_bad_delta_threshold_is_unusable_not_satisfied(value):
    """A bad tolerance means the rule was not graded; it is not success."""
    spec = _spec_with_bad_delta_threshold(value)
    primary, control = _paired_documents(spec)

    verdict = checker.check(spec, primary, control)

    assert verdict["decision_rule_validation"]["valid"] is False
    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert verdict["falsified_hypotheses"] == []
    assert all(item["passed"] is False for item in verdict["rules"])
    assert all(item["unevaluable"] is True for item in verdict["rules"])
    assert verdict["verdict"] == verdict["blockers"][0]


@pytest.mark.parametrize(
    "value",
    [checker.MISSING, *BAD_DELTA_THRESHOLDS],
    ids=["missing", "null", "string", "mapping", "list", "bool"],
)
def test_bad_delta_threshold_exits_two(tmp_path, value):
    spec = _spec_with_bad_delta_threshold(value)
    primary, control = _paired_documents(spec)
    spec_path = tmp_path / "spec.json"
    primary_path = tmp_path / "self.json"
    control_path = tmp_path / "canonical.json"
    spec_path.write_text(json.dumps(spec))
    primary_path.write_text(json.dumps(primary))
    control_path.write_text(json.dumps(control))

    assert (
        checker.main(
            [
                "--spec",
                str(spec_path),
                "--report",
                str(primary_path),
                "--control-report",
                str(control_path),
            ]
        )
        == 2
    )
# --------------------------------------------------------------------------
# Review round 7: a late pairing refusal must withdraw a rule that had already
# been graded. Marking it `unevaluable` while leaving `passed: true` publishes
# a satisfied prediction inside a payload that says nothing could be judged.
# --------------------------------------------------------------------------

PASS_FIRST_SPEC = {
    "spec_id": "EXP-PASS-FIRST",
    "requires_control_report": True,
    "hypotheses": [
        {
            "id": "H2",
            "claim": "self-repair matches the human reference",
            "falsified_if": "held-out rate is more than 0.1 below the control arm",
        }
    ],
    "required_setup": {"correction_source": "self-repair"},
    "control_required_setup": {"correction_source": "canonical"},
    "decision_rules": [
        # Graded first, and satisfied, before the delta rule refuses.
        {
            "id": "D0",
            "hypothesis": "H2",
            "path": "correction_source",
            "comparison": "==",
            "value": "self-repair",
        },
        {
            "id": "D1",
            "hypothesis": "H2",
            "path": "experts[*].metrics.heldout_target_rate",
            "control_path": "experts[*].metrics.heldout_target_rate",
            "pair_on": "experts[*].pairing_key",
            "comparison": "delta>=",
            "value": -0.1,
        },
        {
            "id": "D2",
            "arm": "control",
            "path": "correction_source",
            "comparison": "==",
            "value": "canonical",
        },
    ],
}


def test_a_rule_graded_before_a_late_refusal_still_passes_when_the_run_is_usable():
    """The control: D0 really is satisfied, so the withdrawal below is real."""
    spec = sealed(PASS_FIRST_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is False
    assert verdict["rules"][0]["id"] == "D0"
    assert verdict["rules"][0]["passed"] is True
    assert verdict["rules"][0]["unevaluable"] is False


@pytest.mark.parametrize(
    "mutate",
    [_delete_metric, _null_metric, _string_metric, _mapping_metric, _bool_metric],
    ids=["absent", "null", "string", "mapping", "bool"],
)
@pytest.mark.parametrize("target", ["primary", "control"])
def test_a_late_pairing_refusal_withdraws_an_already_graded_pass(mutate, target):
    spec = sealed(PASS_FIRST_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    document = primary if target == "primary" else control
    mutate(document)
    bind(document, spec)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert verdict["rules_unevaluable"] == verdict["rules_total"] == 3
    for item in verdict["rules"]:
        assert item["passed"] is False, item["id"]
        assert item["unevaluable"] is True, item["id"]


@pytest.mark.parametrize("target", ["primary", "control"])
def test_a_late_unmatched_pairing_key_also_withdraws_a_graded_pass(target):
    spec = sealed(PASS_FIRST_SPEC)
    keys = ("family_a",) if target == "primary" else ("family_b",)
    primary = arm(
        spec=spec,
        source="self-repair",
        heldout=0.9,
        pairing_keys=keys if target == "primary" else ("family_a",),
    )
    control = arm(
        spec=spec,
        source="canonical",
        heldout=0.9,
        pairing_keys=("family_b",) if target == "primary" else keys,
    )

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert all(item["passed"] is False for item in verdict["rules"])
    assert all(item["unevaluable"] is True for item in verdict["rules"])


def test_a_late_refusal_exits_two_with_no_passing_rule_in_the_payload(tmp_path):
    spec = sealed(PASS_FIRST_SPEC)
    primary = arm(spec=spec, source="self-repair", heldout=0.9)
    control = arm(spec=spec, source="canonical", heldout=0.9)
    _null_metric(control)
    bind(control, spec)
    spec_path = tmp_path / "spec.json"
    primary_path = tmp_path / "self.json"
    control_path = tmp_path / "canonical.json"
    spec_path.write_text(json.dumps(spec))
    primary_path.write_text(json.dumps(primary))
    control_path.write_text(json.dumps(control))

    assert (
        checker.main(
            [
                "--spec",
                str(spec_path),
                "--report",
                str(primary_path),
                "--control-report",
                str(control_path),
            ]
        )
        == 2
    )


# --------------------------------------------------------------------------
# Review round 8: a non-finite delta input crashed the checker. It exited 1
# with a traceback and no verdict at all, so a run that could not be judged
# produced neither a result nor a refusal.
# --------------------------------------------------------------------------

NON_FINITE_VALUES = [float("nan"), float("inf"), float("-inf")]
NON_FINITE_IDS = ["nan", "inf", "-inf"]


def _non_finite_arm(spec, source, value):
    """An arm whose metric is non-finite.

    ``bind`` cannot seal it -- the canonical encoder rejects a non-finite float,
    which is exactly why no honest run can report one -- so the manifest is the
    one built from the finite value and the report reads as edited. That is the
    truth about such a document, and it must not stop the checker from saying
    which rule input is not a measurement.
    """
    from grove.provenance import UnrepresentableValue

    document = arm(spec=spec, source=source, heldout=0.9)
    document["experts"][0]["metrics"]["heldout_target_rate"] = value
    # The producer refuses too, which is why such a report is always an edit.
    with pytest.raises(UnrepresentableValue):
        bind(document, spec)
    return document


@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=NON_FINITE_IDS)
@pytest.mark.parametrize("target", ["primary", "control"])
def test_a_non_finite_delta_input_is_refused_with_a_verdict(value, target):
    spec = sealed(PAIRED_SPEC)
    if target == "primary":
        primary = _non_finite_arm(spec, "self-repair", value)
        control = arm(spec=spec, source="canonical", heldout=0.9)
    else:
        primary = arm(spec=spec, source="self-repair", heldout=0.9)
        control = _non_finite_arm(spec, "canonical", value)

    verdict = checker.check(spec, primary, control)

    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["rules_unevaluable"] == verdict["rules_total"]
    assert verdict["falsified_hypotheses"] == []
    assert all(item["passed"] is False for item in verdict["rules"])
    assert all(item["unevaluable"] is True for item in verdict["rules"])
    named = [
        blocker
        for blocker in verdict["blockers"]
        if "D1" in blocker
        and target in blocker
        and "experts[*].metrics.heldout_target_rate" in blocker
        and f"non-finite ({value})" in blocker
    ]
    assert named, verdict["blockers"]
    # The most specific statement is the verdict, not the digest it broke.
    assert verdict["verdict"] == named[0]


@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=NON_FINITE_IDS)
@pytest.mark.parametrize("target", ["primary", "control"])
def test_a_non_finite_delta_input_exits_two_with_json_on_stdout(
    tmp_path, capsys, value, target
):
    spec = sealed(PAIRED_SPEC)
    if target == "primary":
        primary = _non_finite_arm(spec, "self-repair", value)
        control = arm(spec=spec, source="canonical", heldout=0.9)
    else:
        primary = arm(spec=spec, source="self-repair", heldout=0.9)
        control = _non_finite_arm(spec, "canonical", value)
    spec_path = tmp_path / "spec.json"
    primary_path = tmp_path / "self.json"
    control_path = tmp_path / "canonical.json"
    spec_path.write_text(json.dumps(spec))
    primary_path.write_text(json.dumps(primary))
    control_path.write_text(json.dumps(control))

    assert (
        checker.main(
            [
                "--spec",
                str(spec_path),
                "--report",
                str(primary_path),
                "--control-report",
                str(control_path),
            ]
        )
        == 2
    )

    printed = capsys.readouterr().out
    assert printed.strip(), "the command must print a verdict, not nothing"
    verdict = json.loads(printed)
    assert verdict["unusable"] is True
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []


@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=NON_FINITE_IDS)
def test_a_non_finite_value_is_never_a_comparable_number(value):
    """NaN fails every threshold; an infinity passes every threshold."""
    assert checker._numeric(value) is None
    assert checker._delta_input_state([value]) == f"non-finite ({value})"
    passed, _ = checker.evaluate(">=", value, 0.0)
    assert passed is False
    passed, _ = checker.evaluate("delta>=", [value], -0.1)
    assert passed is False


def test_report_hash_marks_an_unrepresentable_payload_instead_of_raising():
    marker = checker._report_hash({"metric": float("nan")})

    assert marker.startswith("unrepresentable: ")
    assert marker != checker.canonical_hash({"metric": 0.0})
    # The canonical encoder itself still refuses; only the integrity path is
    # tolerant, because that is where a corrupt document arrives.
    with pytest.raises(checker.UnrepresentableValue):
        checker.canonical_hash({"metric": float("nan")})


@pytest.mark.parametrize(
    ("mutate", "state"),
    [
        (_null_metric, "null/unmeasured"),
        (_string_metric, "non-numeric (str)"),
        (_mapping_metric, "non-numeric (dict)"),
        (_bool_metric, "non-numeric (bool)"),
    ],
    ids=["null", "string", "mapping", "bool"],
)
def test_a_present_but_unusable_delta_input_is_not_called_absent(mutate, state):
    """The rule detail may not say the value is present and also absent."""
    spec = sealed(PAIRED_SPEC)
    primary, control = _paired_documents(spec)
    mutate(control)
    bind(control, spec)

    verdict = checker.check(spec, primary, control)

    detail = verdict["rules"][0]["detail"]
    assert state in detail
    assert "field is absent from the report" not in detail
    assert detail == (
        "rule D1: control delta path experts[*].metrics.heldout_target_rate "
        f"is {state}"
    )


@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=NON_FINITE_IDS)
def test_a_spec_holding_an_unhashable_value_is_refused_with_a_verdict(value):
    """The same defect through the spec door rather than the report door.

    ``spec_digest`` raised, so the command exited 1 with a traceback and no
    verdict -- and exit 1 is the code a caller reads as "a prediction failed".
    """
    unhashable = copy.deepcopy(BASE_SPEC)
    unhashable["decision_rules"][0]["value"] = value

    verdict = checker.check(
        unhashable, report(forced_rate=0.0, claim="adapter_intrinsic")
    )

    assert verdict["unusable"] is True
    assert verdict["spec_intact"] is False
    assert verdict["computed_spec_sha256"] is None
    assert verdict["rules_failed"] == 0
    assert verdict["falsified_hypotheses"] == []
    assert all(item["passed"] is False for item in verdict["rules"])
    assert all(item["unevaluable"] is True for item in verdict["rules"])
    assert verdict["verdict"].startswith(
        "spec holds a value no digest can commit to"
    )
    # Not reported as an edit: nothing was altered, the spec was never sealable.
    assert "spec altered after declaration" not in verdict["blockers"]


@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=NON_FINITE_IDS)
def test_an_unhashable_spec_exits_two_with_json_on_stdout(tmp_path, capsys, value):
    unhashable = copy.deepcopy(BASE_SPEC)
    unhashable["decision_rules"][0]["value"] = value
    spec_path = tmp_path / "spec.json"
    report_path = tmp_path / "report.json"
    spec_path.write_text(json.dumps(unhashable))
    report_path.write_text(
        json.dumps(report(forced_rate=0.0, claim="adapter_intrinsic"))
    )

    assert (
        checker.main(["--spec", str(spec_path), "--report", str(report_path)]) == 2
    )

    printed = capsys.readouterr().out
    assert printed.strip(), "the command must print a verdict, not nothing"
    assert json.loads(printed)["unusable"] is True


def test_an_altered_seal_is_still_reported_as_an_edit():
    """The control: a hashable spec edited after sealing keeps its own message."""
    spec = sealed(BASE_SPEC)
    bound = report(forced_rate=0.0, claim="adapter_intrinsic", spec=spec)
    spec["decision_rules"][0]["value"] = 0.9

    verdict = checker.check(spec, bound)

    assert verdict["spec_intact"] is False
    assert verdict["computed_spec_sha256"] is not None
    assert verdict["verdict"] == "spec altered after declaration"
