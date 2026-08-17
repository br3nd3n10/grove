#!/usr/bin/env python3
"""Check a finished run against a predeclared experiment spec.

The 2026-08-06 audit's "benchmark purity is limited" finding was that holdout
design and training settings moved after early attempts failed, so a passing
result cannot be distinguished from a setting that was tuned until it passed.
The fix is procedural, not statistical: write the decision rules down first,
hash them, and let a program -- not a narrative -- decide whether the run met
them.

Usage:
    check_experiment_spec.py --spec experiments/EXP-002-....json \\
        --report /srv/storage/grove/evaluations/second-real-cycle.json

    check_experiment_spec.py --spec experiments/EXP-003-....json \\
        --report exp003-self.json --control-report exp003-canonical.json

Exit codes:
    0  spec intact, report bound to it, and every decision rule satisfied
    1  spec intact and bound, but at least one rule failed (a falsified prediction)
    2  the run cannot be judged: the spec was altered, the report is not bound to
       this spec version, a required control arm is missing, or the two arms are
       not comparable

A rule failure is a real result, not a bug. Exit 1 means the run happened and
the prediction did not hold.

The spec digest is an integrity binding, not a timestamp. It proves that a
report was produced under the spec version it names; it does not prove when the
spec was written. Repository history supplies that, and an external timestamp or
signature would be stronger still.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SPEC_HASH_FIELD = "spec_sha256"
MISSING = object()
# Distinct from ``None`` and from ``{}``: a spec that never declared a
# control profile is not the same as one that declared an empty one, and
# collapsing the two is how a missing control_required_setup read as
# "spec declares no required_setup" and therefore conformant.
MISSING_DECLARATION = object()

# Setup declarations a program can check, versus declarations only a human can.
SETUP_PROFILE_KEYS = ("machine", "prose")
# Declarations no policy field backs. Accepting one and discarding it lets a
# spec name a gate that never runs.
UNSUPPORTED_SETUP_KEYS = ("min_route_precision",)
# Timing evidence types a preregistration claim would have to rest on. None of
# them has a verifier in this repository yet, so naming one is a statement of
# intent, not evidence. See ``timing_claim``.
SUPPORTED_TIMING_ATTESTATIONS = (
    "rfc3161",
    "osf_registration",
    "signed_tag",
    "transparency_log",
)
# The one field a manifest cannot commit to, because it *is* the commitment.
UNBINDABLE_REPORT_PATHS = ("run_manifest_sha256", "run_manifest")
# Every rule field this evaluator can dereference into a report.
RULE_PATH_FIELDS = ("path", "control_path", "pair_on")
NON_GATING_RULE_METRICS = (
    "route_recall_gold_tags",
    "route_recall_gold_tags_independent",
)


def _non_gating_rule_metric(path: Any) -> str | None:
    if not isinstance(path, str):
        return None
    for metric in NON_GATING_RULE_METRICS:
        if path == metric or path.endswith(f".{metric}"):
            return metric
    return None

DECISION_INPUT_SCHEMA = "grove-decision-input-v1"

SUPPORTED_DECISION_COMPARISONS = frozenset(
    {
        "exists",
        "count>=",
        "set==",
        "subset_of",
        "==",
        "!=",
        "<=",
        ">=",
        "<",
        ">",
        "in",
        "not_in",
        "delta>=",
        "delta<=",
    }
)

# The only two report sources a rule can name. ``arm`` used to be compared with
# ``== "control"`` and everything else routed to the primary report, so
# ``"control "`` -- one trailing space -- silently graded the wrong arm. An
# omitted ``arm`` still means primary; anything else is a schema error.
SUPPORTED_RULE_ARMS = frozenset({"primary", "control"})

def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _expected_value_problem(rule: Mapping[str, Any], index: int) -> str | None:
    """Return why a rule threshold/expected value cannot be graded."""
    comparison = rule.get("comparison")
    if comparison == "exists":
        return None
    if "value" not in rule:
        return f"decision_rules[{index}].value is required for {comparison!r}"

    value = rule.get("value")
    if comparison in {"count>=", "<=", ">=", "<", ">", "delta>=", "delta<="}:
        if not _is_finite_number(value):
            return (
                f"decision_rules[{index}].value for {comparison!r} must be a "
                "finite number"
            )
    elif comparison in {"set==", "subset_of", "in", "not_in"}:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return (
                f"decision_rules[{index}].value for {comparison!r} must be a list"
            )
    elif comparison in {"==", "!="}:
        return None
    return None


def decision_rule_problems(spec: Mapping[str, Any]) -> list[str]:
    """Return structural errors before any rule value is dereferenced."""
    rules = spec.get("decision_rules")
    if rules is None:
        return []
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        return ["decision_rules must be a sequence of mappings"]
    problems: list[str] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            problems.append(f"decision_rules[{index}] must be a mapping")
            continue
        path = rule.get("path")
        if not isinstance(path, str) or not path.strip():
            problems.append(
                f"decision_rules[{index}].path must be a nonempty string"
            )
        comparison = rule.get("comparison")
        if comparison not in SUPPORTED_DECISION_COMPARISONS:
            problems.append(
                f"decision_rules[{index}].comparison {comparison!r} is not "
                f"supported; expected one of {sorted(SUPPORTED_DECISION_COMPARISONS)}"
            )
        value_problem = _expected_value_problem(rule, index)
        if value_problem is not None:
            problems.append(value_problem)
        if "arm" in rule:
            arm = rule["arm"]
            if not isinstance(arm, str) or arm not in SUPPORTED_RULE_ARMS:
                problems.append(
                    f"decision_rules[{index}].arm {arm!r} is not supported; "
                    f"expected one of {sorted(SUPPORTED_RULE_ARMS)} or omission"
                )
        for field in RULE_PATH_FIELDS[1:]:
            if field in rule and (
                not isinstance(rule[field], str) or not rule[field].strip()
            ):
                problems.append(
                    f"decision_rules[{index}].{field} must be a nonempty string"
                )
        for field in RULE_PATH_FIELDS:
            metric = _non_gating_rule_metric(rule.get(field))
            if metric is not None:
                problems.append(
                    f"decision_rules[{index}].{field} names {metric!r}, a "
                    "diagnostic metric that cannot gate a decision rule"
                )
    return problems


def _decision_rules(spec: Mapping[str, Any]) -> list[Any]:
    raw = spec.get("decision_rules")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []

def rollback_evaluation_selector(audit: Any) -> dict[str, Any] | None:
    """Return the exact database row selector recorded by the runner."""
    if not isinstance(audit, Mapping):
        return None
    return {
        "evaluation_id": audit.get("evaluation_id"),
        "run_id": audit.get("run_id"),
    }


class UnrepresentableValue(TypeError):
    """A value a canonical digest must not silently stringify."""


def _reject_unrepresentable(payload: Any, path: str = "") -> None:
    where = path or "<root>"
    if payload is None or isinstance(payload, str | bool | int):
        return
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise UnrepresentableValue(
                f"{where}: NaN and infinity are not JSON"
            )
        return
    if isinstance(payload, list | tuple):
        for index, item in enumerate(payload):
            _reject_unrepresentable(item, f"{path}[{index}]")
        return
    if isinstance(payload, dict):
        for key, item in payload.items():
            if not isinstance(key, str):
                raise UnrepresentableValue(
                    f"{where}: object keys must be strings, got {type(key).__name__}"
                )
            _reject_unrepresentable(item, f"{path}.{key}" if path else key)
        return
    raise UnrepresentableValue(
        f"{where}: {type(payload).__name__} has no JSON representation; "
        "convert it explicitly rather than letting a digest hash its repr"
    )
def canonical_json(payload: Any) -> bytes:
    """The bytes ``grove.provenance.canonical_json`` produces.

    Duplicated rather than imported so this checker stays runnable against a
    report and a spec without the package installed. ``tests`` pin the two
    implementations to the same output, including the rejections.
    """
    _reject_unrepresentable(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    """The digest ``grove.provenance.canonical_hash`` produces."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _report_hash(payload: Any) -> str:
    """``canonical_hash`` over caller-supplied report data, without raising.

    The encoder rejects a non-finite float on purpose, and it should keep doing
    so: a digest must commit to what a document says, and ``NaN`` is not a
    value JSON can say. But an integrity check is exactly where a hostile or
    corrupt report arrives, and raising there made the command exit 1 with a
    traceback and no verdict at all -- neither a graded result nor a readable
    refusal. A payload that cannot be encoded gets a marker instead, which
    never equals a recorded digest, so the report is reported as edited.
    """
    try:
        return canonical_hash(payload)
    except UnrepresentableValue as error:
        return f"unrepresentable: {error}"


def spec_digest(spec: dict[str, Any]) -> str:
    """SHA-256 over the spec with its own hash field removed.

    Uses the same canonical encoder as every other commitment, so a seal, a
    provenance digest and a decision-input digest cannot drift apart.
    """
    payload = {key: value for key, value in spec.items() if key != SPEC_HASH_FIELD}
    return canonical_hash(payload)


def normalize_required_setup(declaration: Any) -> dict[str, dict[str, Any]]:
    """Split a ``required_setup`` declaration into machine and prose halves.

    Mirrors ``grove.experiment.normalize_required_setup``. A flat legacy map is
    read as entirely machine-checkable: a key nobody split out is a key somebody
    expects to be enforced.
    """
    if declaration is None:
        return {"machine": {}, "prose": {}}
    if not isinstance(declaration, dict):
        raise TypeError("required_setup must be a mapping")
    keys = set(declaration)
    if keys and keys <= set(SETUP_PROFILE_KEYS):
        for key in keys:
            if not isinstance(declaration[key], dict):
                raise TypeError(f"required_setup.{key} must be a mapping")
        return {
            "machine": dict(declaration.get("machine", {})),
            "prose": dict(declaration.get("prose", {})),
        }
    return {"machine": dict(declaration), "prose": {}}

def resolve(document: Any, path: str) -> Any:
    """Resolve a dotted path, where ``[*]`` fans out over a list.

    ``experts[*].metrics.forced_replay_measured`` yields a list with one entry
    per expert, so a rule can demand a property of every expert at once.
    """
    current: Any = document
    for segment in path.split("."):
        fanout = segment.endswith("[*]")
        key = segment[:-3] if fanout else segment
        if key:
            if isinstance(current, list):
                current = [
                    item.get(key, MISSING) if isinstance(item, dict) else MISSING
                    for item in current
                ]
            elif isinstance(current, dict):
                current = current.get(key, MISSING)
            else:
                return MISSING
        if fanout and (current is MISSING or not isinstance(current, list)):
            return MISSING
    return current


def decision_rule_input_paths(spec: dict[str, Any]) -> list[str]:
    """Every report path this spec's decision rules can read.

    Mirrors ``grove.experiment.decision_rule_input_paths``. The manifest binds
    what this returns, and the integrity check demands what this returns, so a
    rule cannot read a value nothing committed to.
    """
    paths: set[str] = set()
    for rule in _decision_rules(spec):
        if not isinstance(rule, Mapping):
            continue
        for field in RULE_PATH_FIELDS:
            value = rule.get(field)
            if (
                isinstance(value, str)
                and value.strip()
                and value not in UNBINDABLE_REPORT_PATHS
            ):
                paths.add(value)
    return sorted(paths)


def decision_input_binding(report: dict[str, Any], path: str) -> dict[str, Any]:
    """The binding a sealed report should carry for one rule input.

    Mirrors ``grove.experiment.decision_input_bindings`` for a single path.
    """
    value = resolve(report, path)
    present = value is not MISSING and MISSING not in _flatten(value)
    return {
        "present": present,
        "sha256": _report_hash(
            {"schema": DECISION_INPUT_SCHEMA, "path": path, "value": value}
        )
        if present
        else None,
    }


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        flat: list[Any] = []
        for item in value:
            flat.extend(_flatten(item))
        return flat
    return [value]


def evaluate(comparison: str, observed: Any, expected: Any) -> tuple[bool, str]:
    """Apply one comparison, reporting why it held or failed."""
    if observed is MISSING:
        return False, "field is absent from the report"
    if any(item is MISSING for item in _flatten(observed)):
        # A fanned-out path where some entry lacks the metric. Treat a partly
        # reported metric as a failure; silence must never read as compliance.
        return False, "field is absent from the report for at least one entry"
    if comparison == "exists":
        return True, "field is present"
    if comparison == "count>=":
        size = len(observed) if isinstance(observed, list | dict | str) else 0
        return size >= expected, f"count {size} vs required {expected}"
    if comparison == "set==":
        actual = sorted(observed) if isinstance(observed, list) else observed
        return actual == sorted(expected), f"observed {actual}"
    if comparison == "subset_of":
        # Partial provenance, declared in advance. A run may carry only the gaps
        # its spec already admitted it cannot close; anything else is a surprise
        # and fails. Passing this is never "complete provenance".
        if not isinstance(observed, list):
            return False, f"expected a list, observed {observed!r}"
        surplus = sorted(set(observed) - set(expected))
        return (
            not surplus,
            f"observed {sorted(observed)}, permitted {sorted(expected)}"
            + (f", undeclared {surplus}" if surplus else ""),
        )

    values = _flatten(observed)
    if not values:
        return False, "no values to compare"
    operators = {
        "==": lambda item: item == expected,
        "!=": lambda item: item != expected,
        "<=": lambda item: _numeric(item) is not None and _numeric(item) <= expected,
        ">=": lambda item: _numeric(item) is not None and _numeric(item) >= expected,
        "<": lambda item: _numeric(item) is not None and _numeric(item) < expected,
        ">": lambda item: _numeric(item) is not None and _numeric(item) > expected,
        "in": lambda item: item in expected,
        "not_in": lambda item: item not in expected,
        # The observed values are already primary-minus-control; see _delta.
        # Every pair is checked, so one compliant expert cannot cover another.
        "delta>=": lambda item: (
            _numeric(item) is not None and _numeric(item) >= expected
        ),
        "delta<=": lambda item: (
            _numeric(item) is not None and _numeric(item) <= expected
        ),
    }
    if comparison not in operators:
        raise ValueError(f"unsupported comparison {comparison!r}")
    check = operators[comparison]
    failures = [item for item in values if not check(item)]
    if failures:
        return False, f"observed {values}, offending {failures}"
    return True, f"observed {values}"


def _non_finite(value: Any) -> bool:
    """True for NaN and the infinities, which are not measurements."""
    return isinstance(value, float) and not math.isfinite(value)


def _numeric(value: Any) -> float | None:
    """The value as a float, or ``None`` when it cannot be compared.

    ``NaN`` is excluded on purpose. Every comparison against it is false, so a
    rule reading one failed quietly and reported a falsified hypothesis for a
    measurement that does not exist. The infinities are excluded for the same
    reason in reverse: ``inf >= threshold`` passes every threshold.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return None if _non_finite(value) else float(value)
    return None


# Everything two arms of a paired experiment must share. Anything absent from
# either arm, or present but unresolved, blocks the comparison: an experiment
# whose arms might differ in the training recipe is not a controlled experiment.
#
# The set deliberately covers actual run state, not only declared intent. The
# cohort hash names the catalog, which is fixed before either arm runs; the
# attempted failure ids and both hashes name what the live capture selected.
# The accepted trainable subset may differ by correction source, so it is bound
# in the manifest but is not a pairing identity path.
ARM_IDENTITY_PATHS = (
    "provenance.base_model",
    "provenance.verifiers.suites_sha256",
    "provenance.training_config_sha256",
    "provenance.decoding_config",
    "provenance.source.revision",
    "provenance.source.tree",
    "provenance.source.dirty",
    "provenance.source.status_sha256",
    "provenance.source.worktree_sha256",
    "provenance.worker.host",
    "provenance.worker.framework_versions_sha256",
    "provenance.worker.checkout.revision",
    "provenance.worker.checkout.tree",
    "provenance.worker.checkout.dirty",
    # The name of a dirty path is not its content, on the worker either. A
    # worker can report the same revision, tree and dirty flag while holding
    # different bytes.
    "provenance.worker.checkout.status_sha256",
    "provenance.worker.checkout.worktree_sha256",
    # Which model weights the worker actually held. Framework versions and a
    # checkout say nothing about this, so without it two arms can run different
    # models and still pair.
    "provenance.worker.model_manifest_sha256",
    "provenance.sandbox_image.fingerprint",
    "provenance.models.base.aggregate_sha256",
    "experiment_spec.spec_id",
    "experiment_spec.spec_sha256",
    "run_setup.admission_policy_sha256",
    "run_setup.cohort_manifest_sha256",
    "run_setup.actual_training_failure_ids",
    "run_setup.actual_training_failure_set_sha256",
    "run_setup.attempted_training_failure_set_sha256",
    "run_setup.self_repair_attempts",
    "run_setup.verifier_suite_version",
)

# Paths whose value must never be null or an unresolved collector marker. A
# null on both sides would otherwise "match" and wave the pairing through.
RESOLVED_IDENTITY_REQUIRED = True


# Grove identity digests are raw 64-character lowercase SHA-256 hex, the same
# rule ``grove.provenance.is_sha256_hex`` enforces on the producing side. The
# checker repeats it rather than trusting the producer: a report can be handed
# to this script directly, and ``false`` or ``{}`` in a digest field is not an
# identity two arms can match on.
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and SHA256_HEX.fullmatch(value) is not None


def _is_digest_path(path: str | None) -> bool:
    if not path:
        return False
    return path.rsplit(".", 1)[-1].endswith("_sha256")


def _unresolved(value: Any, path: str | None = None) -> bool:
    """True when a value is absent, null, blank, unavailable, or malformed.

    A ``*_sha256`` path additionally has to hold a real digest. Without that,
    two arms both reporting ``model_manifest_sha256: false`` compared equal and
    paired, which is the opposite of what the field is for.
    """
    if (
        value is MISSING
        or value is None
        or (
            isinstance(value, str)
            and (not value.strip() or value.strip().startswith("unavailable:"))
        )
    ):
        return True
    return _is_digest_path(path) and not _is_sha256_hex(value)


def _gap_name(path: str) -> str:
    """The ``provenance_gaps`` name for an identity path, if it has one.

    Gaps are recorded relative to the provenance record, identity paths
    relative to the report, so ``provenance.models.base.aggregate_sha256`` is
    the gap ``models.base.aggregate_sha256``.
    """
    return path.removeprefix("provenance.")


def permitted_provenance_gaps(spec: dict[str, Any]) -> list[str]:
    """Gaps the spec admitted in advance that it cannot close.

    An explicit ``permitted_provenance_gaps`` list wins. Otherwise the gaps a
    ``subset_of`` rule on ``provenance_gaps`` already permits are used, so a
    spec does not have to say the same thing twice.
    """
    declared = spec.get("permitted_provenance_gaps")
    if isinstance(declared, list):
        return sorted({str(item) for item in declared})
    permitted: set[str] = set()
    for rule in _decision_rules(spec):
        if not isinstance(rule, Mapping):
            continue
        if rule.get("path") == "provenance_gaps" and rule.get("comparison") in {
            "subset_of",
            "set==",
        }:
            permitted.update(str(item) for item in rule.get("value") or [])
    return sorted(permitted)


def required_resolved_identity(spec: dict[str, Any]) -> list[str]:
    """Identity paths this spec refuses to treat as a permitted gap.

    A permitted gap is an honest admission that one non-critical field cannot be
    resolved from here. It is not a licence to leave a field unresolved when
    that field is the only evidence the two arms used the same thing. EXP-003
    names the worker model manifest, because a worker can report the same
    checkout and the same framework versions while holding different weights.
    """
    declared = spec.get("required_resolved_identity")
    if not isinstance(declared, list):
        return []
    return sorted({str(item) for item in declared})


def control_requirement(spec: dict[str, Any]) -> dict[str, Any]:
    """Does this spec need a control arm? Ask the rules, not a flag.

    ``requires_control_report`` was the only thing that turned pairing on. With
    it removed or set false, a spec whose rules compute deltas against a control
    still ran them -- against a control that was never identity-checked. A pair
    of arms with different base models and different source revisions returned
    exit 0, ``arm_pairing: null`` and "all predeclared rules satisfied".

    A rule needs a control when it computes a delta, names a ``control_path``,
    pairs entries with ``pair_on``, or reads the control arm outright. That is
    derivable from the sealed spec, so it cannot be switched off. The flag is
    kept as metadata and cross-checked: a spec whose flag disagrees with its own
    rules is itself suspect.
    """
    reasons: list[dict[str, Any]] = []
    for rule in _decision_rules(spec):
        if not isinstance(rule, dict):
            continue
        why = []
        if str(rule.get("comparison", "")).startswith("delta"):
            why.append("delta comparison")
        if rule.get("control_path"):
            why.append("declares control_path")
        if rule.get("pair_on"):
            why.append("declares pair_on")
        if rule.get("arm") == "control":
            why.append("reads the control arm")
        if why:
            reasons.append({"rule": rule.get("id"), "because": why})
    declared = spec.get("requires_control_report")
    return {
        "required": bool(reasons),
        "rules": reasons,
        "declared_flag": declared,
        # A flag that says "no control needed" while the rules demand one is a
        # spec that contradicts itself, and grading it would pick a side.
        "flag_contradicts_rules": bool(reasons) and declared is False,
        "flag_overclaims": bool(declared) and not reasons,
    }


def spec_binding(spec: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Check that the report was produced under *this* sealed spec.

    A self-consistent spec file proves only that the file matches its own hash
    right now. Re-sealing an edited spec restores that consistency. The binding
    that survives re-sealing is the digest the run itself recorded, so a report
    made under the old spec fails against the new one.
    """
    recorded = spec.get(SPEC_HASH_FIELD)
    bound = resolve(report, "experiment_spec.spec_sha256")
    bound_id = resolve(report, "experiment_spec.spec_id")
    if bound is MISSING:
        return {
            "bound": False,
            "reason": "report records no experiment_spec.spec_sha256",
            "report_spec_sha256": None,
        }
    if bound_id is not MISSING and bound_id != spec.get("spec_id"):
        return {
            "bound": False,
            "reason": f"report was produced under spec {bound_id!r}",
            "report_spec_sha256": bound,
        }
    if bound != recorded:
        return {
            "bound": False,
            "reason": "report was produced under a different version of this spec",
            "report_spec_sha256": bound,
        }
    return {
        "bound": True,
        "reason": "report is bound to this spec version",
        "report_spec_sha256": bound,
    }


def arm_pairing(
    report: dict[str, Any],
    control: dict[str, Any],
    *,
    permitted_gaps: Sequence[str] = (),
    required_identity: Sequence[str] = (),
) -> dict[str, Any]:
    """Confirm two arms differ only in the variable under test.

    An identity field that is missing, null, or unresolved on either side is a
    mismatch, not a match. Two nulls agreeing proves nothing, and that is
    exactly how a pair of arms built from different source revisions used to
    slip through.

    A gap the spec declared in advance, and that both arms actually report, is
    the one exception. Pairing through it means the arms match on **partial**
    provenance. One arm resolved and the other not is still a mismatch, and an
    unresolved field nobody declared is still a mismatch.

    ``required_identity`` overrides the exception. A path listed there can never
    be waived: it is the evidence the comparison rests on, so leaving it
    unresolved is not partial provenance, it is no comparison. When such a path
    is unresolved, no other gap may be waived either -- a base-model gap is only
    tolerable while some other model identity still resolves.
    """
    permitted = set(permitted_gaps)
    required = set(required_identity)
    mismatches = []
    allowed_gaps = []
    report_gaps = set(report.get("provenance_gaps") or [])
    control_gaps = set(control.get("provenance_gaps") or [])

    def _pair(path: str) -> tuple[Any, Any, bool, bool]:
        left = resolve(report, path)
        right = resolve(control, path)
        return left, right, _unresolved(left, path), _unresolved(right, path)

    # Critical identity first: if it is missing, every waiver is withdrawn.
    unresolved_required: list[str] = []
    for path in required:
        _, _, left_gap, right_gap = _pair(path)
        if left_gap or right_gap:
            unresolved_required.append(path)
    waivers_allowed = not unresolved_required

    for path in ARM_IDENTITY_PATHS:
        left, right, left_gap, right_gap = _pair(path)
        gap = _gap_name(path)
        critical = path in required
        if left_gap and right_gap:
            if (
                waivers_allowed
                and not critical
                and gap in permitted
                and gap in report_gaps
                and gap in control_gaps
            ):
                allowed_gaps.append(gap)
                continue
            if critical:
                reason = (
                    f"{gap} is required for paired identity but is unavailable"
                )
            elif not waivers_allowed and gap in permitted:
                reason = (
                    "absent, null or unresolved on both arms; the spec permits "
                    "this gap, but a required identity path "
                    f"({sorted(unresolved_required)}) is itself unresolved, so "
                    "no gap may be waived"
                )
            elif gap in permitted:
                reason = (
                    "absent, null or unresolved on both arms; the spec permits "
                    "this gap but at least one arm does not report it in "
                    "provenance_gaps"
                )
            else:
                reason = "absent, null or unresolved on both arms"
            mismatches.append(
                {
                    "path": path,
                    "reason": reason,
                    "report": None if left is MISSING else left,
                    "control": None if right is MISSING else right,
                }
            )
        elif left_gap or right_gap:
            mismatches.append(
                {
                    "path": path,
                    "reason": (
                        f"{gap} is required for paired identity but is unavailable"
                        if critical
                        else "resolved on one arm and unresolved on the other"
                    ),
                    "report": None if left is MISSING else left,
                    "control": None if right is MISSING else right,
                }
            )
        elif left != right:
            mismatches.append(
                {"path": path, "reason": "differs", "report": left, "control": right}
            )
    # A required path outside ARM_IDENTITY_PATHS still has to resolve and match.
    for path in sorted(required - set(ARM_IDENTITY_PATHS)):
        left, right, left_gap, right_gap = _pair(path)
        if left_gap or right_gap:
            mismatches.append(
                {
                    "path": path,
                    "reason": (
                        f"{_gap_name(path)} is required for paired identity but "
                        "is unavailable"
                    ),
                    "report": None if left is MISSING else left,
                    "control": None if right is MISSING else right,
                }
            )
        elif left != right:
            mismatches.append(
                {"path": path, "reason": "differs", "report": left, "control": right}
            )
    left_arm = resolve(report, "correction_source")
    right_arm = resolve(control, "correction_source")
    if left_arm is MISSING or right_arm is MISSING:
        mismatches.append(
            {"path": "correction_source", "reason": "absent from one arm"}
        )
    elif left_arm == right_arm:
        mismatches.append(
            {
                "path": "correction_source",
                "reason": "both arms used the same correction source",
            }
        )
    return {
        "paired": not mismatches,
        "mismatches": mismatches,
        # Named loudly: a pair that relies on these is a partial-provenance
        # pair, never a fully reproducible one.
        "permitted_gaps_used": sorted(set(allowed_gaps)),
        "provenance_completeness": (
            "partial" if allowed_gaps else "complete_for_checked_identity"
        ),
    }


def setup_conformance(
    spec: dict[str, Any],
    report: dict[str, Any],
    *,
    declaration: Any | None = None,
    require_declaration: bool = False,
    label: str = "spec",
) -> dict[str, Any]:
    """Compare a declared ``required_setup`` with the run's recorded setup.

    A spec hash proves which declaration a run named. It does not prove the run
    was configured the way the declaration demanded. Without this, an EXP-002
    report produced under a different correction source passed every rule.

    A declared machine key the run setup does not record is a mismatch. It used
    to land in ``unchecked_keys`` and still return conformant, so deleting a key
    from a spec silently disabled the check it names. Prose declarations are
    reported separately and never counted as conformance -- a program cannot
    verify an authoring rule, and saying nothing must not read as saying yes.

    ``require_declaration`` is for the control arm. Deleting
    ``control_required_setup`` used to collapse to "spec declares no
    required_setup", which read as conformant: a control arm running with
    ``compare_corrections: false`` passed. A paired spec that declares nothing
    about its control arm has not checked its control arm.
    """
    if declaration is None:
        declaration = spec.get("required_setup")
    if declaration is MISSING_DECLARATION:
        declaration = None
    try:
        split = normalize_required_setup(declaration)
    except (TypeError, ValueError) as error:
        return {
            "conformant": False,
            "reason": f"required_setup schema is invalid: {error}",
            "checked": {},
            "missing_machine_keys": [],
            "unchecked_prose_keys": [],
            "unsupported_keys": [],
            "mismatches": [],
        }
    machine, prose = split["machine"], split["prose"]
    unsupported = sorted(
        key for key in (*machine, *prose) if key in UNSUPPORTED_SETUP_KEYS
    )
    if not machine:
        if require_declaration:
            return {
                "conformant": False,
                "reason": (
                    f"{label} declares no machine-checkable required_setup, so "
                    "nothing about this arm was checked. A paired design must "
                    "declare what its control arm was configured to do"
                ),
                "checked": {},
                "missing_machine_keys": [],
                "unchecked_prose_keys": sorted(prose),
                "unsupported_keys": unsupported,
                "mismatches": [],
            }
        if not prose:
            return {
                "conformant": True,
                "reason": "spec declares no required_setup",
                "checked": {},
                "missing_machine_keys": [],
                "unchecked_prose_keys": [],
                "unsupported_keys": unsupported,
                "mismatches": [],
            }
    setup = resolve(report, "run_setup")
    if setup is MISSING or not isinstance(setup, dict):
        return {
            "conformant": False,
            "reason": "spec declares required_setup but the report records no run_setup",
            "checked": {},
            "missing_machine_keys": sorted(machine),
            "unchecked_prose_keys": sorted(prose),
            "unsupported_keys": unsupported,
            "mismatches": [],
        }
    mismatches = []
    checked: dict[str, Any] = {}
    missing: list[str] = []
    for key, expected in machine.items():
        if key not in setup:
            missing.append(key)
            mismatches.append({"key": key, "declared": expected, "observed": None})
            continue
        checked[key] = setup[key]
        if setup[key] != expected:
            mismatches.append(
                {"key": key, "declared": expected, "observed": setup[key]}
            )
    conformant = not mismatches and not unsupported
    if conformant:
        reason = "run setup matches every machine-checkable declared key"
    elif missing:
        reason = (
            "run setup does not record every machine-checkable declared key: "
            f"{sorted(missing)}"
        )
    elif unsupported:
        reason = f"spec declares unsupported setup key(s) {unsupported}"
    else:
        reason = "run setup contradicts the sealed declaration"
    return {
        "conformant": conformant,
        "reason": reason,
        "checked": checked,
        "missing_machine_keys": sorted(missing),
        # Prose declarations a program cannot verify. Listed so nobody reads
        # silence as enforcement, and never counted towards conformance.
        "unchecked_prose_keys": sorted(prose),
        "unsupported_keys": unsupported,
        "mismatches": mismatches,
    }


def timing_claim(spec: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """What, if anything, evidences that the spec preceded the run.

    A spec digest proves the file matches itself and that a report names that
    version. It proves nothing about *when* the declaration was made. The old
    ``declared_before_run: true`` flag had no consumer anywhere in the
    repository, so a preregistration claim rested on an unread boolean.

    Replacing it with a self-reported ``{type, timestamp}`` map was no better.
    ``{"type": "rfc3161", "timestamp": "0000-not-a-time"}`` verified: nothing
    parsed the timestamp, read a token, checked a signature, or looked up a
    registration, and two timestamps were compared as strings. A field anyone
    can type is not evidence of anything.

    So there are exactly two outcomes today:

    * no attestation -- ``seal_self_consistent``, timing ``unverified``,
      non-blocking. This is the honest posture both shipped specs take.
    * any attestation, or any preregistration claim -- refused. Verifying an
      RFC 3161 token needs its DER bytes, the TSA signature, a trusted
      certificate, the message imprint bound to the spec digest, and a parsed
      ``genTime``. Verifying a signed tag, a Rekor entry or an OSF registration
      needs comparable artifacts. None of those verifiers exists here, so
      claiming the check passed would be the same fabrication in a new place.

    ``SUPPORTED_TIMING_ATTESTATIONS`` therefore lists what a verifier would
    have to handle, not what this code accepts.
    """
    attestation = spec.get("timing_attestation")
    claims_preregistration = bool(
        spec.get("declared_before_run") or spec.get("preregistered")
    )
    has_attestation = isinstance(attestation, dict) and bool(attestation)
    if not has_attestation and not claims_preregistration:
        return {
            "claim": "seal_self_consistent",
            "verified": False,
            "status": "unverified",
            "usable_for_preregistration_claim": False,
            "reason": (
                "the spec digest binds a report to a spec version; it is not a "
                "timestamp and no external timing attestation was supplied"
            ),
            "blocking": False,
        }
    if not has_attestation:
        return {
            "claim": "preregistered",
            "verified": False,
            "status": "unverified",
            "usable_for_preregistration_claim": False,
            "reason": (
                "the spec claims preregistration but supplies no timing "
                "attestation; a spec digest is not a timestamp"
            ),
            "blocking": True,
        }
    kind = attestation.get("type")
    known = kind in SUPPORTED_TIMING_ATTESTATIONS
    return {
        "claim": "preregistered",
        "verified": False,
        "status": (
            "attestation_verifier_not_implemented" if known else "unsupported_attestation"
        ),
        "usable_for_preregistration_claim": False,
        "attestation_type": kind if isinstance(kind, str) else None,
        "reason": (
            (
                f"timing attestation type {kind!r} has no verifier in this "
                "repository. A self-reported type and timestamp are not "
                "evidence: verifying this claim needs the signed artifact "
                "itself, a trusted key or log checkpoint, and a timestamp "
                "parsed out of that artifact rather than typed beside it"
            )
            if known
            else (
                f"timing attestation type {kind!r} is not one of "
                f"{sorted(SUPPORTED_TIMING_ATTESTATIONS)}"
            )
        ),
        "blocking": True,
    }


def report_integrity(
    spec: dict[str, Any], report: dict[str, Any], *, strict: bool | None = None
) -> dict[str, Any]:
    """Recompute the digests a report carries about itself.

    Every number the rules read comes out of an editable JSON file. Without
    this, changing ``forced_regression_rate`` from 0.5 to 0.0 and
    ``forgetting_claim`` from ``router_shielded`` to ``adapter_intrinsic``
    turned a falsified run into "all predeclared rules satisfied".

    Recomputed here: the provenance digest, the run-manifest digest, every
    decision-rule input the sealed spec can read, each expert's metrics and
    adapter digest, and the manifest's other bindings back into the report body.

    Absence is not neutrality. A report with no provenance digest used to record
    ``intact: null`` and raise no problem, so a strict paired report sealed
    without one could have *both* arms' source revisions edited and still return
    exit 0 with ``arm_pairing: paired``. Under a spec that requires integrity, a
    missing digest is now ``unverified`` and unusable; without one it is
    reported as ``unverified`` rather than passed over.

    This detects edits. It does not defeat an attacker who can rewrite the whole
    file, because nothing here is signed or externally anchored.
    """
    if strict is None:
        strict = bool(spec.get("requires_report_integrity") or _decision_rules(spec))
    problems: list[str] = []
    unverified: list[str] = []
    checks: dict[str, Any] = {}
    provenance = report.get("provenance")
    recorded = (
        provenance.get("provenance_sha256") if isinstance(provenance, dict) else None
    )
    if isinstance(provenance, dict) and isinstance(recorded, str) and recorded:
        payload = {
            key: value
            for key, value in provenance.items()
            if key != "provenance_sha256"
        }
        computed = _report_hash(payload)
        checks["provenance_sha256"] = {
            "recorded": recorded,
            "computed": computed,
            "intact": recorded == computed,
        }
        if recorded != computed:
            problems.append("provenance was edited after it was hashed")
    else:
        checks["provenance_sha256"] = {
            "recorded": recorded,
            "computed": None,
            "intact": None,
            "reason": (
                "the report records no provenance_sha256, so its provenance is "
                "not bound to anything"
            ),
        }
        unverified.append(
            "provenance carries no digest, so edits to it cannot be detected"
        )

    manifest = report.get("run_manifest")
    recorded_manifest = report.get("run_manifest_sha256")
    if not isinstance(manifest, dict) or not recorded_manifest:
        return {
            "bound": False,
            "status": "unbound",
            "authoritative": False,
            "reason": (
                "the report carries no run_manifest and run_manifest_sha256, so "
                "its metrics are not bound to anything"
            ),
            "checks": checks,
            "problems": problems,
            "unverified": unverified,
        }
    computed_manifest = _report_hash(manifest)
    checks["run_manifest_sha256"] = {
        "recorded": recorded_manifest,
        "computed": computed_manifest,
        "intact": recorded_manifest == computed_manifest,
    }
    if recorded_manifest != computed_manifest:
        problems.append("run manifest was edited after it was hashed")

    bindings: list[dict[str, Any]] = []

    def _bind(name: str, expected: Any, observed: Any) -> None:
        bindings.append(
            {"binding": name, "manifest": expected, "report": observed,
             "intact": expected == observed}
        )
        if expected != observed:
            problems.append(f"{name} does not match the run manifest")

    _bind(
        "run_setup_sha256",
        manifest.get("run_setup_sha256"),
        _report_hash(report.get("run_setup") or {}),
    )
    run_setup = report.get("run_setup") or {}
    _bind(
        "actual_training_failure_set_sha256",
        manifest.get("actual_training_failure_set_sha256"),
        run_setup.get("actual_training_failure_set_sha256"),
    )
    _bind(
        "attempted_training_failure_set_sha256",
        manifest.get("attempted_training_failure_set_sha256"),
        run_setup.get("attempted_training_failure_set_sha256"),
    )
    _bind(
        "trained_failure_set_sha256",
        manifest.get("trained_failure_set_sha256"),
        run_setup.get("trained_failure_set_sha256"),
    )
    _bind(
        "provenance_sha256",
        manifest.get("provenance_sha256"),
        (report.get("provenance") or {}).get("provenance_sha256"),
    )
    _bind(
        "experiment_spec",
        manifest.get("experiment_spec"),
        report.get("experiment_spec"),
    )
    _bind(
        "correction_source",
        manifest.get("correction_source"),
        report.get("correction_source"),
    )
    _bind(
        "provenance_gaps",
        list(manifest.get("provenance_gaps") or []),
        list(report.get("provenance_gaps") or []),
    )
    _bind(
        "training_proposals_sha256",
        manifest.get("training_proposals_sha256"),
        _report_hash(report.get("training_proposals") or []),
    )
    if "rollback" in report or "rollback_sha256" in manifest:
        _bind(
            "rollback_sha256",
            manifest.get("rollback_sha256"),
            _report_hash(report.get("rollback")),
        )
    if "rollback_audit" in report or "rollback_audit_sha256" in manifest:
        _bind(
            "rollback_audit_sha256",
            manifest.get("rollback_audit_sha256"),
            _report_hash(report.get("rollback_audit")),
        )
    if "rollback_audit" in report or "rollback_evaluation_selector" in manifest:
        _bind(
            "rollback_evaluation_selector",
            manifest.get("rollback_evaluation_selector"),
            rollback_evaluation_selector(report.get("rollback_audit")),
        )
    if "evaluation_ids" in report or "evaluation_ids" in manifest:
        _bind(
            "evaluation_ids",
            manifest.get("evaluation_ids"),
            list(report.get("evaluation_ids") or []),
        )

    manifest_experts = {
        entry.get("id"): entry for entry in manifest.get("experts") or []
    }
    report_experts = report.get("experts") or []
    if len(manifest_experts) != len(report_experts):
        problems.append("the manifest and the report list different experts")
    for expert in report_experts:
        entry = manifest_experts.get(expert.get("id"))
        if entry is None:
            problems.append(f"expert {expert.get('id')} is absent from the manifest")
            continue
        observed = _report_hash(expert.get("metrics") or {})
        bindings.append(
            {
                "binding": f"experts[{expert.get('id')}].metrics_sha256",
                "manifest": entry.get("metrics_sha256"),
                "report": observed,
                "intact": entry.get("metrics_sha256") == observed,
            }
        )
        if entry.get("metrics_sha256") != observed:
            problems.append(
                f"expert {expert.get('id')} metrics were edited after the run"
            )
        # The adapter digest identifies the trained weights. No rule reads it
        # today, so forging it was not a rule false pass -- but the report still
        # claimed integrity over a value nothing checked.
        manifest_adapter = entry.get("adapter_sha256")
        report_adapter = (expert.get("artifact") or {}).get("adapter_sha256")
        bindings.append(
            {
                "binding": f"experts[{expert.get('id')}].adapter_sha256",
                "manifest": manifest_adapter,
                "report": report_adapter,
                "intact": manifest_adapter == report_adapter,
            }
        )
        if manifest_adapter != report_adapter:
            problems.append(
                f"expert {expert.get('id')} adapter digest does not match the "
                "run manifest"
            )

    # The generic binding: every path the sealed spec's own rules can read.
    # Naming bound fields by hand is how three EXP-003 rule inputs stayed
    # unbound, so the expected set is derived from the spec instead.
    expected_paths = decision_rule_input_paths(spec)
    recorded_inputs = manifest.get("decision_inputs")
    if expected_paths and not isinstance(recorded_inputs, dict):
        problems.append(
            "the run manifest binds no decision inputs, so every value the "
            f"rules read is unbound: {expected_paths}"
        )
        recorded_inputs = {}
    recorded_inputs = recorded_inputs if isinstance(recorded_inputs, dict) else {}
    input_checks: list[dict[str, Any]] = []
    for path in expected_paths:
        recorded = recorded_inputs.get(path)
        observed = decision_input_binding(report, path)
        if not isinstance(recorded, dict):
            problems.append(f"decision input {path} is not bound by the manifest")
            input_checks.append(
                {"path": path, "manifest": None, "report": observed, "intact": False}
            )
            continue
        intact = recorded.get("present") == observed["present"] and recorded.get(
            "sha256"
        ) == observed["sha256"]
        input_checks.append(
            {
                "path": path,
                "manifest": recorded,
                "report": observed,
                "intact": intact,
            }
        )
        if not intact:
            problems.append(f"decision input {path} was edited after the run")
    for path, recorded in sorted(recorded_inputs.items()):
        if path in set(expected_paths) or not isinstance(recorded, dict):
            continue
        observed = decision_input_binding(report, path)
        if recorded.get("sha256") != observed["sha256"]:
            problems.append(f"decision input {path} was edited after the run")
            input_checks.append(
                {
                    "path": path,
                    "manifest": recorded,
                    "report": observed,
                    "intact": False,
                }
            )
    checks["decision_inputs"] = input_checks
    checks["bindings"] = bindings
    # Three states, never two. "Nothing disagrees" is not the same as
    # "everything was checked", and collapsing them is how a report with no
    # provenance digest passed as intact.
    if problems:
        status = "tampered"
        reason = "; ".join(problems)
    elif unverified:
        status = "unverified"
        reason = "; ".join(unverified)
    else:
        status = "intact"
        reason = "every bound value matches the run manifest"
    return {
        "bound": not problems and not (strict and unverified),
        "status": status,
        "authoritative": status == "intact",
        "reason": reason,
        "checks": checks,
        "problems": problems,
        "unverified": unverified,
    }


def _rule_input_sources(
    rule: Mapping[str, Any],
    report: dict[str, Any],
    control: dict[str, Any] | None,
) -> list[tuple[str, str, dict[str, Any], str]]:
    """Every (arm, kind, document, path) one rule actually reads."""
    path = rule.get("path")
    if not isinstance(path, str) or not path.strip():
        return []
    sources: list[tuple[str, str, dict[str, Any], str]] = []
    if str(rule.get("comparison", "")).startswith("delta"):
        sources.append(("primary", "delta", report, path))
        control_path = rule.get("control_path", path)
        if control is not None and isinstance(control_path, str):
            sources.append(("control", "delta", control, control_path))
    elif rule.get("arm") == "control":
        if control is not None:
            sources.append(("control", "input", control, path))
    else:
        sources.append(("primary", "input", report, path))
    pair_on = rule.get("pair_on")
    if isinstance(pair_on, str) and pair_on.strip():
        sources.append(("primary", "pairing key", report, pair_on))
        if control is not None:
            sources.append(("control", "pairing key", control, pair_on))
    return sources


def non_finite_rule_inputs(
    spec: dict[str, Any],
    report: dict[str, Any],
    control: dict[str, Any] | None,
) -> list[str]:
    """Rule inputs holding NaN or an infinity, named by rule, arm and path.

    No honest run can report one. Sealing a report hashes its metrics, and the
    canonical encoder rejects a non-finite float, so a report carrying one was
    edited after the seal. It used to reach the integrity hash first and raise
    there: the command exited 1 with a traceback and printed no verdict, so a
    run that could not be judged produced neither a result nor a refusal.

    Checked before grading, because ``NaN`` compares false against every
    threshold and an infinity compares true against every threshold. Either one
    would otherwise be published as a decided rule.
    """
    problems: list[str] = []
    for index, rule in enumerate(_decision_rules(spec)):
        if not isinstance(rule, Mapping):
            continue
        rule_id = rule.get("id", f"rule_{index}")
        for arm, kind, document, path in _rule_input_sources(rule, report, control):
            offending = [
                value for value in _flatten(resolve(document, path))
                if _non_finite(value)
            ]
            if offending:
                problems.append(
                    f"rule {rule_id}: {arm} {kind} path {path} is non-finite "
                    f"({offending[0]})"
                )
    return problems


def spec_substance(spec: dict[str, Any]) -> dict[str, Any]:
    """A sealed spec with nothing in it must not be able to pass.

    Zero rules and zero hypotheses would otherwise return exit 0 and read as a
    satisfied preregistration.
    """
    rules = _decision_rules(spec)
    hypotheses = spec.get("hypotheses") or []
    missing = []
    if not rules:
        missing.append("decision_rules")
    if not hypotheses:
        missing.append("hypotheses")
    if any(not str(item.get("falsified_if", "")).strip() for item in hypotheses):
        missing.append("hypotheses[].falsified_if")
    return {
        "substantive": not missing,
        "missing": missing,
        "rule_count": len(rules),
        "hypothesis_count": len(hypotheses),
    }


def check(
    spec: dict[str, Any],
    report: dict[str, Any],
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recorded = spec.get(SPEC_HASH_FIELD)
    # A spec holding a value no digest can commit to is not a sealed spec, and
    # raising here exited 1 with a traceback and no verdict -- the same code a
    # caller reads as "the run happened and a prediction failed".
    try:
        computed: str | None = spec_digest(spec)
        unhashable_spec: str | None = None
    except UnrepresentableValue as error:
        computed = None
        unhashable_spec = str(error)
    spec_intact = unhashable_spec is None and recorded == computed
    substance = spec_substance(spec)
    rule_errors = decision_rule_problems(spec)
    rules = _decision_rules(spec)
    binding = spec_binding(spec, report)
    conformance = setup_conformance(spec, report)
    timing = timing_claim(spec, report)
    # A decision rule grades values from the report, so its integrity binding is
    # mandatory even when the legacy opt-in flag is absent. The flag remains
    # accepted for sealed-spec compatibility, but it cannot switch this gate off.
    requires_integrity = bool(spec.get("requires_report_integrity") or rules)
    integrity = report_integrity(spec, report, strict=requires_integrity)
    control_integrity = (
        report_integrity(spec, control, strict=requires_integrity)
        if control is not None
        else None
    )
    # Derived from the rules, never from the optional flag: a spec whose rules
    # compute deltas needs a control whether or not it says so.
    control_need = control_requirement(spec)
    requires_control = control_need["required"]
    permitted_gaps = permitted_provenance_gaps(spec)
    required_identity = required_resolved_identity(spec)
    pairing: dict[str, Any] | None = None
    control_conformance: dict[str, Any] | None = None
    if requires_control and control is not None:
        pairing = arm_pairing(
            report,
            control,
            permitted_gaps=permitted_gaps,
            required_identity=required_identity,
        )
        control_conformance = setup_conformance(
            spec,
            control,
            declaration=spec.get("control_required_setup", MISSING_DECLARATION),
            require_declaration=True,
            label="spec control_required_setup",
        )

    results: list[dict[str, Any]] = []
    pairing_errors: list[str] = []
    # A rule that needs a control it never got compared nothing. Marking it
    # failed reported "H2 falsified" for an experiment that did not happen.
    control_absent = requires_control and control is None
    # A tampered report grades nothing. Its rule outcomes describe values the
    # run did not produce, so reporting one as a falsified hypothesis would
    # publish an edit as a scientific result. An unverified strict report is the
    # same: nothing established that the values are the ones the run produced.
    report_data_is_graded = bool(rules)
    unbound_report = integrity["status"] == "unbound" and report_data_is_graded
    unbound_control = (
        control_integrity is not None
        and control_integrity["status"] == "unbound"
        and requires_control
    )

    # Phase one: everything that has to hold before a rule value means
    # anything. These are collected *before* any rule is graded, because a
    # truth value computed under one of them is not a scientific outcome. The
    # checker used to grade first and refuse afterwards, so a run could return
    # ``unusable: true`` alongside ``rules_failed: 2`` and a falsified
    # hypothesis -- a published claim about a run it had just said could not be
    # judged.
    blockers: list[str] = []
    if unhashable_spec is not None:
        blockers.append(
            "spec holds a value no digest can commit to, so it is not sealed: "
            f"{unhashable_spec}"
        )
    # Early, because it is the most specific statement available and because a
    # non-finite value is the reason every later digest over it is a marker.
    blockers.extend(
        f"rule input is not a measurement: {problem}"
        for problem in non_finite_rule_inputs(spec, report, control)
    )
    if rule_errors:
        blockers.extend(f"decision rule schema: {error}" for error in rule_errors)
    if not spec_intact and unhashable_spec is None:
        blockers.append("spec altered after declaration")
    if not substance["substantive"]:
        blockers.append(
            "spec declares nothing falsifiable: missing "
            + ", ".join(substance["missing"])
        )
    if not binding["bound"]:
        blockers.append(f"report is not bound to this spec: {binding['reason']}")
    if not conformance["conformant"]:
        blockers.append(f"primary arm setup: {conformance['reason']}")
    if timing["blocking"]:
        blockers.append(f"timing claim: {timing['reason']}")
    if integrity["problems"] or unbound_report or (
        requires_integrity and not integrity["bound"]
    ):
        blockers.append(f"report integrity: {integrity['reason']}")
    if control_integrity is not None and (
        control_integrity["problems"]
        or unbound_control
        or (requires_integrity and not control_integrity["bound"])
    ):
        blockers.append(f"control report integrity: {control_integrity['reason']}")
    if control_need["flag_contradicts_rules"]:
        blockers.append(
            "spec sets requires_control_report false while its own rules "
            f"need a control arm: {control_need['rules']}"
        )
    if control_absent:
        blockers.append(
            "the sealed rules need a control report and none was supplied: "
            f"{control_need['rules']}"
        )
    if pairing is not None and not pairing["paired"]:
        blockers.append("control and primary arms are not comparable")
    if control_conformance is not None and not control_conformance["conformant"]:
        blockers.append(f"control arm setup: {control_conformance['reason']}")

    if blockers:
        # Nothing is dereferenced or compared. Each rule is emitted so the
        # output stays readable, and every one of them says why it was not
        # graded rather than carrying an outcome nobody may cite.
        for index, raw_rule in enumerate(rules):
            rule = raw_rule if isinstance(raw_rule, Mapping) else {}
            if rule_errors:
                errors = [
                    error
                    for error in rule_errors
                    if f"decision_rules[{index}]" in error
                ] or rule_errors
                detail = "decision rule schema is invalid: " + "; ".join(errors)
            else:
                detail = f"run unusable: {blockers[0]}"
            results.append(
                {
                    "id": rule.get("id", f"rule_{index}"),
                    "hypothesis": rule.get("hypothesis"),
                    "description": rule.get("description", ""),
                    "arm": rule.get("arm", "primary"),
                    "path": rule.get("path"),
                    "comparison": rule.get("comparison"),
                    "expected": rule.get("value"),
                    "passed": False,
                    "unevaluable": True,
                    "detail": detail,
                }
            )
    else:
        # Phase two: the run is judgeable, so rule truth values are real
        # results. A pairing failure discovered here still withdraws them all.
        for index, rule in enumerate(rules):
            source = control if rule.get("arm") == "control" else report
            pairing_error = None
            evaluation_error = None
            if rule["comparison"].startswith("delta"):
                observed, detail_prefix, pairing_error = _delta(
                    rule, report, control
                )
            else:
                observed = (
                    resolve(source, rule["path"]) if source is not None else MISSING
                )
                detail_prefix = ""
            if pairing_error:
                # Nothing was compared, so nothing is evaluated. Feeding the
                # MISSING sentinel to ``evaluate`` appended "field is absent
                # from the report" to a detail that had just said the value was
                # present and non-numeric -- one rule contradicting itself.
                pairing_errors.append(pairing_error)
                passed = False
                detail = pairing_error
                detail_prefix = ""
            else:
                try:
                    passed, detail = evaluate(
                        rule["comparison"], observed, rule.get("value")
                    )
                except (TypeError, ValueError) as error:
                    passed = False
                    detail = f"rule value cannot be evaluated: {error}"
                    evaluation_error = str(error)
            results.append(
                {
                    "id": rule.get("id", f"rule_{index}"),
                    "hypothesis": rule.get("hypothesis"),
                    "description": rule.get("description", ""),
                    "arm": rule.get("arm", "primary"),
                    "path": rule["path"],
                    "comparison": rule["comparison"],
                    "expected": rule.get("value"),
                    "passed": passed,
                    "unevaluable": bool(pairing_error)
                    or evaluation_error is not None,
                    "detail": f"{detail_prefix}{detail}",
                }
            )
        for error in pairing_errors:
            blockers.append(f"arms could not be paired: {error}")
        if blockers:
            # One unpairable rule makes the whole run unjudgeable, so no rule
            # outcome survives as a scientific claim. ``passed`` is cleared with
            # ``unevaluable``: a rule graded before the refusal was reached kept
            # ``passed: true`` inside a payload that says the run could not be
            # judged, which is the same overclaim as a falsified hypothesis,
            # only in the flattering direction.
            for item in results:
                item["passed"] = False
                item["unevaluable"] = True

    failed = [
        item for item in results if not item["passed"] and not item["unevaluable"]
    ]
    unevaluable = [item for item in results if item["unevaluable"]]
    falsified = sorted(
        {item["hypothesis"] for item in failed if item.get("hypothesis")}
    )
    unusable = bool(blockers)
    if blockers:
        verdict = blockers[0]
    elif failed:
        verdict = "predeclared rule(s) failed"
    else:
        verdict = "all predeclared rules satisfied"
    return {
        "spec_id": spec.get("spec_id"),
        "spec_intact": spec_intact,
        "spec_substance": substance,
        "decision_rule_validation": {
            "valid": not rule_errors,
            "errors": rule_errors,
        },
        "spec_binding": binding,
        "setup_conformance": conformance,
        "control_setup_conformance": control_conformance,
        "control_required": requires_control,
        "control_requirement": control_need,
        "control_supplied": control is not None,
        "arm_pairing": pairing,
        "permitted_provenance_gaps": permitted_gaps,
        "required_resolved_identity": required_identity,
        "timing": timing,
        "report_integrity": integrity,
        "control_report_integrity": control_integrity,
        "recorded_spec_sha256": recorded,
        "computed_spec_sha256": computed,
        "rules_total": len(results),
        "rules_failed": len(failed),
        "rules_unevaluable": len(unevaluable),
        "falsified_hypotheses": falsified,
        "blockers": blockers,
        "unusable": unusable,
        "verdict": verdict,
        "rules": results,
    }


def _pairing_keys(
    document: dict[str, Any], key_path: str
) -> tuple[list[Any] | None, str]:
    """Stable identifiers for the fanned-out entries of one arm.

    Expert ids are generated UUIDs, so positional order is not a pairing and
    neither is the id itself: the same training data yields a different id every
    run. A spec should name a correction-source-independent key such as
    ``experts[*].pairing_key``. A missing, null or duplicated key is refused
    rather than guessed at.
    """
    keys = _flatten(resolve(document, key_path))
    if not keys or MISSING in keys:
        return None, "absent on at least one entry"
    if any(
        key is None
        or (isinstance(key, str) and not key.strip())
        for key in keys
    ):
        return None, "null or empty on at least one entry"
    rendered = [str(key) for key in keys]
    if len(set(rendered)) != len(rendered):
        return None, "duplicated across entries"
    return keys, ""


def _delta_input_state(values: list[Any]) -> str | None:
    """Why one arm's delta input cannot enter a subtraction, if it cannot.

    Absent, unmeasured and non-numeric are three different facts, and none of
    them is a measured difference. Collapsing them into ``MISSING`` turned an
    unmeasured metric into a failed rule and a falsified hypothesis, which
    claims the experiment ran and the prediction did not hold.
    """
    if not values or MISSING in values:
        return "absent"
    for value in values:
        if value is None:
            return "null/unmeasured"
    for value in values:
        if _non_finite(value):
            return f"non-finite ({value})"
    for value in values:
        if _numeric(value) is None:
            return f"non-numeric ({type(value).__name__})"
    return None


def _delta(
    rule: dict[str, Any], report: dict[str, Any], control: dict[str, Any] | None
) -> tuple[Any, str, str | None]:
    """Every primary value minus its paired control value.

    Returns a list so ``evaluate`` applies the declared tolerance to each pair.
    Taking only the first entry silently exempted every expert after the first,
    which is how a second expert with a -0.8 delta passed a -0.1 tolerance.

    The third return value is a pairing error. A pairing that cannot be made is
    not a falsified hypothesis -- nothing was compared -- so the caller turns it
    into a blocker and exit 2 rather than a failed rule and exit 1.
    """
    rule_id = rule.get("id", "<unnamed rule>")
    if control is None:
        return MISSING, "no control report: ", None
    primary_path = rule["path"]
    control_path = rule.get("control_path", primary_path)
    primary = _flatten(resolve(report, primary_path))
    baseline = _flatten(resolve(control, control_path))
    for arm, values, arm_path in (
        ("primary", primary, primary_path),
        ("control", baseline, control_path),
    ):
        state = _delta_input_state(values)
        if state is not None:
            note = f"{arm} delta path {arm_path} is {state}"
            return MISSING, f"{note}: ", f"rule {rule_id}: {note}"
    if len(primary) != len(baseline):
        note = (
            f"arms have {len(primary)} and {len(baseline)} values,"
            f" so no pairing is possible"
        )
        return MISSING, f"{note}: ", f"rule {rule_id}: {note}"

    order = list(range(len(primary)))
    pairing_note = "positional"
    key_path = rule.get("pair_on")
    if key_path:
        left_keys, left_reason = _pairing_keys(report, key_path)
        right_keys, right_reason = _pairing_keys(control, key_path)
        reason = left_reason or right_reason
        if left_keys is None or right_keys is None:
            note = f"cannot pair arms on {key_path}: key {reason}"
        elif len(left_keys) != len(primary):
            note = (
                f"cannot pair arms on {key_path}: {len(left_keys)} keys for"
                f" {len(primary)} values"
            )
        elif sorted(map(str, left_keys)) != sorted(map(str, right_keys)):
            note = (
                f"cannot pair arms on {key_path}: the arms carry different key"
                " sets, so no expert has a counterpart"
            )
        else:
            note = ""
        if note:
            return MISSING, f"{note}: ", f"rule {rule_id}: {note}"
        lookup = {str(key): index for index, key in enumerate(right_keys)}
        order = [lookup[str(key)] for key in left_keys]
        pairing_note = f"paired on {key_path}"
    elif len(primary) > 1:
        note = (
            f"{len(primary)} values per arm but the rule declares no"
            f" pair_on key, so the pairing is ambiguous"
        )
        return MISSING, f"{note}: ", f"rule {rule_id}: {note}"

    deltas = []
    for index, position in enumerate(order):
        # Both arms were classified above, so every value here is numeric.
        deltas.append(_numeric(primary[index]) - _numeric(baseline[position]))
    return (
        deltas,
        f"{pairing_note}; primary {primary} vs control {baseline}, deltas {deltas}; ",
        None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--control-report",
        type=Path,
        help="matched control-arm report, for specs that declare a paired design",
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help="write the spec's own sha256 into it; use once, before the run",
    )
    parser.add_argument(
        "--reseal",
        action="store_true",
        help=(
            "allow --seal to overwrite an existing digest. Any report bound to "
            "the previous digest will then fail, which is the intended cost."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    if args.seal:
        existing = spec.get(SPEC_HASH_FIELD)
        if existing and not args.reseal:
            print(
                json.dumps(
                    {
                        "error": "spec is already sealed",
                        SPEC_HASH_FIELD: existing,
                        "hint": "pass --reseal to overwrite; bound reports will fail",
                    },
                    indent=2,
                )
            )
            return 2
        spec[SPEC_HASH_FIELD] = spec_digest(spec)
        args.spec.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"sealed": str(args.spec), SPEC_HASH_FIELD: spec[SPEC_HASH_FIELD]},
                indent=2,
            )
        )
        return 0
    if args.report is None:
        parser.error("--report is required unless --seal is given")
    report = json.loads(args.report.read_text())
    if not isinstance(report, dict):
        print(json.dumps({"error": "report JSON must contain an object"}, indent=2))
        return 2
    control = None
    if args.control_report is not None:
        control = json.loads(args.control_report.read_text())
        if not isinstance(control, dict):
            print(
                json.dumps(
                    {"error": "control report JSON must contain an object"}, indent=2
                )
            )
            return 2
    verdict = check(spec, report, control)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if verdict["unusable"]:
        return 2
    return 1 if verdict["rules_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
