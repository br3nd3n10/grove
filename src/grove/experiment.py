from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from grove.benchmark import LongitudinalBenchmark
from grove.coding_tasks import CodingTask, coding_catalog, second_cycle_catalog
from grove.corrections import (
    CanonicalReferenceSource,
    SelfRepairSource,
    correction_proposals,
    summarize_correction_sources,
)
from grove.mlx_backend import MlxRemoteBackend
from grove.mlx_trainer import MlxLoraTrainer
from grove.models import Artifact, DatasetRole, Expert, Task, utc_now
from grove.provenance import (
    canonical_hash,
    canonical_json,
    collect_provenance,
    git_revision,
    is_digest_field,
    is_sha256_hex,
    missing_fields,
    worker_metadata,
)
from grove.remote import MlxSshWorker
from grove.runtime import GroveRuntime
from grove.sandbox import LxdSandbox, SandboxPolicy
from grove.sleep import SleepCycle, SleepPolicy, forgetting_claim, oracle_free
from grove.store import GroveStore
from grove.verifiers import SandboxedPythonVerifier, VerifierRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
BASE_MODEL_COMMIT = "b3252a2f97102b1fb1571fec2c9b27219a8536be"
BASE_MODEL = f"{BASE_MODEL_ID}@{BASE_MODEL_COMMIT}"
BASE_MODEL_SOURCE = (
    "/Users/grove-worker/grove/cache/huggingface/hub/"
    "models--mlx-community--Qwen2.5-Coder-1.5B-Instruct-4bit/"
    f"snapshots/{BASE_MODEL_COMMIT}"
)
VERIFIER_SUITE_VERSION = "escaped-path-v2+python-core-v1"
# The suite a two-cycle run deploys: the same cycle-1 suites plus the second
# family's. Kept separate so a single-cycle run keeps its historical identity.
MULTI_CYCLE_VERIFIER_SUITE_VERSION = (
    "escaped-path-v2+python-core-v1+path-restructure-v1"
)
# The one second failure family the catalog implements. A third cycle needs a
# third authored family, so growth cycles beyond two are refused, not faked.
SECOND_CYCLE_FAMILY = "path_restructure"
SUPPORTED_GROWTH_CYCLES = (1, 2)
DECODING_CONFIG = {"temperature": 0.0, "max_tokens": 768}


def _self_repair_decoding_record(
    decoding: Mapping[str, Any] | None,
    *,
    correction_source: str,
) -> dict[str, Any]:
    """Record a declared value for optional self-repair decoding.

    Canonical corrections do not use self-repair decoding, but a bare ``None``
    is interpreted as an unresolved provenance value. The explicit declaration
    keeps that intentional absence distinct from a collector gap. Supplied
    decoding is copied so the report records the run's exact configuration
    without retaining a caller-owned mapping.
    """
    if decoding:
        return dict(decoding)
    if correction_source == "canonical":
        return {
            "applicable": False,
            "reason": "correction source 'canonical'; self-repair decoding not used",
        }
    if correction_source == "self-repair":
        return {
            "applicable": True,
            "declared": False,
            "reason": "self-repair ran under the default greedy decoding; no sampled decoding was declared",
        }
    raise ValueError(f"unsupported correction source: {correction_source!r}")


# Predeclared admission thresholds. Written out in full rather than left to
# library defaults so a reader can see exactly what this run promised before it
# ran, and so tightening a gate is a visible diff.
#
# This is deliberately stricter than the 2026-07-31 configuration, which had a
# two-task replay cohort and would now be rejected. Reproducing that run is not
# a goal; measuring the next one properly is.
REAL_CYCLE_POLICY = SleepPolicy(
    min_cluster_size=3,
    min_target_fix_rate=0.8,
    min_plasticity_gain=0.5,
    min_heldout_fix_rate=0.75,
    require_heldout_targets=True,
    measure_forced_replay=True,
    # Deliberately permissive: the 2026-07-31 adapter is router-shielded, not
    # intrinsically stable, and the metrics record says so. A predeclared
    # experiment that intends to claim intrinsic stability sets this to 0.0.
    max_forced_regression_rate=1.0,
    # A bare-base reference is what separates "this adapter forgets nothing"
    # from "the previous deployment forgot nothing". EXP-002 claims the former.
    measure_base_reference=True,
    # EXP-002 declares a 50-task prior-passing replay cohort. Enforcing it here
    # means a thin cohort rejects the candidate instead of producing a stability
    # claim nobody can support. The catalog cannot currently supply 50, so a
    # real cycle fails the capacity preflight until the cohort is authored.
    min_replay_examples=50,
    require_measured_replay=True,
    require_measured_route_recall=True,
    min_route_recall=0.5,
    max_route_false_positive_rate=0.0,
)


def _verifier_suite_manifest(catalog: Sequence[CodingTask]) -> list[dict[str, Any]]:
    """Hashable description of the hidden suites, without revealing the cases."""
    return [
        {
            "task_id": item.suite.task_id,
            "version": item.suite.version,
            "case_count": len(item.suite.cases),
            "cases_sha256": hashlib.sha256(
                json.dumps(
                    [[case.payload, case.expected] for case in item.suite.cases],
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest(),
        }
        for item in catalog
    ]


def _artifact_hash(directory: Path) -> tuple[str, int]:
    aggregate = hashlib.sha256()
    size = 0
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = str(path.relative_to(directory))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aggregate.update(relative.encode())
        aggregate.update(digest.encode())
        size += path.stat().st_size
    return aggregate.hexdigest(), size


def checkout_provenance(*, collect_worker: bool = False) -> dict[str, Any]:
    """Reproducibility metadata for this checkout, without running a cycle.

    Useful before and after an experiment: the two records should differ only in
    fields the run is allowed to change, and ``provenance_gaps`` names everything
    the collectors could not resolve on this host.

    Worker collection is opt-in because it opens an SSH round trip. When it is
    skipped the worker section is still emitted, marked unavailable, and counted
    as a gap -- never silently dropped.
    """
    trainer = MlxLoraTrainer(model=BASE_MODEL_SOURCE, base_model_revision=BASE_MODEL)
    record = collect_provenance(
        repo_root=REPO_ROOT,
        base_model=BASE_MODEL,
        verifier_suite_version=VERIFIER_SUITE_VERSION,
        training_config=trainer.training_config(),
        decoding_config=DECODING_CONFIG,
        verifier_suites=_verifier_suite_manifest(coding_catalog()),
        model_paths={"base": BASE_MODEL_SOURCE},
        sandbox_image=SandboxPolicy().image,
        worker=_worker_provenance(trainer.worker) if collect_worker else None,
        extra={"admission_policy": asdict(REAL_CYCLE_POLICY)},
    )
    return {"provenance": record, "provenance_gaps": missing_fields(record)}


def _worker_provenance(worker: MlxSshWorker) -> dict[str, Any]:
    """Worker identity, framework versions and model manifest; no credentials.

    The base-model path is passed to the worker so it can hash its own model
    files. A checkout revision and a framework version say nothing about which
    weights were loaded, and two arms of a paired experiment must be able to
    show they used the same ones.
    """
    return worker_metadata(
        lambda: worker.preflight(BASE_MODEL_SOURCE), host=worker.config.host
    )


class ExperimentSetupError(RuntimeError):
    """The declared experiment cannot be run as configured.

    Raised before the run spends anything. A setup that contradicts its own
    sealed spec, or that is arithmetically impossible, is a configuration bug,
    not an experimental result -- reporting it as a failed hypothesis would be
    a lie about what was tested.
    """


# Roles whose tasks the live capture actually attempts. Only these can ever
# supply a prior-passing replay negative, so they bound the replay cohort.
CAPTURED_ROLES = (DatasetRole.REGRESSION, DatasetRole.TRAIN)


def replay_capacity(
    catalog: Sequence[CodingTask], policy: SleepPolicy
) -> dict[str, Any]:
    """Can the declared replay cohort exist at all in this catalog?

    The live capture only attempts regression and training tasks, so the number
    of those is a hard ceiling on the prior-passing replay pool. In practice the
    pool is much smaller, because training tasks are training tasks precisely
    because the base fails them. If even the ceiling is below the declared
    minimum, every candidate is guaranteed to fail the denominator gate after
    training has already been paid for.
    """
    by_role = {
        role.value: sum(1 for item in catalog if item.role is role)
        for role in CAPTURED_ROLES
    }
    ceiling = sum(by_role.values())
    return {
        "declared_min_replay_examples": policy.min_replay_examples,
        "captured_task_ceiling": ceiling,
        "captured_by_role": by_role,
        "feasible": ceiling >= policy.min_replay_examples,
    }


def run_setup_manifest(
    catalog: Sequence[CodingTask],
    policy: SleepPolicy,
    *,
    correction_source: str,
    self_repair_attempts: int,
    compare_corrections: bool,
    database: Path,
    reset: bool,
    arm: str = "primary",
    self_repair_decoding: Mapping[str, Any] | None = None,
    growth_cycles: int = 1,
    second_family: str | None = None,
    verifier_suite_version: str = VERIFIER_SUITE_VERSION,
) -> dict[str, Any]:
    """What this run is actually configured to do, in checkable form.

    A sealed spec proves what was promised. This proves what was set up. The
    checker compares the two, so a spec declaring one correction source cannot
    be satisfied by a report produced under another.

    ``actual_training_failure_ids`` is left absent here and filled in after the
    live capture, because the catalog cohort is not the set of failures the run
    actually trained on. Pairing two arms on the catalog alone would call two
    runs identical when they trained different tasks.

    Decoding is recorded per purpose. Grading is sandbox execution and has no
    decoding; baseline, held-out, replay and future-probe generation are greedy
    by construction; only repair-attempt generation may sample, and only when
    ``self_repair_decoding`` declares it, with the base seed recorded here so
    the sealed spec can bind it.
    """
    cohorts = {
        role.value: sorted(item.task.id for item in catalog if item.role is role)
        for role in DatasetRole
        if any(item.role is role for item in catalog)
    }
    repair_decoding = (
        dict(self_repair_decoding) if self_repair_decoding is not None else None
    )
    return {
        "arm": arm,
        "correction_source": correction_source,
        "self_repair_attempts": self_repair_attempts,
        "self_repair_decoding": repair_decoding,
        "compare_corrections": compare_corrections,
        "min_replay_examples": policy.min_replay_examples,
        "base_model": BASE_MODEL,
        "verifier_suite_version": verifier_suite_version,
        "growth_cycles": growth_cycles,
        "second_family": second_family,
        "database": str(database),
        "database_reset": reset,
        "admission_policy": asdict(policy),
        "admission_policy_sha256": canonical_hash(asdict(policy)),
        "cohort_counts": {name: len(ids) for name, ids in cohorts.items()},
        "cohort_task_ids": cohorts,
        "cohort_manifest_sha256": canonical_hash(cohorts),
        "decoding_config": dict(DECODING_CONFIG),
        "decoding_by_purpose": {
            "baseline_evaluation": dict(DECODING_CONFIG),
            "heldout_evaluation": dict(DECODING_CONFIG),
            "replay_evaluation": dict(DECODING_CONFIG),
            "self_repair": (
                repair_decoding
                if repair_decoding is not None
                else dict(DECODING_CONFIG)
            ),
        },
    }


def record_actual_training_failures(
    setup: dict[str, Any], task_ids: Sequence[str]
) -> dict[str, Any]:
    """Record every training failure the live capture attempted.

    The capture set is the identity used to pair correction-source arms. It is
    not the trainable set: a proposal can fail verification and contribute no
    row to the MLX dataset. Keep both facts explicit so a report cannot call
    every attempted failure a trained example.
    """
    ordered = sorted(task_ids)
    digest = canonical_hash(ordered)
    setup["actual_training_failure_ids"] = ordered
    setup["actual_training_failure_set_sha256"] = digest
    setup["attempted_training_failure_ids"] = list(ordered)
    setup["attempted_training_failure_set_sha256"] = digest
    return setup


def record_second_cycle_failures(
    setup: dict[str, Any],
    *,
    attempted: Sequence[str],
    trained: Sequence[str],
) -> dict[str, Any]:
    """Record the second cycle's capture and trainable sets separately.

    Same discipline as cycle 1: the attempted set names what the live capture
    selected; the trained set names only failures whose canonical correction
    was verifier-accepted. Both are committed through ``run_setup_sha256``.
    """
    ordered_attempted = sorted(attempted)
    ordered_trained = sorted(trained)
    setup["second_cycle_attempted_training_failure_ids"] = ordered_attempted
    setup["second_cycle_attempted_training_failure_set_sha256"] = canonical_hash(
        ordered_attempted
    )
    setup["second_cycle_trained_failure_ids"] = ordered_trained
    setup["second_cycle_trained_failure_set_sha256"] = canonical_hash(
        ordered_trained
    )
    return setup



def record_trained_failures(
    setup: dict[str, Any], task_ids: Sequence[str]
) -> dict[str, Any]:
    """Record failures with accepted corrections that entered training data."""
    ordered = sorted(task_ids)
    setup["trained_failure_ids"] = ordered
    setup["trained_failure_set_sha256"] = canonical_hash(ordered)
    return setup


# Setup declarations split into what a program can check and what it cannot.
# A prose rule ("write replay tasks blind") is not verifiable by this code; a
# correction source is. Mixing them let a missing machine key be reported as
# merely "unchecked" and still pass.
SETUP_PROFILE_KEYS = ("machine", "prose")
# Keys no policy field backs any more. Accepting one and discarding it lets a
# spec declare a gate that never runs.
UNSUPPORTED_SETUP_KEYS = ("min_route_precision",)
SUPPORTED_CORRECTION_SOURCES = ("canonical", "self-repair")

# Comparisons understood by both the runner preflight and the standalone
# checker. A rule with another operator cannot be evaluated honestly.
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
# Mirrors ``scripts.check_experiment_spec.SUPPORTED_RULE_ARMS``. Omission means
# primary; an unknown or whitespace-padded value is a schema error, never a
# silent fallback to the primary report.
SUPPORTED_RULE_ARMS = frozenset({"primary", "control"})
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



def decision_rule_problems(spec: Mapping[str, Any]) -> list[str]:
    """Return structural errors that would make a decision rule unevaluable."""
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


def _correction_setting_problems(
    arm: str, machine: Mapping[str, Any]
) -> list[str]:
    """Validate correction settings declared by one arm profile."""
    problems: list[str] = []
    if "correction_source" in machine:
        source = machine["correction_source"]
        if source not in SUPPORTED_CORRECTION_SOURCES:
            problems.append(
                f"{arm} arm declares unknown correction source {source!r}; "
                f"expected one of {sorted(SUPPORTED_CORRECTION_SOURCES)}"
            )
    if "self_repair_attempts" in machine:
        attempts = machine["self_repair_attempts"]
        if isinstance(attempts, bool) or not isinstance(attempts, int):
            problems.append(
                f"{arm} arm self_repair_attempts must be an integer; got "
                f"{attempts!r}"
            )
        elif attempts < 1:
            problems.append(
                f"{arm} arm self_repair_attempts must be at least 1; got "
                f"{attempts}"
            )
    if "self_repair_decoding" in machine:
        problems.extend(
            self_repair_decoding_problems(
                machine["self_repair_decoding"],
                label=f"{arm} arm self_repair_decoding",
            )
        )
    return problems


def self_repair_decoding_problems(
    decoding: Any, *, label: str = "self_repair_decoding"
) -> list[str]:
    """Schema errors in a declared repair-attempt decoding config.

    Only repair-attempt generation may sample; a declaration must therefore say
    exactly what it samples with, and carry an integer base seed so every
    attempt seed derived from it is reproducible and reportable. Evaluation and
    grading decoding are not configurable here at all: they stay greedy by
    construction.
    """
    if decoding is None:
        return []
    if not isinstance(decoding, Mapping):
        return [f"{label} must be a mapping; got {decoding!r}"]
    problems: list[str] = []
    temperature = decoding.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        problems.append(f"{label}.temperature must be a number; got {temperature!r}")
    elif not 0.0 < float(temperature) <= 2.0:
        problems.append(
            f"{label}.temperature must be in (0.0, 2.0] -- a sampled repair "
            f"regime with temperature {temperature!r} is not sampled"
        )
    base_seed = decoding.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        problems.append(
            f"{label}.base_seed must be an integer so per-attempt seeds are "
            f"reproducible; got {base_seed!r}"
        )
    max_tokens = decoding.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        problems.append(
            f"{label}.max_tokens must be a positive integer; got {max_tokens!r}"
        )
    unknown = sorted(set(decoding) - {"temperature", "base_seed", "max_tokens"})
    if unknown:
        problems.append(
            f"{label} declares unsupported key(s) {unknown}; supported keys are "
            "temperature, base_seed and max_tokens"
        )
    return problems


def resolve_self_repair_configuration(
    machine: Mapping[str, Any],
    *,
    attempts: int | None = None,
    decoding: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Attempts and repair decoding for a run, driven by the arm declaration.

    An explicit argument wins; otherwise the sealed spec's own machine
    declaration configures the run, so raising the repair budget or turning on
    sampled attempts is a property of the spec under which the run launches,
    never an edit to policy defaults. With neither, the legacy configuration
    (3 greedy attempts) applies.
    """
    if attempts is None:
        attempts = machine.get("self_repair_attempts", 3)
    if decoding is None:
        decoding = machine.get("self_repair_decoding")
    return attempts, (dict(decoding) if decoding is not None else None)


def normalize_required_setup(declaration: Any) -> dict[str, dict[str, Any]]:
    """Split a ``required_setup`` declaration into machine and prose halves.

    The explicit form is ``{"machine": {...}, "prose": {...}}``. A flat legacy
    map is read as entirely machine-checkable, which is the strict reading: a
    key nobody split out is a key somebody expects to be enforced.
    """
    if declaration is None:
        return {"machine": {}, "prose": {}}
    if not isinstance(declaration, Mapping):
        raise TypeError("required_setup must be a mapping")
    keys = set(declaration)
    if keys and keys <= set(SETUP_PROFILE_KEYS):
        for key in keys:
            if not isinstance(declaration[key], Mapping):
                raise TypeError(f"required_setup.{key} must be a mapping")
        return {
            "machine": dict(declaration.get("machine", {})),
            "prose": dict(declaration.get("prose", {})),
        }
    return {"machine": dict(declaration), "prose": {}}


def setup_profiles(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The arm-specific setup declarations a spec offers.

    Preserve falsey declarations so a malformed profile is rejected by the
    schema check instead of becoming an empty, apparently valid profile.
    """
    profiles: dict[str, Any] = {"primary": spec.get("required_setup", {})}
    if "control_required_setup" in spec:
        profiles["control"] = spec["control_required_setup"]
    return profiles


def select_setup_profile(
    spec: Mapping[str, Any], *, correction_source: str, arm: str | None = None
) -> tuple[str, Any]:
    """Pick the arm profile this run must satisfy.

    A paired spec declares one setup per arm. Validating the control run against
    the primary declaration is how the canonical EXP-003 arm became unlaunchable:
    its own spec demanded ``correction_source: self-repair``. An explicit ``arm``
    wins; otherwise the profile whose declared correction source matches the run
    is selected, and an unmatched run falls back to the primary profile so the
    contradiction is reported rather than hidden.
    """
    profiles = setup_profiles(spec)
    if arm is not None:
        if arm not in profiles:
            raise ExperimentSetupError(
                f"spec declares no {arm!r} arm setup profile; "
                f"available profiles: {sorted(profiles)}"
            )
        return arm, profiles[arm]
    matching = [
        name
        for name, declaration in profiles.items()
        if normalize_required_setup(declaration)["machine"].get("correction_source")
        == correction_source
    ]
    if len(matching) == 1:
        return matching[0], profiles[matching[0]]
    return "primary", profiles["primary"]


def validate_required_setup(
    spec: Mapping[str, Any],
    setup: Mapping[str, Any],
    *,
    declaration: Any | None = None,
) -> dict[str, Any]:
    """Compare a declared ``required_setup`` against the actual setup.

    A declared machine key that the setup manifest does not carry is a mismatch,
    not an unchecked footnote. The old behaviour listed it under
    ``unchecked_keys`` and still returned satisfied, so deleting
    ``correction_source`` from a spec turned the strongest binding in the
    protocol into a no-op. Prose keys are reported separately and never counted
    as conformance.
    """
    if declaration is None:
        declaration = spec.get("required_setup")
    split = normalize_required_setup(declaration)
    machine, prose = split["machine"], split["prose"]
    unsupported = sorted(
        key
        for key in (*machine, *prose)
        if key in UNSUPPORTED_SETUP_KEYS
    )
    mismatches: list[dict[str, Any]] = []
    checked: dict[str, Any] = {}
    missing: list[str] = []
    for key, expected in machine.items():
        if key not in setup:
            missing.append(key)
            mismatches.append({"key": key, "declared": expected, "observed": None})
            continue
        observed = setup[key]
        checked[key] = observed
        if observed != expected:
            mismatches.append({"key": key, "declared": expected, "observed": observed})
    return {
        "declared_keys": sorted(machine),
        "checked": checked,
        "missing_machine_keys": sorted(missing),
        # Declarations a program cannot verify. Listed so nobody reads silence
        # as enforcement, and never counted towards conformance.
        "unchecked_prose_keys": sorted(prose),
        "unsupported_keys": unsupported,
        "mismatches": mismatches,
        "satisfied": not mismatches and not unsupported,
    }


SPEC_HASH_FIELD = "spec_sha256"


def load_sealed_spec(spec_path: str | Path) -> dict[str, Any]:
    """Load a predeclared spec and verify its seal before any model work runs.

    Binding happens here, not at analysis time. The run records the digest it
    was launched under, so re-sealing an edited spec afterwards cannot make an
    old report look compliant -- the recorded digest simply stops matching.
    """
    path = Path(spec_path)
    spec = json.loads(path.read_text())
    recorded = spec.get(SPEC_HASH_FIELD)
    if not recorded:
        raise ValueError(
            f"experiment spec {path} is not sealed; "
            "run scripts/check_experiment_spec.py --seal before the cycle"
        )
    computed = canonical_hash(
        {key: value for key, value in spec.items() if key != SPEC_HASH_FIELD}
    )
    if recorded != computed:
        raise ValueError(
            f"experiment spec {path} was altered after sealing: "
            f"recorded {recorded}, computed {computed}"
        )
    return spec


def preflight_experiment(
    spec: Mapping[str, Any],
    *,
    correction_source: str,
    self_repair_attempts: Any,
    compare_corrections: bool,
    arm: str | None = None,
    self_repair_decoding: Any = None,
    growth_cycles: Any = 1,
) -> dict[str, Any]:
    """Reject an inadmissible run before it costs anything.

    Every check here is pure: no database, no sandbox, no SSH, no inference.
    An unknown correction source used to be caught only after the sandbox was
    built, the baseline evaluated and the live failures captured, which meant a
    typo bought a full capture pass before failing.
    """
    problems: list[str] = []
    if correction_source not in SUPPORTED_CORRECTION_SOURCES:
        problems.append(
            f"unknown correction source {correction_source!r}; expected one of "
            f"{sorted(SUPPORTED_CORRECTION_SOURCES)}"
        )
    if isinstance(self_repair_attempts, bool) or not isinstance(
        self_repair_attempts, int
    ):
        problems.append(
            f"self_repair_attempts must be an integer; got "
            f"{self_repair_attempts!r}"
        )
    elif self_repair_attempts < 1:
        problems.append(
            f"self_repair_attempts must be at least 1; got {self_repair_attempts}"
        )
    problems.extend(self_repair_decoding_problems(self_repair_decoding))
    problems.extend(growth_cycle_problems(growth_cycles))
    if compare_corrections and len(SUPPORTED_CORRECTION_SOURCES) < 2:
        problems.append("compare_corrections needs at least two correction sources")

    profiles: dict[str, Any] = {}
    if spec:
        problems.extend(_spec_substance_problems(spec))
        problems.extend(
            f"decision rule schema: {problem}"
            for problem in decision_rule_problems(spec)
        )
        profiles = setup_profiles(spec)
        if arm is not None and arm not in profiles:
            problems.append(
                f"spec declares no {arm!r} arm setup profile; "
                f"available profiles: {sorted(profiles)}"
            )
        for name, declaration in profiles.items():
            try:
                split = normalize_required_setup(declaration)
            except (TypeError, ValueError) as error:
                problems.append(f"{name} arm setup schema is invalid: {error}")
                continue
            unsupported = sorted(
                key for key in (*split["machine"], *split["prose"])
                if key in UNSUPPORTED_SETUP_KEYS
            )
            problems.extend(_correction_setting_problems(name, split["machine"]))
            if unsupported:
                problems.append(
                    f"{name} arm setup declares unsupported key(s) {unsupported}; "
                    "no policy field backs them, so the declaration would be "
                    "accepted and discarded"
                )
        problems.extend(_timing_claim_problems(spec))
        problems.extend(_control_profile_problems(spec))
    if problems:
        raise ExperimentSetupError(
            "experiment preflight rejected this run: " + "; ".join(problems)
        )
    return {
        "correction_source": correction_source,
        "self_repair_attempts": self_repair_attempts,
        "self_repair_decoding": (
            dict(self_repair_decoding) if self_repair_decoding else None
        ),
        "compare_corrections": compare_corrections,
        "growth_cycles": growth_cycles,
        "available_arms": sorted(profiles),
        "control_requirement": control_requirement(spec),
    }


def growth_cycle_problems(growth_cycles: Any) -> list[str]:
    """Schema errors in a declared growth-cycle count.

    The catalog implements exactly one second failure family, so a run may
    declare one or two cycles. A third cycle without a third authored family
    would silently rerun the second one, which is not a third cycle.
    """
    if isinstance(growth_cycles, bool) or not isinstance(growth_cycles, int):
        return [f"growth_cycles must be an integer; got {growth_cycles!r}"]
    if growth_cycles not in SUPPORTED_GROWTH_CYCLES:
        return [
            (
                f"growth_cycles must be one of {sorted(SUPPORTED_GROWTH_CYCLES)}; "
                f"got {growth_cycles} -- the catalog carries exactly one second "
                "failure family"
            )
        ]
    return []


def resolve_growth_cycles(
    machine: Mapping[str, Any], *, growth_cycles: int | None = None
) -> int:
    """Growth-cycle count for a run, driven by the arm declaration.

    Mirrors ``resolve_self_repair_configuration``: an explicit argument wins;
    otherwise the sealed spec's machine block configures the run, so a second
    cycle is a property of the spec the run launches under, never an edited
    default. With neither, one cycle -- the historical behaviour.
    """
    if growth_cycles is not None:
        return growth_cycles
    declared = machine.get("growth_cycles", 1)
    return declared


def control_requirement(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Does this spec need a control arm? Ask the rules, not a flag.

    Mirrors ``scripts/check_experiment_spec.control_requirement``. A rule needs
    a control when it computes a delta, names a ``control_path``, pairs entries
    with ``pair_on``, or reads the control arm. ``requires_control_report`` was
    the only switch, so removing it left delta rules running against a control
    that was never identity-checked.
    """
    reasons: list[dict[str, Any]] = []
    for rule in spec.get("decision_rules") or []:
        if not isinstance(rule, Mapping):
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
        "flag_contradicts_rules": bool(reasons) and declared is False,
        "flag_overclaims": bool(declared) and not reasons,
    }


def _control_profile_problems(spec: Mapping[str, Any]) -> list[str]:
    """A paired spec must say what its control arm was configured to do.

    Deleting ``control_required_setup`` made the control conformance check
    collapse to "spec declares no required_setup", which read as conformant: a
    control arm running with ``compare_corrections: false`` passed. A spec whose
    rules need a control and that declares nothing about it has not designed a
    control.
    """
    need = control_requirement(spec)
    problems: list[str] = []
    if need["flag_contradicts_rules"]:
        problems.append(
            "spec sets requires_control_report false while its own rules need a "
            f"control arm: {need['rules']}"
        )
    if not need["required"]:
        return problems
    declaration = spec.get("control_required_setup")
    try:
        machine = normalize_required_setup(declaration)["machine"]
    except (TypeError, ValueError) as error:
        return [*problems, f"control arm setup schema is invalid: {error}"]
    if not machine:
        problems.append(
            "the sealed rules need a control arm but the spec declares no "
            "machine-checkable control_required_setup, so nothing about the "
            "control run would be checked"
        )
    return problems


def _spec_substance_problems(spec: Mapping[str, Any]) -> list[str]:
    """The same emptiness checks the analysis-time checker applies."""
    problems: list[str] = []
    if not spec.get("decision_rules"):
        problems.append("sealed spec declares no decision_rules")
    hypotheses = spec.get("hypotheses") or []
    if not hypotheses:
        problems.append("sealed spec declares no hypotheses")
    if any(not str(item.get("falsified_if", "")).strip() for item in hypotheses):
        problems.append("sealed spec has a hypothesis with no falsification condition")
    return problems


def _timing_claim_problems(spec: Mapping[str, Any]) -> list[str]:
    """Refuse a preregistration claim this repository cannot verify.

    A self-reported ``{"type": "rfc3161", "timestamp": "0000-not-a-time"}``
    used to satisfy the check. Verifying an RFC 3161 token needs its DER bytes,
    the TSA signature, a trusted certificate and a ``genTime`` parsed out of the
    token; a signed tag, a Rekor entry and an OSF registration need comparable
    artifacts. None of those verifiers exists here, so the honest answer is to
    refuse the claim rather than accept a typed field as proof.
    """
    attestation = spec.get("timing_attestation")
    claims = bool(spec.get("declared_before_run") or spec.get("preregistered"))
    if attestation:
        return [
            (
                "spec supplies a timing_attestation, but no verifier for it "
                "exists in this repository. A type and a timestamp anyone can "
                "type are not evidence; declare seal_self_consistent with a "
                "null timing_attestation until a real verifier lands"
            )
        ]
    if claims:
        return [
            (
                "spec claims preregistration timing with no attestation at "
                "all. A spec digest proves self-consistency, not when the spec "
                "was written; declare seal_self_consistent instead"
            )
        ]
    return []


# Identity paths a run can resolve before it spends anything: the control
# host's own checkout, and one worker round trip. Everything else -- the
# sandbox fingerprint, the actual failure set -- only exists once the run has
# started, so it is checked at analysis time instead.
PREFLIGHTABLE_IDENTITY_PREFIXES = ("provenance.source.", "provenance.worker.")


def preflight_required_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the identity a paired spec declares it cannot do without.

    ``required_resolved_identity`` is how EXP-003 says that an unresolved worker
    model manifest makes its A/B comparison meaningless. Discovering that after
    two MLX training runs is the expensive way to learn it.

    Only paths this host can resolve before the run are checked; the rest are
    reported as deferred so nobody reads silence as a pass.
    """
    required = [str(item) for item in spec.get("required_resolved_identity") or ()]
    if not required:
        return {"required": [], "checked": {}, "unresolved": [], "deferred": []}
    preflightable = [
        path
        for path in required
        if path.startswith(PREFLIGHTABLE_IDENTITY_PREFIXES)
    ]
    deferred = [path for path in required if path not in preflightable]
    document: dict[str, Any] = {"provenance": {}}
    if any(path.startswith("provenance.source.") for path in preflightable):
        document["provenance"]["source"] = git_revision(REPO_ROOT)
    if any(path.startswith("provenance.worker.") for path in preflightable):
        trainer = MlxLoraTrainer(
            model=BASE_MODEL_SOURCE, base_model_revision=BASE_MODEL
        )
        document["provenance"]["worker"] = _worker_provenance(trainer.worker)
    checked: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []
    for path in preflightable:
        value = resolve_report_path(document, path)
        rendered = None if value is MISSING_INPUT else value
        checked[path] = rendered
        if value is MISSING_INPUT or _is_unresolved_identity(value, path):
            unresolved.append({"path": path, "value": rendered})
    return {
        "required": required,
        "checked": checked,
        "unresolved": unresolved,
        # Named, not skipped: these are checked when the report is graded.
        "deferred": deferred,
    }


def _is_unresolved_identity(value: Any, path: str | None = None) -> bool:
    """Absent, null, blank, unavailable -- or a digest field holding a non-digest.

    A worker that answers ``model_manifest_sha256: false`` has reported no model
    identity. Preflight has to say so here, before two MLX training runs are
    spent discovering it at grading time.
    """
    if value is None or (
        isinstance(value, str)
        and (not value.strip() or value.strip().startswith("unavailable:"))
    ):
        return True
    return is_digest_field(path) and not is_sha256_hex(value)


def select_training_proposal(
    task: Any,
    failed_response: str,
    source: Any,
    verify: Any,
    *,
    reused: Mapping[str, Any] | None = None,
) -> tuple[Any, bool]:
    """The correction to train on, preferring the one already measured.

    ``compare_correction_sources`` used to generate proposals and throw them
    away; the training loop then asked the same source again. Self-repair is
    nondeterministic and gets several attempts, so the verified yield in the
    report described corrections that were never trained on. Reusing the exact
    proposal makes the reported comparison a description of the training run.

    Returns the proposal and whether it came from the comparison, so the cost
    of each phase can be counted separately instead of double-counted.
    """
    if reused:
        existing = reused.get(task.id)
        if existing is not None:
            return existing, True
    return source.propose(task, failed_response, verify), False


def _expert_family(expert: Expert, families: Sequence[str]) -> str | None:
    """Which declared failure family an active expert serves, if exactly one.

    The clusterer builds an expert's routing profile from its cluster's task
    tags, so the family name is carried in the profile. Ambiguity is reported
    as ``None`` rather than guessed.
    """
    tags = set(expert.routing_profile.get("tags", []))
    owned = [family for family in families if family in tags]
    return owned[0] if len(owned) == 1 else None


def coexistence_checkpoint(
    runtime: GroveRuntime,
    store: GroveStore,
    *,
    capability: Mapping[str, Any],
    replay: Sequence[Task],
    heldout_by_family: Mapping[str, Sequence[Task]],
    cycle_one_training_ids: Sequence[str],
) -> dict[str, Any]:
    """Multi-expert measurements for one checkpoint of a multi-cycle run.

    Everything here measures the *deployed pool*, not a single candidate:

    - routed replay over the prior-passing suite (``replay`` holds tasks whose
      latest recorded attempt passed, so every task is prior-passing by
      construction and the denominator is the whole suite);
    - route recall, precision and false-positive rate across every deployed
      expert at once, on oracle-free copies -- positives are held-out tasks the
      owning expert passes when forced on, negatives are prior-passing replay
      tasks belonging to no deployed expert's family;
    - per-family interference probes: each expert's held-out cohort re-run
      routed through the full pool, and forced-on, so a later expert stealing
      an earlier expert's traffic is a measured drop, not a hidden one;
    - per-expert forced-on replay against the bare base, so ``forgetting_claim``
      stays honest per expert;
    - capability and parameter growth copied from the checkpoint benchmark.

    Denominators are recorded beside every rate, and an unmeasurable rate is
    ``None``, never zero.
    """
    active = store.routable_experts()
    families = sorted(heldout_by_family)
    expert_by_family: dict[str, Expert | None] = {
        family: next(
            (
                expert
                for expert in active
                if _expert_family(expert, families) == family
            ),
            None,
        )
        for family in families
    }
    owned_families = {
        family for family, expert in expert_by_family.items() if expert is not None
    }
    replay = list(replay)

    routed_results = runtime.run(replay, record=False) if replay else []
    regressed = [
        result.task.id
        for result in routed_results
        if not result.verification.passed
    ]
    replay_regression_rate = len(regressed) / len(replay) if replay else None
    base_results = (
        runtime.run(replay, experts=[], record=False) if replay and active else []
    )
    base_pass_ids = {
        result.task.id for result in base_results if result.verification.passed
    }

    router = runtime.router
    negatives = [
        task
        for task in replay
        if str(task.metadata.get("failure_type")) not in owned_families
    ]
    negatives_routed = sum(
        router.route(oracle_free(task), active).expert_id is not None
        for task in negatives
    )

    per_family: dict[str, Any] = {}
    total_positives = 0
    total_correctly_routed = 0
    total_positives_routed_to_any = 0
    for family in families:
        expert = expert_by_family[family]
        heldout = list(heldout_by_family[family])
        entry: dict[str, Any] = {
            "expert_id": expert.id if expert else None,
            "heldout_examples": len(heldout),
            "heldout_forced_rate": None,
            "heldout_routed_rate": None,
            "route_positives": 0,
            "route_positives_routed_to_own_expert": 0,
            "route_recall": None,
            "forced_replay_denominator": 0,
            "forced_regression_rate": None,
            "forced_regression_task_ids": [],
            "forced_regression_reference": "base_no_experts",
            "forgetting_claim": "unmeasured",
        }
        if expert is not None and heldout:
            forced_heldout = runtime.run(
                heldout, force_expert=expert, record=False
            )
            entry["heldout_forced_rate"] = sum(
                result.verification.passed for result in forced_heldout
            ) / len(heldout)
            routed_heldout = runtime.run(heldout, record=False)
            entry["heldout_routed_rate"] = sum(
                result.verification.passed for result in routed_heldout
            ) / len(heldout)
            positives = [
                result.task
                for result in forced_heldout
                if result.verification.passed
            ]
            decisions = [
                router.route(oracle_free(task), active) for task in positives
            ]
            correct = sum(
                decision.expert_id == expert.id for decision in decisions
            )
            routed_to_any = sum(
                decision.expert_id is not None for decision in decisions
            )
            entry["route_positives"] = len(positives)
            entry["route_positives_routed_to_own_expert"] = correct
            entry["route_recall"] = (
                correct / len(positives) if positives else None
            )
            total_positives += len(positives)
            total_correctly_routed += correct
            total_positives_routed_to_any += routed_to_any
        if expert is not None and replay:
            forced = runtime.run(replay, force_expert=expert, record=False)
            forced_regressed = [
                result.task.id
                for result in forced
                if result.task.id in base_pass_ids
                and not result.verification.passed
            ]
            entry["forced_replay_denominator"] = len(base_pass_ids)
            entry["forced_regression_rate"] = (
                len(forced_regressed) / len(base_pass_ids)
                if base_pass_ids
                else None
            )
            entry["forced_regression_task_ids"] = sorted(forced_regressed)
            entry["forgetting_claim"] = forgetting_claim(
                forced_regression_rate=entry["forced_regression_rate"],
                forced_regression_reference="base_no_experts",
                routed_regression_rate=replay_regression_rate,
            )
        per_family[family] = entry

    routed_total = total_positives_routed_to_any + negatives_routed
    replay_ids = {task.id for task in replay}
    return {
        "active_experts": len(active),
        "active_expert_ids": sorted(expert.id for expert in active),
        "added_parameters": capability.get("added_parameters"),
        "capability": capability.get("capability"),
        "evaluation_id": capability.get("evaluation_id"),
        "replay_examples": len(replay),
        "replay_pass_rate": (
            1 - len(regressed) / len(replay) if replay else None
        ),
        "replay_regression_rate": replay_regression_rate,
        "replay_regressed_task_ids": sorted(regressed),
        "cycle_1_training_targets_in_replay": len(
            replay_ids & set(cycle_one_training_ids)
        ),
        "route_positive_source": "heldout_forced_pass_oracle_free",
        "route_probe_metadata": "oracle_free",
        "route_positives": total_positives,
        "route_negatives": len(negatives),
        "route_negatives_routed": negatives_routed,
        "route_recall": (
            total_correctly_routed / total_positives if total_positives else None
        ),
        # Deliberately cohort-dependent, as every precision over mixed cohorts
        # is; the denominators are recorded beside it so a reader can see what
        # moved. The numerator counts only positives routed to their *own*
        # family's expert, so cross-expert confusion lowers it.
        "route_precision": (
            total_correctly_routed / routed_total if routed_total else None
        ),
        "route_precision_cohort_dependent": True,
        "route_false_positive_rate": (
            negatives_routed / len(negatives) if negatives else None
        ),
        "per_family": per_family,
    }


def run_first_real_cycle(
    database: str | Path = "/srv/storage/grove/grove-real.db",
    report_path: str | Path = "/srv/storage/grove/evaluations/first-real-cycle.json",
    *,
    reset: bool = False,
    correction_source: str = "canonical",
    self_repair_attempts: int | None = None,
    compare_corrections: bool = False,
    spec_path: str | Path | None = None,
    arm: str | None = None,
    self_repair_decoding: Mapping[str, Any] | None = None,
    growth_cycles: int | None = None,
) -> dict[str, Any]:
    """Run one real MLX growth cycle.

    ``correction_source`` selects where training targets come from. ``canonical``
    reproduces the 2026-07-31 configuration (human-written reference solutions);
    ``self-repair`` makes the model write its own corrections, still admitted only
    when the hidden verifier passes them. ``compare_corrections`` additionally
    records both sources' verified yield over the same failures, which is the
    cheap half of the audit's self-versus-human question.

    ``spec_path`` binds the run to a sealed predeclared spec. The seal is
    verified before any sandbox or model work starts, and the digest is written
    into the report so the analysis step can prove which version was run.

    ``arm`` names which setup profile of a paired spec this run must satisfy.
    A paired spec declares one setup per arm; without this the control arm is
    graded against the primary arm's declaration and cannot launch at all.

    ``self_repair_attempts`` and ``self_repair_decoding`` default to whatever
    the selected arm's sealed ``required_setup`` machine block declares, so an
    experiment raises the repair budget or samples repair attempts through its
    spec, not by editing policy defaults. Explicit arguments win; with neither,
    the legacy 3 greedy attempts apply. Evaluation, baseline, held-out and
    replay decoding is greedy temperature 0.0 regardless.

    ``growth_cycles`` selects single-cycle (the historical behaviour) or the
    two-cycle coexistence mode. Like the repair settings it defaults to the
    selected arm's sealed machine declaration. In two-cycle mode the run
    performs the complete first cycle, then -- against the same store, with the
    first expert still admitted, deployed and routable -- captures the second
    failure family, trains and gates a second candidate through the identical
    admission policy, and records multi-expert coexistence measurements at
    every checkpoint. One invocation, one database, no reset between cycles.
    """
    database = Path(database)
    report_path = Path(report_path)
    run_started_at = utc_now()
    # Everything in the setup block runs before the first byte is written. A
    # run that cannot satisfy its declaration or its infrastructure preflight
    # must cost nothing and must not reset an existing evidence database.
    experiment_spec: dict[str, Any] | None = None
    spec: dict[str, Any] = {}
    if spec_path is not None:
        spec = load_sealed_spec(spec_path)
        experiment_spec = {
            "spec_id": spec.get("spec_id"),
            SPEC_HASH_FIELD: spec[SPEC_HASH_FIELD],
            "path": str(Path(spec_path)),
        }
    selected_arm, arm_declaration = select_setup_profile(
        spec, correction_source=correction_source, arm=arm
    )
    try:
        arm_machine = normalize_required_setup(arm_declaration)["machine"]
    except (TypeError, ValueError):
        # A malformed profile is reported by the preflight below, with the
        # other schema problems, rather than as a bare TypeError here.
        arm_machine = {}
    self_repair_attempts, self_repair_decoding = resolve_self_repair_configuration(
        arm_machine,
        attempts=self_repair_attempts,
        decoding=self_repair_decoding,
    )
    growth_cycles = resolve_growth_cycles(
        arm_machine, growth_cycles=growth_cycles
    )
    preflight = preflight_experiment(
        spec,
        correction_source=correction_source,
        self_repair_attempts=self_repair_attempts,
        compare_corrections=compare_corrections,
        arm=arm,
        self_repair_decoding=self_repair_decoding,
        growth_cycles=growth_cycles,
    )
    catalog = coding_catalog()
    second_catalog = second_cycle_catalog() if growth_cycles >= 2 else []
    full_catalog = [*catalog, *second_catalog]
    suite_version = (
        MULTI_CYCLE_VERIFIER_SUITE_VERSION
        if growth_cycles >= 2
        else VERIFIER_SUITE_VERSION
    )
    setup = run_setup_manifest(
        full_catalog,
        REAL_CYCLE_POLICY,
        correction_source=correction_source,
        self_repair_attempts=self_repair_attempts,
        compare_corrections=compare_corrections,
        database=database,
        reset=reset,
        arm=selected_arm,
        self_repair_decoding=self_repair_decoding,
        growth_cycles=growth_cycles,
        second_family=SECOND_CYCLE_FAMILY if growth_cycles >= 2 else None,
        verifier_suite_version=suite_version,
    )
    setup_check = validate_required_setup(
        spec, setup, declaration=arm_declaration
    )
    setup_check["arm"] = selected_arm
    if not setup_check["satisfied"]:
        detail = "; ".join(
            f"{item['key']} declared {item['declared']!r} but the run uses "
            f"{item['observed']!r}"
            for item in setup_check["mismatches"]
        )
        if setup_check["missing_machine_keys"]:
            detail += (
                "; the run setup manifest does not record "
                f"{setup_check['missing_machine_keys']}, so those declarations "
                "cannot be checked at all"
            )
        if setup_check["unsupported_keys"]:
            detail += (
                f"; unsupported declared key(s) {setup_check['unsupported_keys']}"
            )
        raise ExperimentSetupError(
            f"run setup contradicts the sealed spec ({selected_arm} arm): {detail}"
        )
    capacity = replay_capacity(catalog, REAL_CYCLE_POLICY)
    if not capacity["feasible"]:
        raise ExperimentSetupError(
            "declared replay cohort is impossible for this catalog: the policy "
            f"requires {capacity['declared_min_replay_examples']} prior-passing "
            "replay tasks but the live capture attempts at most "
            f"{capacity['captured_task_ceiling']} tasks "
            f"({capacity['captured_by_role']}). Author the predeclared replay "
            "cohort, or reseal a spec and policy that declare a smaller pilot."
        )
    # Identity the sealed spec says a paired comparison cannot do without. It is
    # collected here, from a worker round trip and a local git read, because
    # discovering after training that the two arms cannot be compared wastes the
    # whole run. Nothing has been written or launched at this point.
    identity_check = preflight_required_identity(spec)
    if identity_check["unresolved"]:
        raise ExperimentSetupError(
            "required run identity is unavailable, so this run could not be "
            "paired even if it succeeded: "
            + "; ".join(
                f"{item['path']} is required for paired identity but is "
                f"unavailable ({item['value']})"
                for item in identity_check["unresolved"]
            )
        )

    # The sandbox check is an infrastructure preflight, not run work. Keep it
    # ahead of reset so an unavailable image or profile cannot destroy the
    # evidence database that the caller meant to preserve.
    sandbox = LxdSandbox()
    try:
        sandbox_preflight = sandbox.preflight()
    except Exception as error:
        raise ExperimentSetupError(
            f"sandbox preflight failed before database reset: {error}"
        ) from error

    if reset and database.exists():
        database.unlink()
    by_id = {item.task.id: item for item in full_catalog}
    regression = [item for item in catalog if item.role is DatasetRole.REGRESSION]
    training = [item for item in catalog if item.role is DatasetRole.TRAIN]
    targets = [item for item in catalog if item.role is DatasetRole.TARGET]
    future = [item for item in catalog if item.role is DatasetRole.FUTURE]
    second_training = [
        item for item in second_catalog if item.role is DatasetRole.TRAIN
    ]
    second_targets = [
        item for item in second_catalog if item.role is DatasetRole.TARGET
    ]
    verifiers = VerifierRegistry()
    verifiers.register(
        "sandboxed_python",
        SandboxedPythonVerifier(sandbox, [item.suite for item in full_catalog]),
    )
    with GroveStore(database) as store:
        if store.summary()["attempts"]:
            raise RuntimeError(
                "real cycle database is not empty; use reset for a new experiment"
            )
        baseline_deployment = store.publish_deployment(
            base_model_revision=BASE_MODEL,
            expert_ids=(),
            router_version="profile-router-v1",
            verifier_suite_version=suite_version,
            decoding_config={"temperature": 0.0, "max_tokens": 768},
            reason="frozen baseline",
        )
        for item in full_catalog:
            store.save_task(item.task)
            if item.role in {
                DatasetRole.REGRESSION,
                DatasetRole.TARGET,
                DatasetRole.FUTURE,
            }:
                store.assign_dataset_role(
                    task_id=item.task.id,
                    role=item.role,
                    content=item.task.prompt,
                )

        backend = MlxRemoteBackend(model=BASE_MODEL_SOURCE, max_tokens=768)
        runtime = GroveRuntime(store, backend, verifiers=verifiers)
        benchmark = LongitudinalBenchmark(store, runtime)
        cohorts = {
            "regression_known_skills": [item.task for item in regression],
            "plasticity_escaped_path": [item.task for item in targets],
        }
        if growth_cycles >= 2:
            # Fixed from the baseline on, so capability is comparable at every
            # checkpoint: a cohort that grows between checkpoints cannot
            # support a monotonicity claim.
            cohorts["plasticity_path_restructure"] = [
                item.task for item in second_targets
            ]
        heldout_by_family = {
            "escaped_path": [item.task for item in targets],
        }
        if growth_cycles >= 2:
            heldout_by_family[SECOND_CYCLE_FAMILY] = [
                item.task for item in second_targets
            ]
        baseline = benchmark.evaluate(cohorts, label="frozen_baseline")
        cycle_one_training_ids = [item.task.id for item in training]
        coexistence_baseline = (
            coexistence_checkpoint(
                runtime,
                store,
                capability=baseline,
                # No task has a recorded prior-passing attempt yet, so the
                # replay suite is honestly empty and its rates are None.
                replay=[],
                heldout_by_family=heldout_by_family,
                cycle_one_training_ids=cycle_one_training_ids,
            )
            if growth_cycles >= 2
            else None
        )

        live_results = runtime.run(
            [item.task for item in [*regression, *training]],
            run_id="initial_failure_capture",
            record=True,
        )
        live_summary = {
            "tasks": len(live_results),
            "passed": sum(result.verification.passed for result in live_results),
            "failed": sum(not result.verification.passed for result in live_results),
            "results": {
                result.task.id: {
                    "passed": result.verification.passed,
                    "score": result.verification.score,
                    "reason": result.verification.reason,
                }
                for result in live_results
            },
        }

        reference_source = CanonicalReferenceSource(
            {item.task.id: item.reference_solution for item in full_catalog}
        )
        repair_decoding: dict[str, Any] | None = None
        repair_seed: int | None = None
        if self_repair_decoding:
            repair_seed = int(self_repair_decoding["base_seed"])
            repair_decoding = {
                key: value
                for key, value in self_repair_decoding.items()
                if key in ("temperature", "max_tokens")
            }
        self_source = SelfRepairSource(
            backend,
            attempts=self_repair_attempts,
            decoding=repair_decoding,
            seed=repair_seed,
        )
        available_sources = {
            "canonical": reference_source,
            "self-repair": self_source,
        }
        # The correction source was validated in the zero-cost preflight; this
        # only turns the validated name into the object.
        primary_source = available_sources[correction_source]

        train_failures = [
            (failure, by_id[failure.task.id])
            for failure in store.unresolved_failures()
            if by_id[failure.task.id].role is DatasetRole.TRAIN
        ]
        record_actual_training_failures(
            setup, [failure.task.id for failure, _ in train_failures]
        )
        correction_comparison = None
        # Proposals from the comparison are reused for training. Regenerating
        # them meant the reported verified yield described corrections nobody
        # trained on, which is not a comparison of anything.
        reused_proposals: dict[str, Any] = {}
        if compare_corrections:
            pairs = [
                (failure.task, failure.response) for failure, _ in train_failures
            ]
            proposals_by_source = correction_proposals(
                pairs, [reference_source, self_source], verifiers.verify
            )
            correction_comparison = summarize_correction_sources(
                proposals_by_source, pairs
            )
            reused_proposals = {
                item.task_id: item
                for item in proposals_by_source[primary_source.name]
            }
            correction_comparison["training_proposal_reuse"] = {
                "enabled": True,
                "source": primary_source.name,
                "reused_task_ids": sorted(reused_proposals),
            }

        correction_ids = []
        training_generation_calls = 0
        training_proposals: list[dict[str, Any]] = []
        trained_failure_ids: list[str] = []
        for failure, item in train_failures:
            proposal, was_reused = select_training_proposal(
                failure.task,
                failure.response,
                primary_source,
                verifiers.verify,
                reused=reused_proposals,
            )
            reused = proposal if was_reused else None
            if not was_reused:
                training_generation_calls += proposal.generation_calls
            response_sha256 = (
                hashlib.sha256(proposal.response.encode()).hexdigest()
                if proposal.response is not None
                else None
            )
            training_proposals.append(
                {
                    "task_id": proposal.task_id,
                    "source": proposal.source,
                    "reused_from_comparison": reused is not None,
                    "accepted": proposal.accepted,
                    "attempts": proposal.attempts,
                    "generation_calls": proposal.generation_calls,
                    "response_sha256": response_sha256,
                    "decoding": proposal.provenance.get("decoding"),
                    "attempt_seeds": proposal.provenance.get("attempt_seeds"),
                }
            )
            if proposal.response is None or proposal.verification is None:
                continue
            correction = store.record_correction(
                failure_id=failure.id,
                response=proposal.response,
                source=proposal.source,
                verification=proposal.verification,
                provenance={
                    "suite_version": item.suite.version,
                    "attempts": proposal.attempts,
                    "correction_sha256": response_sha256,
                    "reused_from_comparison": reused is not None,
                    **proposal.provenance,
                },
            )
            if correction.accepted:
                correction_ids.append(correction.id)
                trained_failure_ids.append(item.task.id)
                store.assign_dataset_role(
                    task_id=item.task.id,
                    role=DatasetRole.TRAIN,
                    # Split identity is the model input, not the serialized
                    # training row; this makes an identical prompt collide
                    # across train/target/regression roles as intended.
                    content=item.task.prompt,
                    failure_id=failure.id,
                    correction_id=correction.id,
                )
        record_trained_failures(setup, trained_failure_ids)
        if correction_comparison is not None:
            correction_comparison["training_generation_calls"] = (
                training_generation_calls
            )

        trainer = MlxLoraTrainer(
            model=BASE_MODEL_SOURCE,
            base_model_revision=BASE_MODEL,
        )
        sleep = SleepCycle(
            store,
            runtime,
            trainer,
            policy=REAL_CYCLE_POLICY,
            base_model_revision=BASE_MODEL,
            router_version="profile-router-v1",
            verifier_suite_version=suite_version,
            decoding_config=DECODING_CONFIG,
            heldout_targets=heldout_by_family,
        )
        cycle = sleep.run()

        def verify_and_store_artifacts(cycle_report) -> None:
            produced = [
                store.get_expert(expert_id)
                for expert_id in (
                    *cycle_report.experts_admitted,
                    *cycle_report.experts_rejected,
                )
            ]
            for expert in produced:
                local_path = Path(expert.artifact["local_adapter_path"])
                digest, size = _artifact_hash(local_path)
                if digest != expert.artifact["adapter_sha256"]:
                    raise RuntimeError(
                        f"adapter hash mismatch after transfer: {expert.id}"
                    )
                store.save_artifact(
                    Artifact(
                        id=f"artifact_{expert.id}",
                        kind="mlx-lora-adapter",
                        path=str(local_path),
                        sha256=digest,
                        size_bytes=size,
                        provenance={
                            "expert_id": expert.id,
                            "base_model": BASE_MODEL,
                            "dataset_sha256": expert.artifact["dataset_sha256"],
                        },
                    )
                )

        verify_and_store_artifacts(cycle)

        after_growth = benchmark.evaluate(cohorts, label="after_growth")

        # ------------------------------------------------------------------
        # Second growth cycle. The first expert stays admitted, deployed and
        # routable throughout: nothing is reset, and the second capture runs
        # through the live router exactly as arriving work would.
        # ------------------------------------------------------------------
        second_cycle_report = None
        second_live_summary = None
        second_correction_ids: list[str] = []
        coexistence = None
        if growth_cycles >= 2:
            coexistence_cycle_1 = coexistence_checkpoint(
                runtime,
                store,
                capability=after_growth,
                replay=store.successful_tasks(
                    limit=REAL_CYCLE_POLICY.replay_limit
                ),
                heldout_by_family=heldout_by_family,
                cycle_one_training_ids=cycle_one_training_ids,
            )
            # The second capture re-runs the regression and cycle-1 training
            # tasks routed, alongside the new family. Cycle-1 targets that now
            # pass through the deployed expert therefore enter the recorded
            # prior-passing suite, which is what puts them in cycle 2's replay
            # cohort; the new family's failures are captured for training.
            second_live_results = runtime.run(
                [
                    item.task
                    for item in [*regression, *training, *second_training]
                ],
                run_id="second_failure_capture",
                record=True,
            )
            second_live_summary = {
                "tasks": len(second_live_results),
                "passed": sum(
                    result.verification.passed for result in second_live_results
                ),
                "failed": sum(
                    not result.verification.passed
                    for result in second_live_results
                ),
                "results": {
                    result.task.id: {
                        "passed": result.verification.passed,
                        "score": result.verification.score,
                        "reason": result.verification.reason,
                        "routed_expert": result.route.expert_id,
                    }
                    for result in second_live_results
                },
            }
            # Only the second family's failures receive corrections here.
            # Residual cycle-1 failures stay visible and untrained: retraining
            # the first family would turn the coexistence question into a
            # retraining question.
            second_failures = [
                (failure, by_id[failure.task.id])
                for failure in store.unresolved_failures()
                if by_id[failure.task.id].role is DatasetRole.TRAIN
                and by_id[failure.task.id].task.metadata.get("failure_type")
                == SECOND_CYCLE_FAMILY
            ]
            second_trained_ids: list[str] = []
            for failure, item in second_failures:
                proposal = primary_source.propose(
                    failure.task, failure.response, verifiers.verify
                )
                response_sha256 = (
                    hashlib.sha256(proposal.response.encode()).hexdigest()
                    if proposal.response is not None
                    else None
                )
                training_proposals.append(
                    {
                        "cycle": 2,
                        "task_id": proposal.task_id,
                        "source": proposal.source,
                        "reused_from_comparison": False,
                        "accepted": proposal.accepted,
                        "attempts": proposal.attempts,
                        "generation_calls": proposal.generation_calls,
                        "response_sha256": response_sha256,
                        "decoding": proposal.provenance.get("decoding"),
                        "attempt_seeds": proposal.provenance.get("attempt_seeds"),
                    }
                )
                if proposal.response is None or proposal.verification is None:
                    continue
                correction = store.record_correction(
                    failure_id=failure.id,
                    response=proposal.response,
                    source=proposal.source,
                    verification=proposal.verification,
                    provenance={
                        "suite_version": item.suite.version,
                        "attempts": proposal.attempts,
                        "correction_sha256": response_sha256,
                        "reused_from_comparison": False,
                        "cycle": 2,
                    },
                )
                if correction.accepted:
                    second_correction_ids.append(correction.id)
                    second_trained_ids.append(item.task.id)
                    store.assign_dataset_role(
                        task_id=item.task.id,
                        role=DatasetRole.TRAIN,
                        content=item.task.prompt,
                        failure_id=failure.id,
                        correction_id=correction.id,
                    )
            record_second_cycle_failures(
                setup,
                attempted=[failure.task.id for failure, _ in second_failures],
                trained=second_trained_ids,
            )
            second_cycle_report = sleep.run()
            verify_and_store_artifacts(second_cycle_report)
            after_second_growth = benchmark.evaluate(
                cohorts, label="after_second_growth"
            )
            coexistence_cycle_2 = coexistence_checkpoint(
                runtime,
                store,
                capability=after_second_growth,
                replay=store.successful_tasks(
                    limit=REAL_CYCLE_POLICY.replay_limit
                ),
                heldout_by_family=heldout_by_family,
                cycle_one_training_ids=cycle_one_training_ids,
            )
            capability_0 = coexistence_baseline["capability"]
            capability_1 = coexistence_cycle_1["capability"]
            capability_2 = coexistence_cycle_2["capability"]
            coexistence = {
                "baseline": coexistence_baseline,
                "after_cycle_1": coexistence_cycle_1,
                "after_cycle_2": coexistence_cycle_2,
                "capability": {
                    "baseline": capability_0,
                    "after_cycle_1": capability_1,
                    "after_cycle_2": capability_2,
                    "delta_cycle_1": capability_1 - capability_0,
                    "delta_cycle_2": capability_2 - capability_1,
                    "monotonic_non_decreasing": (
                        capability_1 >= capability_0
                        and capability_2 >= capability_1
                    ),
                },
            }

        future_results = runtime.run(
            [item.task for item in future], run_id="future_stream_probe", record=False
        )
        future_probe = {
            "tasks": len(future_results),
            "passed": sum(result.verification.passed for result in future_results),
            "results": {
                result.task.id: {
                    "passed": result.verification.passed,
                    "score": result.verification.score,
                    "reason": result.verification.reason,
                    "routed_expert": result.route.expert_id,
                }
                for result in future_results
            },
        }
        rollback_metrics = None
        rollback_audit = None
        restored_deployment = None
        admitted_any = bool(
            cycle.experts_admitted
            or (second_cycle_report and second_cycle_report.experts_admitted)
        )
        if admitted_any:
            grown_deployment = store.current_deployment()
            assert grown_deployment is not None
            store.rollback_to(baseline_deployment.id, "real rollback drill")
            rollback_evaluation = benchmark.evaluate(cohorts, label="rollback_drill")
            # The metrics block stays exactly the metrics, so an audit against
            # the database compares like with like. The row that produced them
            # is named separately, which is what makes the audit authoritative
            # instead of a label-and-recency guess.
            rollback_audit = {
                "evaluation_id": rollback_evaluation["evaluation_id"],
                "run_id": rollback_evaluation["run_id"],
            }
            rollback_metrics = {
                key: value
                for key, value in rollback_evaluation.items()
                if key not in {"evaluation_id", "run_id"}
            }
            restored_deployment = store.rollback_to(
                grown_deployment.id, "restore admitted deployment after drill"
            )

        provenance = collect_provenance(
            repo_root=REPO_ROOT,
            base_model=BASE_MODEL,
            verifier_suite_version=suite_version,
            training_config=trainer.training_config(),
            decoding_config=DECODING_CONFIG,
            verifier_suites=_verifier_suite_manifest(full_catalog),
            model_paths={"base": BASE_MODEL_SOURCE},
            sandbox_image=sandbox.policy.image,
            # The worker did the training, so its identity is part of the run.
            worker=_worker_provenance(trainer.worker),
            extra={
                "correction_source": correction_source,
                "self_repair_attempts": self_repair_attempts,
                "self_repair_decoding": _self_repair_decoding_record(
                    self_repair_decoding,
                    correction_source=correction_source,
                ),
                "admission_policy": asdict(REAL_CYCLE_POLICY),
                "growth_cycles": growth_cycles,
                "database": str(database),
            },
        )

        report = {
            "base_model": BASE_MODEL,
            "arm": selected_arm,
            "experiment_spec": experiment_spec,
            "preflight": preflight,
            "run_setup": setup,
            "required_setup_check": setup_check,
            "replay_capacity": capacity,
            "sandbox": sandbox_preflight,
            "provenance": provenance,
            "provenance_gaps": missing_fields(provenance),
            "baseline": baseline,
            "live_capture": live_summary,
            "correction_source": correction_source,
            "correction_comparison": correction_comparison,
            "training_proposals": training_proposals,
            "verified_corrections": len(correction_ids),
            "cycle": asdict(cycle),
            "after_growth": after_growth,
            "growth_cycles": growth_cycles,
            "second_live_capture": second_live_summary,
            "second_cycle": (
                asdict(second_cycle_report) if second_cycle_report else None
            ),
            "second_cycle_verified_corrections": (
                len(second_correction_ids) if growth_cycles >= 2 else None
            ),
            "coexistence": coexistence,
            "future_probe": future_probe,
            "rollback": rollback_metrics,
            "rollback_audit": rollback_audit,
            "restored_deployment": (
                asdict(restored_deployment) if restored_deployment else None
            ),
            "curve": benchmark.curve(),
            "store": store.summary(),
            "evaluation_integrity": store.verify_evaluations(),
            "ledger_integrity": store.verify_ledger(),
            "experts": [
                {
                    "id": expert.id,
                    # Correction-source-independent identity, so two arms of a
                    # paired experiment can be matched at all. Expert ids are
                    # fresh UUIDs and never match across runs.
                    "pairing_key": expert.metrics.get("pairing_key"),
                    "status": expert.status.value,
                    "artifact": expert.artifact,
                    "metrics": expert.metrics,
                }
                for expert in store.experts()
            ],
            "deployments": [asdict(manifest) for manifest in store.deployments()],
            "run_started_at": run_started_at,
            "run_finished_at": utc_now(),
            "evaluation_ids": [
                evaluation_id
                for evaluation_id in (
                    baseline.get("evaluation_id"),
                    after_growth.get("evaluation_id"),
                    (
                        (coexistence or {})
                        .get("after_cycle_2", {})
                        .get("evaluation_id")
                    ),
                    (rollback_audit or {}).get("evaluation_id"),
                )
                if evaluation_id
            ],
        }
        seal_report(report, spec=spec)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Indented for a human reader; the digests above committed to the
        # same values through the strict canonical encoder.
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return report


RUN_MANIFEST_SCHEMA = "grove-run-manifest-v2"
# The one field a manifest cannot commit to, because it *is* the commitment.
# It is checked by recomputing the manifest, not by binding a resolved value.
UNBINDABLE_REPORT_PATHS = ("run_manifest_sha256", "run_manifest")
# Every rule field the evaluator can dereference into a report. This list is
# the registry: adding a path-bearing field to a decision rule without adding
# it here is what let three EXP-003 rule inputs stay unbound.
RULE_PATH_FIELDS = ("path", "control_path", "pair_on")


def decision_rule_input_paths(spec: Mapping[str, Any]) -> list[str]:
    """Every report path the sealed spec's decision rules can read.

    Naming the bound fields by hand is how the binding drifts: a new rule reads
    a new path, nobody updates the manifest, and editing that path becomes a
    silent false pass. The path set is derived from the spec instead, so a rule
    that reads something the manifest does not commit to cannot exist.
    """
    paths: set[str] = set()
    for rule in spec.get("decision_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        for field in RULE_PATH_FIELDS:
            value = rule.get(field)
            if isinstance(value, str) and value and value not in UNBINDABLE_REPORT_PATHS:
                paths.add(value)
    return sorted(paths)


class _MissingInput:
    """Sentinel for a report path the document does not carry."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING_INPUT = _MissingInput()
DECISION_INPUT_SCHEMA = "grove-decision-input-v1"


def resolve_report_path(document: Any, path: str) -> Any:
    """Resolve a dotted report path, where ``[*]`` fans out over a list.

    Mirrors ``scripts/check_experiment_spec.resolve``. ``tests`` pin the two to
    the same output, because a sealing resolver that disagrees with the checking
    resolver would bind one value and verify another.
    """
    current: Any = document
    for segment in path.split("."):
        fanout = segment.endswith("[*]")
        key = segment[:-3] if fanout else segment
        if key:
            if isinstance(current, list):
                current = [
                    item.get(key, MISSING_INPUT) if isinstance(item, Mapping) else MISSING_INPUT
                    for item in current
                ]
            elif isinstance(current, Mapping):
                current = current.get(key, MISSING_INPUT)
            else:
                return MISSING_INPUT
        if fanout and (current is MISSING_INPUT or not isinstance(current, list)):
            return MISSING_INPUT
    return current


def decision_input_bindings(
    report: Mapping[str, Any], paths: Sequence[str]
) -> dict[str, Any]:
    """Commit to the exact resolved value behind each decision-rule path.

    Array order is preserved, so reordering experts changes the digest. An
    absent path is recorded as absent rather than skipped: a rule that reads
    nothing must not be able to start reading something later.
    """
    bindings: dict[str, Any] = {}
    for path in paths:
        value = resolve_report_path(report, path)
        present = value is not MISSING_INPUT and MISSING_INPUT not in _flatten_input(value)
        bindings[path] = {
            "present": present,
            "sha256": canonical_hash(
                {"schema": DECISION_INPUT_SCHEMA, "path": path, "value": value}
            )
            if present
            else None,
        }
    return bindings


def _flatten_input(value: Any) -> list[Any]:
    if isinstance(value, list):
        flat: list[Any] = []
        for item in value:
            flat.extend(_flatten_input(item))
        return flat
    return [value]


def _rollback_evaluation_selector(audit: Any) -> dict[str, Any] | None:
    """Select the exact persisted evaluation row behind rollback metrics."""
    if not isinstance(audit, Mapping):
        return None
    return {
        "evaluation_id": audit.get("evaluation_id"),
        "run_id": audit.get("run_id"),
    }


def build_run_manifest(
    report: Mapping[str, Any], *, rule_paths: Sequence[str] = ()
) -> dict[str, Any]:
    """One canonical record binding a report's claims to its own evidence.

    Without it the checker read numbers straight out of an editable JSON file:
    changing ``forced_regression_rate`` from 0.5 to 0.0 turned a falsified run
    into a satisfied one and nothing objected.

    ``decision_inputs`` is the general binding, derived from the sealed spec's
    own rules. The named digests below are kept because they cover run state no
    rule happens to read -- the setup manifest, the provenance record, the
    training proposals -- and a report should commit to those whether or not a
    rule looks at them.

    This is a local integrity binding, not an attestation. It detects an edited
    report; it cannot stop someone who rewrites the manifest too. Signing the
    manifest or anchoring its digest externally is what would.
    """
    provenance = report.get("provenance") or {}
    setup = report.get("run_setup") or {}
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "arm": report.get("arm"),
        "correction_source": report.get("correction_source"),
        "experiment_spec": report.get("experiment_spec"),
        "run_setup_sha256": canonical_hash(setup),
        "cohort_manifest_sha256": setup.get("cohort_manifest_sha256"),
        "admission_policy_sha256": setup.get("admission_policy_sha256"),
        "actual_training_failure_set_sha256": setup.get(
            "actual_training_failure_set_sha256"
        ),
        "attempted_training_failure_set_sha256": setup.get(
            "attempted_training_failure_set_sha256"
        ),
        "trained_failure_set_sha256": setup.get("trained_failure_set_sha256"),
        "provenance_sha256": provenance.get("provenance_sha256"),
        "rollback_sha256": canonical_hash(report.get("rollback")),
        "rollback_audit_sha256": canonical_hash(report.get("rollback_audit")),
        "rollback_evaluation_selector": _rollback_evaluation_selector(
            report.get("rollback_audit")
        ),
        "provenance_gaps": list(report.get("provenance_gaps") or []),
        "training_proposals_sha256": canonical_hash(
            report.get("training_proposals") or []
        ),
        "decision_inputs": decision_input_bindings(report, rule_paths),
        "experts": [
            {
                "id": expert.get("id"),
                "pairing_key": (expert.get("metrics") or {}).get("pairing_key"),
                "adapter_sha256": (expert.get("artifact") or {}).get("adapter_sha256"),
                "metrics_sha256": canonical_hash(expert.get("metrics") or {}),
            }
            for expert in report.get("experts") or []
        ],
        "evaluation_ids": list(report.get("evaluation_ids") or []),
        "deployment_ids": [
            manifest.get("id") for manifest in report.get("deployments") or []
        ],
        "run_started_at": report.get("run_started_at"),
        "run_finished_at": report.get("run_finished_at"),
    }


def seal_report(
    report: dict[str, Any], *, spec: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Attach the run manifest and its digest to a finished report.

    The manifest is built from the JSON round trip of the report, not from the
    live Python objects, so the digest a reader recomputes from the file is the
    digest that was written. The round trip is strict: a value JSON cannot
    represent fails here rather than being stringified into a digest.

    A spec whose decision rules grade report values cannot be sealed without a
    provenance digest. The legacy flag remains accepted, but it cannot switch
    this integrity requirement off.
    """
    requires_integrity = bool(
        spec
        and (
            spec.get("requires_report_integrity")
            or spec.get("decision_rules")
        )
    )
    if requires_integrity:
        digest = (report.get("provenance") or {}).get("provenance_sha256")
        if not isinstance(digest, str) or not digest:
            raise ExperimentSetupError(
                "cannot seal a report whose decision rules grade report data "
                "without provenance.provenance_sha256"
            )
    serialized = json.loads(canonical_json(report))
    rule_paths = decision_rule_input_paths(spec or {})
    manifest = build_run_manifest(serialized, rule_paths=rule_paths)
    report["run_manifest"] = manifest
    report["run_manifest_sha256"] = canonical_hash(manifest)
    return report
