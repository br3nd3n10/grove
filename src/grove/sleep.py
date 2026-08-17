from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

from grove.interfaces import ExpertTrainer
from grove.models import (
    Cluster,
    CycleReport,
    Expert,
    ExpertStatus,
    GateDecision,
    Task,
    utc_now,
)
from grove.provenance import canonical_hash
from grove.routing import FingerprintClusterer
from grove.runtime import GroveRuntime
from grove.store import GroveStore


@dataclass(frozen=True, slots=True)
class SleepPolicy:
    min_cluster_size: int = 3
    min_target_fix_rate: float = 0.8
    max_regression_rate: float = 0.0
    min_plasticity_gain: float = 0.5
    min_heldout_fix_rate: float = 0.75
    require_heldout_targets: bool = False
    replay_limit: int = 200
    max_candidates_per_cycle: int = 5
    # Routed replay only measures the deployed configuration, where the router
    # shields the candidate from out-of-family prompts. Forced replay runs the
    # same tasks with the candidate switched on unconditionally, which measures
    # the adapter itself. Keep both; they answer different questions.
    measure_forced_replay: bool = True
    # Also evaluate replay with no experts at all, separating "the base could do
    # this" from "the previous deployment could do this". Costs one extra pass
    # over the replay cohort, so it is opt-in.
    measure_base_reference: bool = False
    # Permissive by default so the recorded forgetting claim, not a silent gate,
    # carries the honesty burden. Predeclared experiments tighten this to 0.0
    # when they intend to assert adapter-intrinsic stability.
    max_forced_regression_rate: float = 1.0
    # Smallest prior-passing replay cohort that may support a stability claim.
    # Zero means "no declared minimum"; a candidate still reports its cohort
    # size and an unmeasured claim when the cohort is empty.
    min_replay_examples: int = 0
    # When true, an unmeasurable stability or recall result fails admission
    # rather than passing quietly. Predeclared experiments set these.
    require_measured_replay: bool = False
    require_measured_route_recall: bool = False
    # Routing quality gates. A candidate that never wins its own family is not
    # deployable, and one that grabs base-passing replay traffic is exactly the
    # failure mode that hides forgetting behind the router.
    min_route_recall: float = 0.5
    max_route_false_positive_rate: float = 0.0
    # There is deliberately no min_route_precision. Route precision's
    # denominator mixes the held-out positive cohort with the replay negative
    # cohort, so resizing the replay buffer moves it even when the router is
    # unchanged. The field used to be accepted as an InitVar and silently
    # dropped, which let a spec declare a precision gate that never ran. A
    # keyword that cannot gate anything now raises TypeError instead.


PAIRING_KEY_SCHEMA = "grove-expert-pairing-key-v1"


def cluster_pairing_key(cluster: Cluster) -> str:
    """A cross-run identity for the expert a cluster produces.

    Expert ids are fresh UUIDs, so pairing two arms of an A/B experiment on
    ``experts[*].id`` can never match: the same training data yields a different
    id every run and the checker either refuses or, worse, reads the refusal as
    a falsified hypothesis. This key is built from what the two arms are meant
    to share -- the failure family and the exact set of failing tasks selected
    for training -- and from nothing that the correction source changes. The
    key therefore identifies the training selection, not the subset that later
    receives an accepted correction and becomes trainable.

    Matching keys mean the arms selected the same cluster over the same failed
    tasks. They do not prove the training data was byte-identical; the
    correction texts differ by design, which is the variable under test.
    """
    return canonical_hash(
        {
            "schema": PAIRING_KEY_SCHEMA,
            "cluster_label": cluster.label,
            "training_task_ids": sorted(
                failure.task.id for failure in cluster.failures
            ),
        }
    )


def oracle_free(task: Task) -> Task:
    """A routing-probe copy of a task with its gold labels removed.

    Held-out tasks keep the family tags the clusterer used to build the expert's
    routing profile, so scoring the router on the original task measures tag
    equality, not routing. Stripping tags and metadata leaves the router only
    the prompt, which is all a live request would carry.
    """
    return replace(task, tags=(), metadata={})


def forgetting_claim(
    *,
    forced_regression_rate: float | None,
    forced_regression_reference: str,
    routed_regression_rate: float | None,
) -> str:
    """The honest name for a forgetting measurement.

    Shared by candidate probation and the multi-expert coexistence probe so a
    per-expert forced-on measurement in a two-expert deployment resolves under
    exactly the same vocabulary an admission gate uses. Only
    ``adapter_intrinsic`` licenses a plain "no forgetting" sentence, and only
    against a bare-base reference; routed regression outranks any shield story.
    """
    if forced_regression_rate is None:
        return "unmeasured"
    if routed_regression_rate is None or routed_regression_rate > 0:
        return "regression"
    if forced_regression_rate > 0:
        return "router_shielded"
    if forced_regression_reference == "base_no_experts":
        return "adapter_intrinsic"
    return "unverified_reference"


@dataclass(frozen=True, slots=True)
class ProbationResult:
    decision: GateDecision
    fixed_failure_ids: tuple[str, ...]


class SleepCycle:
    """Capture -> cluster -> demand -> train -> probation -> admit/reject."""

    def __init__(
        self,
        store: GroveStore,
        runtime: GroveRuntime,
        trainer: ExpertTrainer,
        *,
        clusterer: FingerprintClusterer | None = None,
        policy: SleepPolicy | None = None,
        base_model_revision: str = "unversioned-base",
        router_version: str = "profile-router-v1",
        verifier_suite_version: str = "verifier-registry-v1",
        decoding_config: Mapping[str, object] | None = None,
        heldout_targets: Mapping[str, Sequence[Task]] | None = None,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.trainer = trainer
        self.clusterer = clusterer or FingerprintClusterer()
        self.policy = policy or SleepPolicy()
        self.base_model_revision = base_model_revision
        self.router_version = router_version
        self.verifier_suite_version = verifier_suite_version
        self.decoding_config = dict(decoding_config or {"temperature": 0.0})
        self.heldout_targets = heldout_targets or {}

    def run(self) -> CycleReport:
        started_at = utc_now()
        cycle_id = self.store.start_cycle()
        clusters = self.clusterer.cluster(self.store.unresolved_failures())
        admitted: list[str] = []
        rejected: list[str] = []
        skipped: list[dict[str, object]] = []
        trained = 0

        for cluster in clusters:
            if trained >= self.policy.max_candidates_per_cycle:
                skipped.append(
                    {
                        "cluster": cluster.label,
                        "reason": "cycle candidate limit reached",
                    }
                )
                continue
            demand = self._demand_gate(cluster)
            if not demand.passed:
                skipped.append({"cluster": cluster.label, "reason": demand.reason})
                continue

            candidate_id = f"expert_{uuid.uuid4().hex[:12]}"
            candidate = self.trainer.train(cluster, candidate_id)
            if candidate.status is not ExpertStatus.CANDIDATE:
                raise ValueError("trainer must return an isolated candidate expert")
            self.store.save_expert(candidate, event="expert.candidate_trained")
            trained += 1

            probation = self._probation_gate(cluster, candidate)
            metrics = {
                **candidate.metrics,
                **probation.decision.metrics,
                "pairing_key": cluster_pairing_key(cluster),
            }
            if probation.decision.passed:
                accepted = replace(
                    candidate,
                    status=ExpertStatus.ACTIVE,
                    metrics=metrics,
                    updated_at=utc_now(),
                )
                self.store.save_expert(accepted, event="expert.admitted")
                self.store.mark_failures_resolved(
                    probation.fixed_failure_ids, accepted.id
                )
                self._publish_deployment(f"admitted {accepted.id}")
                admitted.append(accepted.id)
            else:
                denied = replace(
                    candidate,
                    status=ExpertStatus.REJECTED,
                    metrics={**metrics, "rejection_reason": probation.decision.reason},
                    updated_at=utc_now(),
                )
                self.store.save_expert(denied, event="expert.rejected")
                rejected.append(denied.id)

        finished_at = utc_now()
        report = CycleReport(
            cycle_id=cycle_id,
            clusters_seen=len(clusters),
            candidates_trained=trained,
            experts_admitted=tuple(admitted),
            experts_rejected=tuple(rejected),
            skipped=tuple(skipped),
            started_at=started_at,
            finished_at=finished_at,
        )
        self.store.finish_cycle(cycle_id, asdict(report))
        return report

    def remove_expert(self, expert_id: str, reason: str) -> Expert:
        expert = self.store.get_expert(expert_id)
        if expert.status is not ExpertStatus.ACTIVE:
            raise ValueError(
                f"only active experts can be removed; {expert_id} is {expert.status.value}"
            )
        # Membership of the replacement manifest comes from what is deployed
        # right now, minus the removed expert. Rebuilding it from lifecycle
        # status instead would silently redeploy an active expert that an
        # earlier rollback had unplugged -- removal must never plug anything in.
        current = self.store.current_deployment()
        removed = replace(
            expert,
            status=ExpertStatus.REMOVED,
            metrics={**expert.metrics, "removal_reason": reason},
            updated_at=utc_now(),
        )
        self.store.save_expert(removed, event="expert.removed")
        if current is None:
            # With no deployment there is no prior membership to preserve. The
            # remaining ACTIVE lifecycle records are the only meaningful
            # deployment set, and _publish_deployment derives exactly that set
            # after the removed expert has been saved.
            self._publish_deployment(f"removed {expert_id}: {reason}")
        else:
            remaining = tuple(item for item in current.expert_ids if item != expert_id)
            # A removal is a replacement of the current manifest, not a new
            # configuration. Preserve every pin from that manifest even when
            # this SleepCycle was constructed with different defaults.
            self._publish_deployment(
                f"removed {expert_id}: {reason}",
                expert_ids=remaining,
                base_model_revision=current.base_model_revision,
                router_version=current.router_version,
                verifier_suite_version=current.verifier_suite_version,
                decoding_config=current.decoding_config,
            )
        return removed

    def _publish_deployment(
        self,
        reason: str,
        *,
        expert_ids: Sequence[str] | None = None,
        base_model_revision: str | None = None,
        router_version: str | None = None,
        verifier_suite_version: str | None = None,
        decoding_config: Mapping[str, object] | None = None,
    ) -> None:
        members = (
            tuple(expert_ids)
            if expert_ids is not None
            else tuple(
                expert.id for expert in self.store.experts(ExpertStatus.ACTIVE)
            )
        )
        self.store.publish_deployment(
            base_model_revision=(
                self.base_model_revision
                if base_model_revision is None
                else base_model_revision
            ),
            expert_ids=members,
            router_version=(
                self.router_version if router_version is None else router_version
            ),
            verifier_suite_version=(
                self.verifier_suite_version
                if verifier_suite_version is None
                else verifier_suite_version
            ),
            decoding_config=dict(
                self.decoding_config if decoding_config is None else decoding_config
            ),
            reason=reason,
        )

    def _demand_gate(self, cluster: Cluster) -> GateDecision:
        if len(cluster.failures) < self.policy.min_cluster_size:
            return GateDecision(
                False,
                f"needs {self.policy.min_cluster_size} failures; observed {len(cluster.failures)}",
            )
        correctable = sum(
            failure.correction is not None for failure in cluster.failures
        )
        if correctable < self.policy.min_cluster_size:
            return GateDecision(
                False, "not enough verifier-backed corrections to train safely"
            )
        return GateDecision(True, "recurring, correctable failure cluster")

    def _probation_gate(self, cluster: Cluster, candidate: Expert) -> ProbationResult:
        active = self.store.experts(ExpertStatus.ACTIVE)
        cluster_tasks = [failure.task for failure in cluster.failures]
        target_before = self.runtime.run(cluster_tasks, experts=active, record=False)
        target_after = self.runtime.run(
            cluster_tasks, force_expert=candidate, record=False
        )
        before_rate = self._pass_rate(target_before)
        after_rate = self._pass_rate(target_after)
        fixed_ids = tuple(
            failure.id
            for failure, result in zip(cluster.failures, target_after, strict=True)
            if result.verification.passed
        )

        replay = self.store.successful_tasks(
            limit=self.policy.replay_limit,
            exclude_fingerprints=self._cluster_fingerprints(cluster),
        )
        # routed_before is the previous deployment, which may already contain
        # experts. It is not the bare base unless no expert is active, so the
        # two references are recorded separately and never conflated.
        routed_before = self.runtime.run(replay, experts=active, record=False)
        shadow_pool = [*active, candidate]
        routed_after = self.runtime.run(replay, experts=shadow_pool, record=False)
        base_results = None
        base_reference: dict[str, object] = {
            "base_reference_measured": False,
            "base_replay_pass_rate": None,
        }
        if not active:
            base_results = routed_before
            base_reference = {
                "base_reference_measured": True,
                "base_replay_pass_rate": self._pass_rate(base_results, empty=None),
            }
        elif self.policy.measure_base_reference:
            base_results = self.runtime.run(replay, experts=[], record=False)
            base_reference = {
                "base_reference_measured": True,
                "base_replay_pass_rate": self._pass_rate(base_results, empty=None),
            }

        routed_regression = self._regression_by_task(routed_before, routed_after)
        forced_metrics = self._forced_replay_metrics(
            candidate,
            replay,
            routed_before,
            base_results=base_results,
            routed_regression_rate=routed_regression["rate"],
        )
        plasticity_gain = after_rate - before_rate
        heldout_tasks = list(self.heldout_targets.get(cluster.label, ()))
        heldout_after = self.runtime.run(
            heldout_tasks, force_expert=candidate, record=False
        )
        heldout_rate = self._pass_rate(
            heldout_after,
            empty=0.0 if self.policy.require_heldout_targets else 1.0,
        )
        route_metrics = self._route_metrics(candidate, heldout_after, replay, active)

        metrics = {
            "target_before": before_rate,
            "target_after": after_rate,
            "plasticity_gain": plasticity_gain,
            "heldout_target_rate": heldout_rate,
            "heldout_target_examples": len(heldout_tasks),
            # Aggregate pass rates over the replay cohort. These are capability
            # numbers; the forgetting number is regression_rate, which counts
            # per-task losses so a new pass cannot cancel a lost one.
            "replay_pass_rate_before": self._pass_rate(routed_before, empty=None),
            "replay_pass_rate_routed": self._pass_rate(routed_after, empty=None),
            "regression_rate": routed_regression["rate"],
            "regression_denominator": routed_regression["denominator"],
            "regression_task_ids": routed_regression["regressed_task_ids"],
            "regression_reference_scope": (
                "routed_before_with_active_experts" if active else "base_no_experts"
            ),
            "active_experts_at_probation": len(active),
            "target_examples": len(cluster.failures),
            "replay_examples": len(replay),
            "replay_task_ids": [task.id for task in replay],
            **base_reference,
            **forced_metrics,
            **route_metrics,
        }
        checks = (
            (
                after_rate >= self.policy.min_target_fix_rate,
                "target fix rate below threshold",
            ),
            (
                plasticity_gain >= self.policy.min_plasticity_gain,
                "plasticity gain below threshold",
            ),
            (
                heldout_rate >= self.policy.min_heldout_fix_rate,
                "held-out target fix rate below threshold",
            ),
            self._threshold_check(
                routed_regression["rate"],
                self.policy.max_regression_rate,
                "at most",
                "routed regression budget exceeded",
                "routed regression is unmeasured (no prior-passing replay task)",
                required=self.policy.require_measured_replay,
            ),
            (
                forced_metrics["forced_regression_denominator"]
                >= self.policy.min_replay_examples,
                "prior-passing replay cohort below the declared minimum",
            ),
            self._threshold_check(
                forced_metrics["forced_regression_rate"],
                self.policy.max_forced_regression_rate,
                "at most",
                "forced-adapter regression budget exceeded",
                "forced-adapter stability is unmeasured",
                required=self.policy.require_measured_replay,
            ),
            self._threshold_check(
                route_metrics["route_recall"],
                self.policy.min_route_recall,
                "at least",
                "route recall below threshold",
                "route recall is unmeasured (no independent held-out positive)",
                required=self.policy.require_measured_route_recall,
            ),
            # route_precision is deliberately not a gate: its denominator mixes
            # positives and replay negatives, so changing only replay-buffer
            # size changes the value even when the router did not change.
            self._threshold_check(
                route_metrics["route_false_positive_rate"],
                self.policy.max_route_false_positive_rate,
                "at most",
                "router claims base-passing replay traffic",
                "route false-positive rate is unmeasured (no replay negative)",
                required=self.policy.require_measured_replay,
            ),
        )
        failed = [reason for passed, reason in checks if not passed]
        return ProbationResult(
            GateDecision(
                not failed,
                "; ".join(failed) if failed else "all probation gates passed",
                metrics,
            ),
            fixed_ids,
        )

    @staticmethod
    def _threshold_check(
        observed: float | None,
        threshold: float,
        direction: str,
        breach_reason: str,
        unmeasured_reason: str,
        *,
        required: bool,
    ) -> tuple[bool, str]:
        """Compare a rate that may be unmeasurable.

        An unmeasured rate is never silently treated as a passing zero. It
        either fails outright, when the policy demands a measurement, or is
        skipped and left visible in the metrics as ``None``.
        """
        if observed is None:
            return (not required), unmeasured_reason
        if direction == "at most":
            return observed <= threshold, breach_reason
        return observed >= threshold, breach_reason

    @staticmethod
    def _regression_by_task(before: list, after: list) -> dict[str, object]:
        """Count lost passes per task, never as a net pass-rate difference.

        A net difference lets an unrelated newly fixed task cancel a genuinely
        broken one, which is precisely the arithmetic that hides forgetting.
        The denominator is the set of tasks that passed in the reference run; if
        that set is empty there is nothing to forget and the rate is ``None``,
        not zero.
        """
        paired = list(zip(before, after, strict=True))
        prior_passing = [
            (baseline, candidate)
            for baseline, candidate in paired
            if baseline.verification.passed
        ]
        regressed = [
            candidate.task.id
            for baseline, candidate in prior_passing
            if not candidate.verification.passed
        ]
        return {
            "denominator": len(prior_passing),
            "regressed_task_ids": regressed,
            "rate": (len(regressed) / len(prior_passing)) if prior_passing else None,
        }

    def _forced_replay_metrics(
        self,
        candidate: Expert,
        replay: list[Task],
        routed_before: list,
        *,
        base_results: list | None,
        routed_regression_rate: float | None,
    ) -> dict[str, object]:
        """Measure the adapter itself, with the router shield switched off.

        Routed replay can report zero forgetting purely because the router never
        sends a replay prompt to the candidate. Forcing the candidate on for the
        same tasks separates ``the deployed system does not forget`` from the
        much stronger ``this adapter does not forget``.

        An empty or all-failing replay cohort yields no evidence either way, so
        the claim is ``unmeasured``. Silence must never read as stability.
        """
        reference_results = base_results if base_results is not None else routed_before
        reference_scope = (
            "base_no_experts"
            if base_results is not None
            else "routed_before_with_active_experts"
        )
        if not self.policy.measure_forced_replay:
            return {
                "forced_replay_measured": False,
                "forced_replay_examples": len(replay),
                "forced_replay_rate": None,
                "forced_regression_rate": None,
                "forced_regression_denominator": 0,
                "forced_regression_task_ids": [],
                "forced_regression_reference": reference_scope,
                "forgetting_claim": "unmeasured",
            }
        forced = self.runtime.run(replay, force_expert=candidate, record=False)
        regression = self._regression_by_task(reference_results, forced)
        rate = regression["rate"]
        return {
            "forced_replay_measured": rate is not None,
            "forced_replay_examples": len(replay),
            "forced_replay_rate": self._pass_rate(forced, empty=None),
            "forced_regression_rate": rate,
            "forced_regression_denominator": regression["denominator"],
            "forced_regression_task_ids": regression["regressed_task_ids"],
            "forced_regression_reference": reference_scope,
            # The single most over-claimed number in this project. Only
            # "adapter_intrinsic" licenses a plain "no forgetting" sentence,
            # and only when something was actually at stake against the bare
            # base. Routed regression outranks any router-shield story.
            "forgetting_claim": self._forgetting_claim(
                forced_regression_rate=rate,
                forced_regression_reference=reference_scope,
                routed_regression_rate=routed_regression_rate,
            ),
        }

    @staticmethod
    def _forgetting_claim(
        *,
        forced_regression_rate: float | None,
        forced_regression_reference: str,
        routed_regression_rate: float | None,
    ) -> str:
        return forgetting_claim(
            forced_regression_rate=forced_regression_rate,
            forced_regression_reference=forced_regression_reference,
            routed_regression_rate=routed_regression_rate,
        )

    def _route_metrics(
        self,
        candidate: Expert,
        heldout_results: list,
        replay: list[Task],
        active: list[Expert],
    ) -> dict[str, object]:
        """Score the router against independent evidence, not its own training.

        Positives are held-out tasks the candidate demonstrably solves when
        forced on. Using the training cluster instead would be circular: the
        trainer builds the routing profile out of those very tasks, so recall
        would be guaranteed by construction rather than measured. Negatives are
        prior-passing replay tasks the candidate should leave alone.

        The probe routes oracle-free copies. A held-out task still carries the
        gold family tag the clusterer used to build the candidate's routing
        profile, and ``ProfileRouter.score`` scores an exact tag overlap at
        1.0, so routing the original task measured whether two copies of the
        same label are equal. That is not routing evidence. The gold-tag number
        is still reported, clearly named and clearly non-independent, because
        the difference between the two is itself worth seeing.

        Routing decisions are read straight from the router, so this costs no
        extra model inference.
        """
        pool = [*active, candidate]
        router = self.runtime.router
        positives = [
            result.task for result in heldout_results if result.verification.passed
        ]

        def routed(tasks: Sequence[Task], *, strip: bool) -> int:
            return sum(
                router.route(oracle_free(task) if strip else task, pool).expert_id
                == candidate.id
                for task in tasks
            )

        positives_routed = routed(positives, strip=True)
        negatives_routed = routed(replay, strip=True)
        gold_positives_routed = routed(positives, strip=False)
        routed_total = positives_routed + negatives_routed
        return {
            "route_positive_source": "heldout_forced_pass_oracle_free",
            "route_probe_metadata": "oracle_free",
            "route_positives": len(positives),
            "route_negatives": len(replay),
            "route_positives_routed": positives_routed,
            "route_negatives_routed": negatives_routed,
            "route_recall": (positives_routed / len(positives) if positives else None),
            # Diagnostic only, never a gate and never independent evidence: the
            # gold family tag is the very label the routing profile was built
            # from, so this measures label equality rather than routing.
            "route_recall_gold_tags": (
                gold_positives_routed / len(positives) if positives else None
            ),
            "route_recall_gold_tags_independent": False,
            # Diagnostic only, not a gate: this denominator deliberately mixes
            # positive and negative cohorts, so replay-buffer size can move it
            # even when route recall and false-positive rate are unchanged.
            "route_precision": (
                positives_routed / routed_total if routed_total else None
            ),
            "route_precision_cohort_dependent": True,
            "route_false_positive_rate": (
                negatives_routed / len(replay) if replay else None
            ),
        }

    @staticmethod
    def _cluster_fingerprints(cluster: Cluster) -> set[str]:
        fingerprints = {cluster.label}
        fingerprints.update(failure.fingerprint for failure in cluster.failures)
        return fingerprints

    @staticmethod
    def _pass_rate(results: list, *, empty: float | None = 0.0) -> float | None:
        if not results:
            return empty
        return sum(result.verification.passed for result in results) / len(results)
