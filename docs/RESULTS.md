# Grove results

## 1. Thesis: not proven

Grove's core thesis is that an agent can safely accumulate removable skills from verified failures without changing its frozen base model. **That thesis is not proven.** The measured system has admitted one narrow expert, but the pool has never held two simultaneously active experts. The results support a narrower statement: this control plane can capture evidence, train and gate an isolated adapter, route it, remove it, and refuse a deployment whose measured routing would regress traffic.

The negative results are part of the result. Stability was shown to depend on router shielding rather than adapter-intrinsic retention; this model produced no verified self-repairs in either of two regimes; and the first attempted second expert was rejected because the router over-claimed traffic the base already solved.

## 2. The five sealed experiments

### EXP-001 — first real growth cycle (2026-07-31)

A frozen Qwen2.5-Coder 1.5B base produced one admitted **2,637,824-parameter** LoRA expert after three candidates were rejected. On the original held-out set, the result moved from **0/4 to 3/4**. The expert was removed and restored through append-only, manifest-only deployment changes; the base model was not edited. The [complete experiment report](EXPERIMENT_REPORT_2026-07-31.md) is the source for these measurements.

### Independent verification (2026-08-01)

The independent check reproduced the artifact hashes, ledger checks, and original **0/4 to 3/4** result exactly. Six fresh holdouts written after admission moved from **0/6 for the base to 5/6 with the adapter**. The same check also forced the adapter on and found that it broke `reg_dedupe`, a task the base solved. This is why the original no-regression observation is a routed-system result, not evidence of adapter-intrinsic retention. See the [independent verification](VERIFICATION_2026-08-01.md).

### EXP-002 — forced versus routed replay (2026-08-08 rerun)

The healthy rerun tested whether stability belonged to the adapter or to the router. Forced replay regressed **46/94** prior-passing tasks. Routed replay passed **94/94**, with route precision and recall both **1.0** on that single-expert probe. H3 was falsified: the observed stability is **router shielding**, not adapter-intrinsic retention. The [rerun record](../research/2026-08-08-exp002-rerun.md) distinguishes this measured result from the earlier same-day attempt that was unable to exercise the experiment because its sandbox launches timed out.

### EXP-003 — self-repair under greedy attempts (2026-08-08)

The self-repair arm made three greedy attempts per failure and produced **0/20 verified self-repairs**. No self-repair expert was trained, so the paired arms could not be graded; the sealed checker returned **unusable (exit 2)**. This is a measured zero, not a formal falsification of the paired hypotheses. The [research record](../research/2026-08-08-exp003-ab.md) is explicit about that distinction.

### EXP-004 — self-repair under a fair sampled regime (2026-08-09)

The fair regime used eight sampled, seeded attempts per failure with honest verifier feedback: **0/20 verified self-repairs in 160 calls**. The canonical control arm verified **20/20** and admitted an expert. Because the self-repair arm produced no expert, the paired run was **unusable (exit 2)** rather than formally graded. Across EXP-003 and EXP-004, the measured total was **0 verified self-repairs in 220 attempts**. See the [fair-regime research record](../research/2026-08-09-exp004-fair-self-repair.md).

### EXP-005 — second cycle and two-expert coexistence (2026-08-11)

The `path_restructure` candidate met every competence bar: **17/20** target fixes (**0.85**), **3/4** held-out (**0.75**), and route recall **1.0**. Deployment gates found a different problem. Against the deployed first expert, the router claimed **18/112** base-passing replay tasks for the candidate: route precision was **0.143**, the false-positive rate was **0.161**, and **12/112** tasks regressed under routing (**0.107**) against a sealed budget of **0.0**. The unchanged admission policy rejected the candidate, leaving a pool of one expert.

The checker returned exit **1**: **23 of 29** rules passed and **6 failed**, with H1 and H2 falsified. Capability moved from **0.8468** at baseline to **0.8739** after cycle 1; the cycle-2 delta was exactly **0.0** because the second expert was never deployed. The [second-cycle research record](../research/2026-08-11-exp005-second-cycle.md) is the source for these values and for the conclusion that the router, not candidate competence or the admission gates, was the failing component.

The same report exposed D20, a reporting shortfall: a canonical run left `extra.self_repair_decoding` null, so the report writer recorded an undeclared provenance gap and a sealed rule failed. The report-writer fix closes or declares that value for **future runs only**. It does not widen the permitted gap set, edit the sealed specification, or change the EXP-005 result.

## 3. What is falsified

- **EXP-002 H3 is falsified.** Forced application of the adapter broke 46 of 94 prior-passing tasks while routed replay stayed clean. The no-regression observation belongs to router shielding, not to adapter-intrinsic retention.
- **EXP-005 H1 and H2 are falsified for this router and workload.** A competent second candidate was refused because tag/keyword routing claimed base-passing same-domain traffic and caused routed regressions. The pool therefore never reached two active experts.
- The EXP-003 and EXP-004 self-repair measurements are both zero, but both sealed A/B runs are formally **unusable (exit 2)** because no self-repair expert existed to pair. Their zero yield must not be relabeled as a formally graded falsification.
- D20 was a genuine reporting defect in the EXP-005 report. It is being corrected in the report writer for future runs, not retroactively repaired in the sealed record.

The broad Grove thesis is not classified as falsified by these results; it remains unproven and is constrained by them.

## 4. What stands — the control plane's tested claims

The evidence supports these narrower engineering claims:

- Verified failures, corrections, task roles, candidate artifacts, evaluation results, deployment manifests, and audit events are persisted through the lifecycle.
- Admission gates can reject weak candidates and can refuse a competent candidate whose router behavior would regress already-passing traffic. A rejected candidate does not become routable.
- An admitted expert is a separate artifact that can be removed and restored through deployment state without editing the frozen base or another expert.
- The router's recall can be measured with an oracle-free probe, and routed replay can expose false positives and regressions rather than hiding them behind an aggregate score.
- Sealed specifications bind reports to a declared version and predeclared rules. Paired runs refuse mismatched or unresolved critical identity, including worker source content and model files, instead of silently comparing unlike arms.
- Partial provenance is reported as a named gap rather than hidden. More narrowly, every value read by the declared decision rules is bound to a locally stored run manifest, and an edit to a bound value is detected against that manifest. This is local integrity detection, not tamper-proofing: rewriting the report, manifest, and digest together is outside the control plane's claim.
- Capacity and setup preflight can refuse a run before it spends on a sandbox or worker when the declared prerequisites are not met.

These claims are about a controlled coding workload and the tested control plane. They are not a claim of broad continual learning.

## 5. Open questions

1. **Router discrimination:** can a router at this representational budget distinguish same-domain failure families without claiming traffic the base already solves?
2. **Self-generated corrections:** does a stronger proposer or a near-miss failure family produce any verifier-approved self-repairs, or is yield effectively floored for this model and workload?
3. **Honest routing at scale:** what happens to shielding, interference, and capability when a deployment actually contains multiple experts? This remains blocked by the absence of a two-expert pool.
4. **Transfer beyond paraphrases:** can a later cycle learn genuinely new algorithms rather than only nearby wording and operations? The EXP-005 diagnostic was negative for `path_project` and `path_rename`.
5. **Research controls:** repeat configurations across seeds, create fresh holdouts per cycle, and compare against a full-fine-tune baseline.
6. **Non-deterministic domains:** can the lifecycle use judges or user feedback where deterministic verifiers are unavailable?
7. **Persistent serving:** what are the operational and interaction costs of a resident base with hot-swapped adapters, including router switches that require a fresh prefill?

## 6. Limitations

> **Preregistration timing is not externally verified.** The seals are self-consistent, but they are **not externally timestamped**. Grove has no timing verifier at all, so a preregistration claim is refused rather than accepted. A sealed spec proves which declaration a report is bound to; it does not prove when that declaration existed.

- This is a narrow, manually scheduled coding workload with deterministic Python verifiers, one frozen base model, and one admitted expert family. It is not a production continual-learning service or evidence about arbitrary agent work.
- Routing uses auditable tags and keywords rather than learned embeddings. The measured single-expert precision and recall do not establish broad routing behavior, and the pool has never held two simultaneously active experts.
- The first positive result has only four original holdouts, a small regression suite, and one failure family. The fresh verification strengthens the within-family result but does not establish general algorithmic transfer.
- The self-repair result is a measurement on this model, verifier suite, and workload. EXP-003 and EXP-004 could not grade their intended A/B hypotheses because their primary arms produced no expert.
- Worker execution is not fully independently reproducible: the records contain local provenance and model identities, but no worker-side signed attestation. The local manifest checks are not tamper-proof, and no external timestamp or signed/transparency-logged manifest anchors the run.
- LXD shares the host kernel and is not a hardware security boundary for hostile public submissions.
- The sealed spec JSONs under `experiments/` retain the operator's absolute local report and database paths under `/srv/storage/...`. `for f in experiments/EXP-00*.json; do printf '%s: ' "$f"; grep -o '/srv/storage' "$f" | wc -l; done` reports **3 literal occurrences in EXP-002, 6 in EXP-003, 6 in EXP-004, and none in EXP-005**; `for f in experiments/EXP-00*.json; do printf '%s: ' "$f"; grep -o '/srv/storage[^\" ]*' "$f" | sort -u | wc -l; done` reports **2, 4, 4, and 0 distinct absolute paths**, respectively (one evaluation report and one run database for each referenced run or arm). They remain byte-identical because scrubbing those paths would break the sealed spec hashes. Those paths describe the original operator layout; they are not portable paths for a new checkout.
- The repository does not claim long-horizon storage behavior, multi-expert pruning, non-deterministic verification, a resident serving process, or a fully fine-tuned comparison until those questions are measured.
