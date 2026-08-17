# EXP-005: second growth cycle, two-expert coexistence — H1 and H2 FALSIFIED (exit 1)

**Date:** 2026-08-11
**Spec:** `experiments/EXP-005-second-cycle-coexistence.json` (sealed `e6867048…ecd733`, spec intact, report bound)
**Report:** `/srv/storage/grove/evaluations/exp005-second-cycle.json`
**Checker:** exit **1** — `predeclared rule(s) failed`. 29 rules: 23 passed, **6 failed**, 0 unevaluable. Falsified hypotheses: **H1, H2**. Not unusable.

## Bottom line

The system's first plural did not hold. The second cycle trained a competent
candidate on the new `path_restructure` family — target fix 17/20, held-out 3/4
at the sealed 0.75 bar — and then the unchanged admission policy **rejected
it**: with expert 1 deployed, the router claimed 18 of 112 base-passing replay
tasks for the candidate (false-positive rate 0.161 against a sealed budget of
0.0) and 12 of them regressed under routing (0.107 against a budget of 0.0).
Recorded rejection reason: *"routed regression budget exceeded; router claims
base-passing replay traffic."* The run ended with one active expert, not two.

That is a measured falsification, not a bug and not an infrastructure failure:

- Worker `status: ok` (`grove-worker-1`, arm64, mlx 0.32.0, mlx_lm 0.31.3, clean checkout `323a6a3`); both training jobs completed (34.7 s, 44.1 s).
- Sandbox probes clean (LXD 6.9, image `grove-python-base`, 1 launch attempt, no queue).
- Both live captures graded: 123 tasks (94 passed) before cycle 1, 143 tasks (112 passed) before cycle 2.
- Ledger integrity clean (272 records verified), evaluation integrity clean (4/4), every manifest binding intact, `required_setup_check.satisfied = true`, setup conformance `conformant: true`.

The gates did their job on live evidence: a candidate that met every
competence bar was refused deployment because the *deployed system* around it —
specifically the router — could not keep it off traffic the base already
handles. The failure mode EXP-002 identified in one direction (router shielding
hides adapter forgetting) reappears here in the other (router over-claiming
blocks coexistence).

## Second family and data provenance

- **Family:** `path_restructure` — dict-restructuring operations (`restruct_copy`, `restruct_drop`, `restruct_move`, `restruct_pick`, × 5 variants each = 20 training failures, set sha `bf3a06b5…`).
- **Corrections:** canonical human references (`correction_source: canonical`, confirmed in run_setup and provenance; F3/F3b passed). Self-repair was not under test — EXP-003/EXP-004 measured its 0/20 floor twice.
- **Held-out:** four fresh `path_restructure` prompts authored for this experiment, never read by any prior gate (sealed prose rule).
- **Disjointness:** every `path_restructure` training and held-out prompt is disjoint by content hash from every cycle-1 prompt; the store's leakage rejection enforces it at role-assignment time (sealed prose rule).
- Verifier suite `escaped-path-v2+python-core-v1+path-restructure-v1` (153 tasks); base model `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit@b3252a2f`; worker model manifest `25d34506…` (same digest as EXP-004).

## Checkpoint metrics (fixed 111-task cohort union)

| Metric | Baseline | After cycle 1 | After cycle 2 | Single-expert reference |
|---|---:|---:|---:|---:|
| Active experts | 0 | 1 | **1** (spec expected 2) | 1 |
| Capability | 0.8468 (94/111) | 0.8739 (97/111) | 0.8739 (97/111) | — |
| Capability delta | — | +0.0270 | **0.0000** | — |
| Replay examples (prior-passing) | 0 | 94 | 112 | 94 |
| Routed replay regression | — | 0.0 | 0.0 | 0/94 |
| Route precision / recall | — | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Route false-positive rate | — | 0.0 | 0.0 | 0.0 |
| Cycle-1 training targets in replay | 0 | 0 | 18 (≥ 16 required) | — |
| Added parameters | 0 | 2,637,824 | 2,637,824 | 2,637,824 |

The after-cycle-2 column matches the single-expert baselines (route
precision/recall 1.0/1.0; routed replay clean, now 112/112 vs 94/94) **because
it is still a single-expert deployment**: expert 2 was rejected before it could
ever be routed to in production. Capability's cycle-2 delta is exactly zero —
the cycle banked nothing. Rollback drill returned cleanly to 0 experts at
baseline capability 0.8468.

## Decision rules: 23 passed, 6 failed

| Rule | Hyp | Path | Threshold | Measured | Outcome |
|---|---|---|---|---|---|
| F1–F4c (7 rules) | — | setup: cycles=2, family, canonical source ×2, greedy temp ×3 | as sealed | all as sealed | pass |
| D5 | H1 | second_cycle.experts_admitted | count ≥ 1 | **0** | **FAIL** |
| D6 | H1 | after_cycle_2.active_experts | == 2 | **1** | **FAIL** |
| D7 | H1 | path_restructure heldout_forced_rate (deployed) | ≥ 0.75 | **none** (never deployed) | **FAIL** |
| D8 | H2 | combined route_recall | ≥ 0.5 | 1.0 | pass |
| D8a | H2 | escaped_path route_recall | ≥ 0.5 | 1.0 | pass |
| D8b | H2 | path_restructure route_recall (deployed) | ≥ 0.5 | **none** (never deployed) | **FAIL** |
| D9 | H2 | two-expert route_precision | ≥ 0.8 | 1.0 (one-expert pool) | pass |
| D10 | H2 | route_false_positive_rate | ≤ 0.0 | 0.0 | pass |
| D11 | H3 | replay_examples | ≥ 50 | 112 | pass |
| D12 | H3 | cycle-1 targets in replay | ≥ 16 | 18 | pass |
| D13 | H3 | replay_regression_rate | ≤ 0.0 | 0.0 | pass |
| D14 | H3 | expert 1 heldout routed | ≥ 0.75 | 0.75 | pass |
| D14b | H3 | expert 1 heldout forced | ≥ 0.75 | 0.75 | pass |
| D15 | H4 | capability delta cycle 1 | ≥ 0.0 | +0.0270 | pass |
| D16 | H4 | capability delta cycle 2 | ≥ 0.0 | 0.0 | pass |
| D17 | H4 | added_parameters | > 0 | 2,637,824 | pass |
| D18 | H4 | added_parameters | ≤ 6,000,000 | 2,637,824 | pass |
| D19 | — | expert 1 forgetting_claim | not `unmeasured` | `router_shielded` | pass |
| D19b | — | expert 2 forgetting_claim (deployed) | not `unmeasured` | **`unmeasured`** | **FAIL** |
| D20 | — | provenance_gaps | ⊆ [models.base.aggregate_sha256] | + **`extra.self_repair_decoding`** undeclared | **FAIL** |
| D21, D22 | — | worker model manifest, run manifest | present | present | pass |

D7, D8b and D19b fail as `none`/`unmeasured` because they read the *deployed*
after-cycle-2 state, and the rejected expert never entered it. The candidate's
probation-time values exist (below) but are not what the sealed rules grade.
D20 is a genuine reporting shortfall independent of coexistence: this canonical
run left `self_repair_decoding` null and recorded it as a provenance gap the
spec had not declared permissible. It stands as a failed rule.

## Both experts' admission metrics

| Metric | Expert 1 `expert_188deaea529c` (escaped_path, cycle 1) | Expert 2 candidate `expert_bb821fb6d00f` (path_restructure, cycle 2) | Gate |
|---|---:|---:|---|
| Status | **active** (admitted) | **rejected** | — |
| target_after / before | 0.90 / 0.00 (20 targets) | 0.85 / 0.00 (20 targets) | ≥ 0.80 |
| plasticity_gain | 0.90 | 0.85 | ≥ 0.50 |
| heldout_target_rate (forced) | 0.75 (3/4) | 0.75 (3/4) | ≥ 0.75 |
| route_recall | 1.00 (3/3) | 1.00 (3/3) | ≥ 0.50 |
| route_precision | 1.00 | **0.143** (3 of 21 routed) | cohort-dependent |
| route_false_positive_rate | 0.00 (0/94) | **0.161** (18/112) | ≤ 0.00 |
| routed regression_rate | 0.00 (0/94) | **0.107** (12/112) | ≤ 0.00 |
| routed replay pass rate | 1.00 | 0.893 | — |
| forced_replay_rate | 0.511 | 0.357 | informational |
| forced_regression_rate (vs bare base) | 0.489 (46/94) | 0.585 (base_replay_pass 0.839) | ≤ 1.00 |
| forgetting_claim (probation) | router_shielded | regression | — |
| Parameters / training time | 2,637,824 / 34.7 s | 2,637,824 / 44.1 s | — |

The candidate's competence gates all passed; the deployment gates all failed.
The 12 tasks it regressed under routing are all base-passing `path_*` tasks
(`path_get`, `path_exists`, `path_set`, `path_flatten`, `path_unflatten` and
variants) — same-domain traffic the router handed to an expert trained only on
`restruct_*` prompts. The router's tag/keyword discrimination cannot separate
"dict-path task the base solves" from "dict-restructure task the expert fixes."

## Interference and forced probes per expert (after cycle 2)

- **Expert 1 (escaped_path, deployed):** held-out re-measured **routed 0.75** and **forced 0.75** — both exactly at the admission bar, unchanged. Forced regression vs bare base still 0.489 (46/94); `forgetting_claim: router_shielded` (consistent with EXP-002). Route recall 1.0. No measured interference — but nothing new was deployed next to it, so this passes over a one-expert pool, not a two-expert one.
- **Expert 2 (path_restructure):** no deployed-state probe exists (`heldout_forced_rate: none`, `forgetting_claim: unmeasured`) because rejection precedes deployment. Probation-time forced probes: held-out 0.75, forced replay 0.357, forced regression 0.585.
- **Future probe (diagnostic only, no sealed rule):** `path_project` 0.0 fail, `path_rename` 0.33 fail — both routed to expert 1. No measured transfer to the archived cycle-1 future prompts.

## What remains unproven

- **Two-expert coexistence has now been attempted once and falsified once.** H1 (second expert admitted, both active) and H2 (two-expert route discrimination) are falsified for this router on this workload. The specific mechanism is measured: keyword/tag routing over-claims base-passing same-domain traffic (precision 0.143, FP 0.161) and the sealed zero-regression budget correctly refuses that deployment.
- **H3 (no interference) and H4 (monotone capability) passed, but weakly.** Both were measured over a final pool that never contained two experts, because rejection kept it at one. They demonstrate the *gates'* safety property — a bad coexistence deployment was refused and nothing regressed — not two-expert coexistence itself.
- **The Grove thesis is not proven**, and no outcome of this single experiment could have proven it. The strongest positive evidence remains the single narrow router-dependent expert; the system has still never had two simultaneously active experts.
- Monotone capability growth *through admitted experts* stalled at cycle 2 (delta exactly 0.0): capability did not decrease, but the cycle added nothing.
- D20's undeclared provenance gap (`extra.self_repair_decoding`) is a reporting defect to fix in the report writer for future runs — by declaring or closing the gap, not by widening the permitted set post hoc for this one.
- Preregistration timing remains unverified (seal is self-consistent, not timestamped).

The constructive question this run sharpens, without touching any gate: can a
router discriminate two same-domain families at deployment-grade precision at
all, at this router's representational budget (tags and keywords)? The
candidate was good enough; the router was the falsified component.
