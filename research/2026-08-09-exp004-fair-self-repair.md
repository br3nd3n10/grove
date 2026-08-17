# EXP-004: fair self-repair A/B — checker verdict UNUSABLE (exit 2), self-repair yield 0/20 again

**Date:** 2026-08-09
**Spec:** `experiments/EXP-004-fair-self-repair-ab.json` (sealed `f129e49a…`, spec intact, both reports bound)
**Reports:** primary `/srv/storage/grove/evaluations/exp004-self.json`, control `/srv/storage/grove/evaluations/exp004-canonical.json`
**Checker:** exit **2** — `arms could not be paired: rule D6d: primary delta path experts[*].metrics.heldout_target_rate is absent`. 21/21 rules unevaluable, 0 failed, 0 passed.

## Bottom line

The fair regime ran exactly as sealed and still produced **zero verified self-repairs: 0/20 failures repaired in 160 sampled, seeded generation calls**. With no verified corrections, the primary arm trained no expert, so it carries no pairing key, the delta rules D6d/D6e cannot pair the arms, and the checker refuses the whole run as unusable — the outcome the spec's own limitations preregister for unpairable arms ("unusable (exit 2), not falsified"). Formally this run confirms nothing and falsifies nothing. The measured yield is still a real measurement, and it is the second consecutive zero.

This was **not** an infrastructure failure. Both arms measured:

- Sandbox probes clean in both arms (LXD 6.9, image `grove-python-base`, 1 launch attempt, no queue).
- Live capture graded 123 tasks per arm (94 passed / 29 failed), identical 20-task `escaped_path` training-failure set in both arms (set sha `22d83c46…`).
- Both arms loaded the same model files: worker `model_manifest_sha256 = 25d34506…` in both; base model `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit@b3252a2f`.
- Ledger integrity clean (143 records verified), evaluation integrity clean, run manifests present, `required_setup_check.satisfied = true` in both arms, setup conformance `conformant: true`.
- Every self-repair rejection has a concrete verifier reason (below); no worker or sandbox errors appear anywhere in either report.

## Fair-regime parameters actually used (verified in both reports' `run_setup`)

| Parameter | Value | EXP-003 value |
|---|---|---|
| Repair attempts per failure | 8 | 3 |
| Repair decoding temperature | 0.8 (sampled) | 0.0 (greedy — all attempts near-identical) |
| Repair max_tokens | 768 | — |
| Base seed | 20260809; per-attempt seeds derived `sha256(f"{base_seed}:{task_id}:{attempt}")`, recorded in every proposal (e.g. `path_get`: 1316874798, 1084715577, …) | none |
| Repair feedback | allowlisted honest verifier detail (prose key, mechanically unchecked) | one generic failure sentence |
| Baseline / held-out / replay decoding | greedy, temperature 0.0 (confirmed per-purpose) | same |

## Per-arm verified-correction yield (identical `correction_comparison.per_source` in both arms)

| Source | Failures | Verified | Rate | Generation calls | Mean attempts |
|---|---|---|---|---|---|
| `canonical-reference-v1` | 20 | 20 | **1.00** | 0 | 1.0 |
| `self-repair-v1` | 20 | 0 | **0.00** | 160 | 8.0 |

Coverage: all 20 tasks in `only_canonical-reference-v1`; `both` and `only_self-repair-v1` empty.

Self-repair rejection reasons (20 proposals, 0 accepted): 14 × "candidate exited with status 1", 5 × "one or more hidden cases failed", 1 × "candidate did not produce one valid JSON result". Every attempt reached the sandbox and was graded; none passed the full hidden suite.

**Against EXP-003's 0/20:** EXP-003 spent 60 greedy calls (3 near-identical attempts × 20) with generic feedback and got 0/20. EXP-004 spent 160 sampled, seeded calls with honest feedback and got 0/20. Zero successes in 160 roughly independent sampled attempts puts an approximate 95% upper bound (rule of three) of ~1.9% on the per-attempt repair probability — below the ~3.5% the spec computed as necessary to reach the 0.25 bar under 8 attempts. The fair regime did not lift yield above zero; it tightened the evidence that Qwen2.5-Coder-1.5B-4bit has essentially no repair signal on these tasks.

## Expert admission and metrics

**Primary (self-repair):** 0 candidates trained, 0 experts. Both clusters skipped: "not enough verifier-backed corrections to train safely." `after_growth.active_experts = 0`.

**Control (canonical):** 1 expert trained and admitted (`expert_18b699abbd91`, status active), all gates measured:

| Metric | Measured | Gate |
|---|---|---|
| heldout_target_rate | 0.75 (3/4 holdout tasks) | ≥ 0.75 |
| target_after / target_before | 0.90 / 0.00 (20 targets) | ≥ 0.80 |
| plasticity_gain | 0.90 | ≥ 0.50 |
| route_recall / route_precision | 1.00 / 1.00 | ≥ 0.50 / — |
| route_false_positive_rate | 0.00 | ≤ 0.00 |
| regression_rate (routed, 94 replay) | 0.00 | ≤ 0.00 |
| forced_regression_rate | 0.489 (46/94, base_no_experts reference) | ≤ 1.00 |
| replay_pass_rate routed / before | 1.00 / 1.00 (94 examples ≥ 50 min) | — |
| capability | 0.8785 → 0.9065 | — |

## Decision rules: all 21 unevaluable (checker refuses every rule once pairing fails)

Observed values the checker recorded before refusing, against their thresholds:

| Rule | Path | Threshold | Observed | Note |
|---|---|---|---|---|
| D1 | correction_source | == self-repair | self-repair | would satisfy |
| D2 | comparison sources | set== {canonical-reference-v1, self-repair-v1} | both present | would satisfy |
| D3 (H1) | self-repair verified_rate | ≥ 0.25 | **0.0** | below bar |
| D4 | provenance correction_source | == self-repair | self-repair | would satisfy |
| D5 (H2) | primary experts_admitted | count ≥ 1 | **0** | below bar |
| D5c | control correction_source | == canonical | canonical | would satisfy |
| D6 (H2) | primary heldout_target_rate | ≥ 0.75 | no experts — no values | absent |
| D6c (H2) | control experts_admitted | count ≥ 1 | 1 | would satisfy |
| D6d (H2) | heldout delta vs control | ≥ −0.1 | primary path absent | **the blocker** |
| D6e (H2) | forced_regression delta | ≤ 0.0 | primary path absent | unevaluable |
| D7 | provenance_gaps | ⊆ [models.base.aggregate_sha256] | exactly that | would satisfy |
| D8 | expert pairing_key exists | exists | present (vacuous in primary) | — |
| D9 | run_manifest_sha256 | exists | present | would satisfy |
| D10 | training_proposal_reuse | == true | true | would satisfy |
| D11 | worker model_manifest_sha256 | != null | 25d34506… | would satisfy |
| F1 | self_repair_attempts | == 8 | 8 | would satisfy |
| F2 | repair temperature | == 0.8 | 0.8 | would satisfy |
| F3 | base_seed | == 20260809 | 20260809 | would satisfy |
| F4 / F4b | eval / heldout temperature | == 0.0 | 0.0 / 0.0 | would satisfy |
| F5 | self-repair generation_calls | exists | present (160) | would satisfy |

Per the checker's semantics, none of these is graded: `rules_failed = 0`, `rules_unevaluable = 21`, hypothesis outcomes null.

## What remains unproven

- **H1 (useful self-repair rate) has no formal outcome.** The measured rate was 0.0 against a 0.25 bar, but the checker's pairing precondition refused grading before any rule was scored. Note the structural coupling this exposes: a zero yield guarantees no primary expert, which guarantees unpairable arms, which guarantees exit 2 — so under this checker a zero-yield run can never record H1's falsification, even though the spec's limitation text calls a sub-0.25 measured rate with intact arms "a falsification of H1, which is a result." A future sealed spec could grade H1 independently of arm pairing; that changes no gate and no threshold.
- **H2 (self-trained expert parity) remains untested**, now twice: there has never been a self-repair-trained expert to compare. The canonical arm shows the pipeline can train and admit an expert from these 20 failures; the self-repair arm shows this model cannot feed that pipeline.
- **The Grove thesis is not proven** by any outcome here, and would not have been by a pass. This experiment bears only on correction provenance for one small model on one task family, with a verifier-suite ceiling and deliberately asymmetric inference budgets (160 calls vs 0), all preregistered.
- Preregistration timing is unverified (seal is self-consistent, not timestamped); provenance carries the one permitted gap (`models.base.aggregate_sha256`).

Two experiments, two regimes (3 greedy attempts, then 8 sampled seeded attempts with honest feedback), one result: 0/20 and 0/20. Whatever verified self-repair yield this model has on these tasks, neither regime found it.
