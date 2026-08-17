# Independent verification of the 2026-07-31 experiment

Date: 2026-08-01. Performed independently of the original run, using
`scripts/independent_probe.py`. Raw results:
`docs/data/independent-probe-2026-08-01.json`. No original evidence was
modified; all probes ran with fresh verifier instances and no store writes.

## Forensic checks (all passed)

- SHA-256 of `grove-real-v3.db`, `final-real-cycle.json`, the training
  dataset, and `adapters.safetensors` match the experiment report exactly.
- The adapter is a real MLX LoRA artifact: 112 tensors, exactly 2,637,824
  parameters, and every checked `lora_b` matrix is fully non-zero (LoRA B
  initializes at zero, so real gradient updates occurred).
- The Mac worker is real: `scripts/preflight.sh` completed a live SSH
  round-trip to `grove-worker-1.local` (arm64, macOS 26.4, MLX 0.32.0).
- The LXD image `grove-python-base` exists with the reported fingerprint.
- The audit ledger (89 events) is internally coherent and its timestamps
  match artifact mtimes.
- The rejected candidates from Attempts A–C are preserved in
  `grove-real.db` / `grove-real-v2.db` with the metrics the report claims.
- `docs/original/` matches commit `86c3f53` byte-for-byte.
- Full test suite: 25/25. Deterministic demo lifecycle reproduces the
  README curve (57.1% -> 100%).

## Live reproduction of the headline

Base vs. admitted adapter on the four original holdouts, regenerated and
re-verified from scratch:

| Task | Base | Adapter |
|---|---|---|
| holdout_path_lookup | fail | pass |
| holdout_path_membership | fail | fail (2/3 cases) |
| holdout_path_assign | fail | pass |
| holdout_path_flatten | fail | pass |

Identical to the reported 0/4 -> 3/4, including the same single failure.
Deterministic decoding reproduces exactly.

## New probe 1: fresh holdouts the admission gate never saw

Six newly written paraphrases with newly written hidden cases (expected
values computed from the canonical reference solutions, not by hand).
Neither training nor the admission gate ever observed these prompts or
cases, so unlike the original holdouts they could not have influenced
candidate selection.

| Task (operation) | Base | Adapter |
|---|---|---|
| fresh_lookup (get) | fail | pass |
| fresh_member (exists) | fail (1/3) | pass |
| fresh_assign (set) | fail | pass |
| fresh_remove (delete) | fail | fail |
| fresh_flatten (flatten) | fail (1/3) | pass |
| fresh_expand (unflatten) | fail | pass |

**Base 0/6 -> adapter 5/6.** The transfer to unseen wordings and unseen
cases is real and not an artifact of gate-set selection. The one failure
is deletion — the same operation the original report identified as the
residual weak cluster (`path_delete_v2/v3`). The passing solutions are
the memorized canonical `_split` algorithm correctly re-applied, i.e.
genuine but narrow capability.

## New probe 2: forced-adapter forgetting (not measured originally)

The original 0%-forgetting claim tested the *deployed configuration*,
where the router shields the adapter from out-of-family prompts. This
probe forces the adapter on for the regression tasks, measuring the
adapter itself:

| Task | Base | Adapter forced on |
|---|---|---|
| reg_sum_even | pass | pass |
| reg_dedupe | **pass** | **fail** |
| reg_run_lengths | fail | fail |
| reg_balanced | fail | fail |

`reg_dedupe` is a capability regression: the adapter hijacks a plain
list-deduplication prompt into its escaped-path groove (it emits `_split`
boilerplate and treats the input as paths, then crashes). Router analysis
confirms the deployed system never saw this because only `reg_balanced`
(prompt contains the keyword "nested") routes to the expert — and the
base fails that task anyway.

**Restated claim:** the adapter does interfere destructively outside its
family. "No forgetting" is a property of the router shielding, not of the
adapter. This empirically confirms the report's own caveat that router
breadth and replay coverage are the weak points.

## Future stream (re-confirmed)

`path_rename` 1/3 cases, `path_project` 0/2 with the adapter — matching
the report. No generalization to new algorithms.

## Verdict

- Everything the report records physically happened; nothing was
  fabricated, and the result reproduces exactly under re-execution.
- The capability gain is *more* credible than the original report could
  claim, because it survives evaluation on holdouts created after the
  fact (5/6 vs the gate-tainted 3/4).
- The zero-forgetting claim must be weakened: the adapter measurably
  damages at least one previously working behavior when the router does
  not shield it. Any future cycle that broadens routing (or learns it)
  must re-measure forgetting with the adapter forced on.
- Known validity limits stand: training targets were human-written
  canonical solutions (failures only selected what to train on), and
  transfer is within-family paraphrase robustness, not new algorithms.

## Rollback report artifact audit (2026-08-07)

The named `/srv/storage/grove/evaluations/final-real-cycle.json` contains stale
top-level `rollback.active_experts` and `rollback.added_parameters` values
(`1` and `2637824`). The authoritative SQLite evaluation is labeled
`rollback_drill_corrected` and records `0` and `0`. Use
`scripts/audit_evaluation_report.py` to compare a report with the database;
the corrected, annotated copy is preserved at
`docs/data/final-real-cycle-annotated-2026-08-07.json`. The original external
report remains read-only evidence and should not be cited for those two fields.

**Corrected 2026-08-07 (second pass): that annotation is not verified evidence.**
The four evaluation rows in `/srv/storage/grove/grove-real-v3.db` were written
before digest recording existed, so none of them carries a `metrics_sha256`.
Running the audit against that database today returns
`integrity.status: unverified` with `checked: 0` and exits 2. The annotated file
is preserved unchanged as historical reconciliation, and its own
`rollback_audit_correction` block still calls the SQLite row "authoritative" —
wording written before the three-state distinction existed and no longer
defensible. See `docs/data/final-real-cycle-annotated-2026-08-07.STATUS.md` for
the adjacent status record. Do not cite either file as a clean or tamper-evident
record.
