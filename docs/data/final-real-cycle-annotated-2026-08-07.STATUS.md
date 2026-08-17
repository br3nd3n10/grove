# Status record for `final-real-cycle-annotated-2026-08-07.json`

**Historical reconciliation only. Evaluation integrity is unverified.**

This adjacent record exists because the annotated artifact cannot be corrected
without destroying it. The file is preserved byte-for-byte as it was written on
2026-08-07; this note supersedes the wording inside it.

## What the artifact is

`docs/data/final-real-cycle-annotated-2026-08-07.json` is a copy of the
2026-07-31 real-cycle report with a `rollback_audit_correction` block appended.
The block reconciles two stale top-level fields — `rollback.active_experts` and
`rollback.added_parameters` — against the SQLite evaluation row
`eval_187c1f0faf5c` in `/srv/storage/grove/grove-real-v3.db`.

## Why it is not verified evidence

Its `rollback_audit_correction.note` says the SQLite evaluation record is
"authoritative for rollback metrics". That was written before Grove
distinguished verified evidence from unchecked evidence, and it is no longer
defensible.

The four evaluation rows in that database were written before digest recording
existed. None carries a `metrics_sha256`, so nothing can establish that any of
them still says what it said in 2026-07-31. Running the audit today returns:

```text
integrity.status: unverified
integrity.checked: 0
integrity.authoritative: false
exit 2
```

A digest computed now cannot recreate evidence about then, so backfilling one
would make the record look verified without making it verified. The rows are
left exactly as they are.

## How to cite it

- **May** be cited as: the historical reconciliation of two stale fields,
  produced on 2026-08-07 against a database that predates digest recording.
- **May not** be cited as: a clean record, a tamper-evident record, or
  authoritative evidence for any metric.

Reports produced by the current runner carry a `run_manifest` and per-row
digests, and verify as `clean` when intact. This artifact predates all of that.
