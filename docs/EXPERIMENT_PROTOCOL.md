# Predeclared experiment protocol

Written 2026-08-07, in response to the 2026-08-06 audit
(`fan-out-and-synthesize-a1c80e6c` / `synthesis.md`), whose verdict was
**"promising but under-proven"**: the narrow vertical slice is credible, the
broader continual-learning thesis is not established.

This document exists so the next claim is decided by a rule written before the
run, not by a narrative written after it. Nothing here asserts the thesis is
proven. A 2026-08-07 review of the first pass found three ways the new
measurements could still mislead -- net pass-rate cancellation, an empty cohort
reported as stable, and circular route recall -- plus a spec seal that was not
bound to the run. Those are fixed below; the honest status of each risk follows.

## What the audit said, and what changed

| Audit finding | Status after 2026-08-07 |
|---|---|
| §3 "No forgetting" is router-shielded | **Measured, not solved.** Every probation runs replay twice: routed and with the candidate forced on. Forgetting is counted **per task** — denominator is the replay tasks that passed in the reference run, numerator is how many of those the candidate breaks — so a newly fixed task cannot offset a broken one. The forced comparison uses the **bare base** whenever one exists (`forced_regression_reference`), because comparing against a previous deployment that already contains experts answers a different question. `forgetting_claim` now has five values: `unmeasured`, `regression` (the deployed system itself regressed — this outranks any shield story), `router_shielded`, `unverified_reference` (clean, but only against a routed reference), and `adapter_intrinsic`, which alone licenses a plain "no forgetting" sentence and requires both a non-empty denominator and a bare-base reference. |
| §3 Router breadth is the weak point | **Gated, on independent evidence.** Route recall is scored on held-out tasks the candidate demonstrably solves when forced on, with gold family tags and metadata stripped from the routing input (`route_positive_source: heldout_forced_pass_oracle_free`, `route_probe_metadata: oracle_free`), never on the training cluster that produced the routing profile — that denominator was circular, and so was the family tag itself. The gold-tag number survives as `route_recall_gold_tags` with `route_recall_gold_tags_independent: false`; it is a diagnostic and gates nothing. Route false-positive rate is measured separately on prior-passing replay. Route **precision is no longer a gate and no longer accepted as a policy field**: it mixes the two cohorts, so it moved from 0.75 to 0.60 purely by adding a replay task while the router was unchanged. `SleepPolicy(min_route_precision=...)` raises `TypeError`, and a spec declaring it exits 2. Replay negatives come from each task's *latest* attempt and exclude the candidate's own failure family. |
| §4 Evidence base is too small | **Not closed, and now visibly blocking.** `EXP-002` predeclares at least 50 *prior-passing* replay tasks and `REAL_CYCLE_POLICY` enforces the same minimum. The catalog supplies at most 24 captured tasks, so a capacity preflight rejects a real cycle before the sandbox or worker is touched rather than letting every candidate fail the denominator gate after training was paid for. The refusal prints one `setup_refused` document and exits 2. NIST TN 2045 puts 50 in perspective: a true rate of 0.9 is accepted with probability 0.27 at n=25 and 0.88 at n=100, so this cohort detects gross regression, not a tight interval. |
| §5 Benchmark purity | **Procedurally improved, not solved.** Specs are sealed, the run records the digest it was launched under, and the run's actual setup is recorded and compared against the spec's `required_setup` — a hash proved which declaration a run named, not that the run obeyed it. `required_setup` splits into `machine` and `prose`: a missing machine key is a mismatch and exits 2, while prose declarations are listed as `unchecked_prose_keys` and never counted as conformance. A sealed spec with no rules or no falsification conditions is rejected outright. The cohort design is still inherited and post-hoc. |
| §6 Corrections were human-written | **Alternative implemented, comparison now enforced.** The need for a control arm is **derived from the decision rules** — any `delta*` comparison, `control_path`, `pair_on`, or `arm: control` — not from the optional `requires_control_report` flag, which could previously be removed to switch pairing off entirely. Missing control → exit 2 with the affected rules marked `unevaluable`, never a falsified hypothesis. A paired spec must also declare a machine-checkable `control_required_setup`; an absent one used to read as conformant. Arm identity covers source revision, tree, dirty state, status and worktree **content** digests, worker host, worker framework digest, worker checkout revision/tree/dirty/status/worktree, worker model manifest, sandbox fingerprint, base-model digest, admission-policy digest, cohort manifest, self-repair attempts and the **actual** training failure set; a null or unresolved value on either side is a mismatch. Deltas are evaluated for **every** expert, paired on the stable `experts[*].pairing_key` derived from the cluster label and actual failed training task ids — never the random UUID `experts[*].id`, which cannot match across runs. D6e's `delta>= -1.0` was vacuous for rates in [0,1] and is now `delta<= 0.0`. `--compare-corrections` reuses its proposals for training and reports `generation_calls` per phase. |
| §8 Reproducibility metadata incomplete | **Collectors expose unresolved fields explicitly; provenance is partial by declaration, and the permission is per spec.** Every required section is emitted whether or not it resolved. An empty model directory no longer hashes to the SHA-256 of nothing. A failed `git status` reports unavailable and adds `source.dirty` to the gaps. EXP-002 is single-arm and permits two gaps, `models.base.aggregate_sha256` and `worker.model_manifest_sha256`. EXP-003 is paired and permits **one**: it lists `provenance.worker.model_manifest_sha256` under `required_resolved_identity`, because that digest is the only evidence both arms loaded the same weights. Passing a gap rule means **partial provenance**, never complete provenance. A missing provenance, evaluation or ledger digest is reported as `unverified` and can never yield clean evidence. |
| §8 / conflict 2 Stale rollback artifact | **Annotated, not repaired, and now labelled unverified.** The original `/srv/storage/grove/evaluations/final-real-cycle.json` remains stale by design — it is read-only external evidence. The audit accepts an exact evaluation or run id from the report and requires an exact database match; without one it falls back to label-and-recency and labels the verdict **unbound evidence**. It also recomputes each stored evaluation digest: a row that disagrees is `tampered`, a row nobody hashed is `unverified`, and both exit 2. The 2026-07-31 rows predate digest recording, so `docs/data/final-real-cycle-annotated-2026-08-07.json` is **historical reconciliation only** and must not be cited as a clean or tamper-evident record. |
| §7 Longitudinal curve netted out forgetting | **Fixed.** `evaluate()` now persists per-task outcomes at each checkpoint and `curve()` counts baseline passes that later fail. A `[pass, fail]` to `[fail, pass]` transition reported `forgetting: 0.0`; it now reports one lost pass. With no passing baseline task the value is `None`, not zero. |
| Conflict 1 Unstable sandbox timeout test | **Fixed.** `SandboxResult.execution_seconds` measures only the enforced guest window, with the clock stopped at the kill rather than after LXD teardown, so the test bound no longer depends on container launch or host load. This closes the specific timing flake; it does not prove sandbox behaviour on every host. |

## How a predeclared experiment works

1. Write the spec in `experiments/`. It must list `hypotheses` with a
   `falsified_if` clause for each, `preregistered_limitations`, and
   `decision_rules` that map a report path to a comparison and a threshold.
   `required_setup` is split into two blocks:

   ```json
   "required_setup": {
     "machine": {"correction_source": "canonical", "min_replay_examples": 50},
     "prose":   {"replay_authoring_rule": "written without reference to the family"}
   }
   ```

   A **machine** key is compared against the run's recorded setup key by key.
   A machine key the run setup does not record is a mismatch, not a footnote:
   the run refuses before the sandbox, and the checker exits 2. A **prose** key
   is reported as `unchecked_prose_keys` and never counts towards conformance,
   because no program here can verify an authoring rule. A flat legacy map is
   read as entirely machine-checkable, which is the strict reading.
2. Seal it **before** running anything:

   ```bash
   uv run python scripts/check_experiment_spec.py \
     --spec experiments/EXP-002-forced-replay-and-route-precision.json --seal
   ```

   This writes `spec_sha256` over the spec's own content. Sealing an already
   sealed spec exits 2 unless `--reseal` is given.
   `tests/test_experiment_spec.py` fails if any shipped spec is unsealed.
3. Commit the sealed spec. The digest proves *which version* a report ran
   under. It proves nothing about *when* the spec was written; see
   "Timing is a separate claim" below.
4. Run the cycle with `--spec`, so the run binds itself to the seal. A paired
   spec also needs `--arm`, naming which setup profile the run must satisfy:

   ```bash
   uv run grove --db ... real-cycle --reset \
     --spec experiments/EXP-002-forced-replay-and-route-precision.json \
     --arm primary \
     --report /srv/storage/grove/evaluations/exp002.json
   ```

   The seal is verified before any sandbox or model work starts, and
   `experiment_spec: {spec_id, spec_sha256}` is written into the report.
5. Check the report against the spec:

   ```bash
   uv run python scripts/check_experiment_spec.py \
     --spec experiments/EXP-002-forced-replay-and-route-precision.json \
     --report /srv/storage/grove/evaluations/exp002.json
   ```

   Exit `0` = every rule held. Exit `1` = a prediction failed, which is a
   result. Exit `2` = the run cannot be judged at all.

A sealed file alone would only prove self-consistency, and re-sealing would
restore it. The binding recorded in the report is what actually survives:
editing and re-sealing a spec makes every earlier report exit 2.

### Exactly what exits 2

The refusal list is deliberate. Every one of these means *nothing was
measured*, which is different from a prediction that failed.

- the spec was altered after sealing, or declares no hypotheses, no decision
  rules, or a hypothesis with no `falsified_if`;
- the report is not bound to this spec version;
- a declared machine setup key is missing from, or contradicted by, the run
  setup, on either arm;
- the spec declares a setup key no policy field backs, such as
  `min_route_precision`;
- the spec claims preregistration timing, with or without a
  `timing_attestation`, because no verifier for one exists here;
- report integrity fails: the provenance digest, the run-manifest digest, a
  decision-input binding, an expert's metrics or adapter digest, or any other
  manifest-to-report binding does not recompute;
- a spec that sets `requires_report_integrity` gets a report with no manifest,
  or one whose provenance carries no digest at all — absence is `unverified`,
  and unverified evidence cannot grade a run;
- the sealed rules need a control arm and none was supplied, or
  `requires_control_report: false` contradicts those rules;
- a paired spec declares no machine-checkable `control_required_setup`;
- the two arms are not comparable on any identity path;
- the arms cannot be paired at all — a missing, null, duplicated or unmatched
  `pair_on` key. The affected rules are reported as `unevaluable` and are
  **not** counted as falsified hypotheses;
- a decision rule declares an `arm` other than exactly `primary` or `control`.
  A typo such as `"control "` is a schema error, not a silent fallback to the
  primary report;
- an identity value that should be a digest is not one. A `*_sha256` field must
  hold exactly 64 lowercase hexadecimal characters. `false`, `0`, `{}`, `""`,
  an uppercase digest and a `sha256:`-prefixed digest are all unresolved
  identity, not reported identity, and EXP-003's worker model manifest must be
  a valid digest rather than merely present or truthy;
- a delta input is absent, null/unmeasured or non-numeric on either arm. A
  metric nobody measured is unevaluable, never a failed rule;
- any rule input is **non-finite** — `NaN`, `Infinity` or `-Infinity`. `NaN`
  compares false against every threshold and an infinity compares true against
  every threshold, so either one would be published as a decided rule. No
  honest run can report one: sealing a report hashes its metrics and the
  canonical encoder rejects a non-finite float, so a report carrying one was
  edited after the seal. The refusal names the rule, the arm, the path and the
  value, and the command still prints a verdict rather than a traceback;
- an exact rollback selector — `--evaluation-id`, `--run-id`, or the report's
  own `rollback_audit` binding — names a row that is not a rollback evaluation.
  The rollback evidence scope is a **closed vocabulary**, not a pattern:
  `rollback`, `rollback_drill`, `rollback_drill_corrected`, compared ASCII-only
  and case-insensitively after trimming ASCII spaces. Any pattern wide enough
  to admit a future label is wide enough to admit its negation, so a label that
  merely contains the word (`not_rollback`) and a label that starts with it and
  then denies the rollback (`rollback_disabled`, `rollback_not_run`,
  `rollback_cancelled`, `rollback_skipped`) are all refused, as are non-ASCII
  lookalikes and non-string labels. The row stays visible for inspection, but
  it is not `authoritative`, `binding.scope_refused` is `true`, `--annotate`
  refuses it and writes no file, legacy recency selection excludes it, and
  `audit_evaluation_report.py` exits 2. Adding a genuinely new rollback label
  is a deliberate edit to that vocabulary.

Exit 2 means the run is **unevaluable or refused**, not that a hypothesis
failed. When any of the above holds, the checker publishes no scientific
outcome at all: every emitted rule carries `passed: false` **and**
`unevaluable: true`, `rules_failed` is `0`, and `falsified_hypotheses` is
empty. That holds for a blocker found before grading and for one found during
it: a rule already graded and satisfied has its `passed` withdrawn too, because
a satisfied prediction inside a payload that says nothing could be judged is
the same overclaim as a falsified one, only in the flattering direction. A run
the checker says it cannot judge may not also report a falsified hypothesis.

### The control arm is required by the rules, not by a flag

`control_requirement(spec)` reads the sealed decision rules and returns the
ones that need a control: any `delta*` comparison, any `control_path`, any
`pair_on`, any `arm: "control"`. `requires_control_report` used to be the only
switch, so removing it left the delta rules running against an arm nobody had
identity-checked — a control with a different base model and a different source
revision returned exit 0 and "all predeclared rules satisfied".

The flag survives as metadata and is cross-checked. A spec that sets it false
while its own rules need a control is refused as self-contradictory, at check
time and before the sandbox.

When a control is required and none is supplied the run is unusable, every rule
is `unevaluable`, and the verdict is exit 2. Reporting "H2 falsified" for a
comparison that never happened was the previous behaviour.

### Pairing two arms

Expert ids are fresh UUIDs, so `pair_on: "experts[*].id"` can never match
across two runs. Each expert carries `pairing_key`, a digest of the cluster
label and the exact set of failing task ids that the live capture attempted —
correction-source-independent by construction. Matching keys mean the arms
attempted the same training-failure cluster. They do **not** require the
accepted, trainable subsets to match: verifier-approved correction yield is
part of the correction-source variable. They do **not** mean the training text
was identical; the correction texts differ by design, which is the variable
under test.

`ARM_IDENTITY_PATHS` covers actual run state, not only declared intent:

| path | why it is not optional |
| --- | --- |
| `provenance.source.status_sha256` | which files were dirty |
| `provenance.source.worktree_sha256` | what was *in* those files, tracked diff plus untracked bytes |
| `provenance.worker.framework_versions_sha256` | the worker's MLX and Python versions |
| `provenance.worker.checkout.{revision,tree,dirty}` | which worker code ran |
| `provenance.sandbox_image.fingerprint` | which verifier sandbox graded both arms |
| `run_setup.actual_training_failure_ids` | the failures the live capture attempted; this legacy name is retained for the same selection |
| `run_setup.actual_training_failure_set_sha256` | compatibility digest for the attempted capture set |
| `run_setup.attempted_training_failure_set_sha256` | a digest of the attempted capture set |

The existing `cohort_manifest_sha256` identifies the catalog, which is fixed
before either arm runs. It cannot tell two arms apart when the live capture
picked different failures.
The trained subset is deliberately not an arm-identity path: its digest is
bound by the run manifest for report integrity, but the subset may differ by
correction source and is an outcome of the comparison.

### Partial provenance is pairable; unresolved identity is not

A spec may list `permitted_provenance_gaps` (or permit them through a
`subset_of` rule on `provenance_gaps`). For such a path, pairing succeeds only
when **both** arms are unresolved **and** both arms report the gap in their own
`provenance_gaps`. One arm resolved and the other not is a mismatch. An
unresolved path nobody declared is a mismatch.

When a pair relies on a permitted gap the verdict says so:
`arm_pairing.provenance_completeness: "partial"`.

`required_resolved_identity` overrides the permission. A path listed there can
never be waived, because it is the evidence the comparison rests on. EXP-003
lists `provenance.worker.model_manifest_sha256`:

```json
"permitted_provenance_gaps": ["models.base.aggregate_sha256"],
"required_resolved_identity": ["provenance.worker.model_manifest_sha256"]
```

A worker can report the same checkout revision and the same MLX version while
holding different model files, so without that digest an A/B comparison cannot
show the two arms loaded the same model. Leaving it unresolved is not partial
provenance, it is no comparison:

```text
worker.model_manifest_sha256 is required for paired identity but is unavailable
```

When a required path is unresolved, **no** gap is waived. A base-model gap is
tolerable only while some other model identity still resolves. EXP-002 is a
single-arm report, so it may still declare the worker manifest as a partial
gap; EXP-003 may not.

### Timing is a separate claim, and Grove cannot make it

`declared_before_run` used to be a boolean with no consumer anywhere in the
repository. Replacing it with a self-reported `{type, timestamp}` map was no
better: `{"type": "rfc3161", "timestamp": "0000-not-a-time"}` verified. Nothing
parsed the timestamp, read a token, checked a signature or looked up a
registration, and two timestamps were compared as strings.

There are now exactly two outcomes:

| spec declares | timing verdict | blocking |
| --- | --- | --- |
| `seal_self_consistent: true`, `timing_attestation: null` | `unverified` | no |
| any `timing_attestation`, or `declared_before_run` / `preregistered` | refused | yes |

Both shipped specs take the first row. The second is refused at check time
(exit 2) and before the sandbox at run time, because no verifier exists here.
Verifying an RFC 3161 token means holding its DER bytes, checking the TSA
signature against a trusted certificate, matching `messageImprint` to the spec
digest, and reading `genTime` out of the token rather than from a field typed
beside it. A signed tag needs an immutable tag object and a trusted key; a
Rekor entry needs an inclusion proof and a signed checkpoint; an OSF
registration needs a retained, hashed registration snapshot.

**Grove has none of these, so Grove may not claim preregistered timing.** It
may claim that a report was bound to a named self-consistent sealed spec.

### Report integrity

Every rule reads a value out of an editable JSON file, so every value a rule
reads is bound.

`decision_rule_input_paths(spec)` collects every `path`, `control_path` and
`pair_on` a spec's rules can dereference, and the run manifest records a digest
of the exact resolved value behind each one, array order included:

```json
"decision_inputs": {
  "cycle.experts_admitted": {"present": true, "sha256": "..."},
  "experts[*].pairing_key": {"present": true, "sha256": "..."}
}
```

That set is derived from the spec rather than listed by hand, which is what
went wrong before: `correction_comparison`, `cycle` and the top-level pairing
key were rule inputs nobody had added to the manifest, so editing a self-repair
yield from a failing value to a passing one returned exit 0 and
`report_integrity: intact`. The checker now derives the same set, demands every
path be bound, and recomputes each digest. A missing or mismatched binding is
exit 2.

The manifest additionally commits to the spec digest, the run setup, the
provenance digest, the actual training failure set, the training proposals and
each expert's metrics — run state no rule happens to read, but that a report
should still commit to. The checker recomputes all of them plus
`provenance.provenance_sha256`.

A tampered report grades nothing: every rule is marked `unevaluable`,
`falsified_hypotheses` is empty, and the verdict is exit 2. An edit is not a
result.

Every digest in the repository — spec seals, provenance, run manifests,
decision inputs — goes through one encoder, **Grove canonical JSON v1**
(`grove.provenance.canonical_json`): UTF-8, keys sorted by code point, array
order preserved, no insignificant whitespace, NaN and the infinities rejected,
and any value that is not a JSON primitive, list or string-keyed object
rejected outright. RFC 8785 (JCS) is the reference; the one thing not pinned
here is JCS number formatting, which would need a dependency this repository
does not carry. The old encoder passed `default=str`, so a value JSON cannot
represent was replaced by its `repr` and hashed as if it were data.

The SQLite store adds two local bindings: every evaluation row stores a digest
of its own metrics, and the ledger is a hash chain where each entry commits to
its payload and to the previous entry's digest. `grove.store.verify_evaluations`
and `grove.store.verify_ledger` recompute both, and every report embeds their
verdicts.

This detects edits. It is not signed and not externally anchored, so it does
not defeat someone who can rewrite every local artifact **and** recompute every
digest. An in-toto/DSSE signature over the manifest, or a Sigstore Rekor entry
for its digest, is what would.

## EXP-002 — forced replay and route precision

Spec: `experiments/EXP-002-forced-replay-and-route-precision.json`.

Predeclared question: does a grown expert preserve prior competence *without*
the router shield?

Prerequisites: reachable MLX worker (`uv run grove worker-preflight`), the
`grove-python-base` LXD image, a fresh database, and a replay cohort of at least
50 prior-passing tasks authored without reference to the expert's failure
family. `REAL_CYCLE_POLICY` enforces that minimum, so a thinner cohort rejects
the candidate rather than producing a stability claim.

**This does not run today.** The catalog supplies at most 24 captured tasks
against the 50 this spec declares, so the command below exits 2 at the capacity
preflight, prints one `setup_refused` document to stderr, and creates nothing.
Author the cohort first.

```bash
uv run grove --db /srv/storage/grove/grove-exp002.db real-cycle --reset \
  --spec experiments/EXP-002-forced-replay-and-route-precision.json \
  --arm primary \
  --report /srv/storage/grove/evaluations/exp002.json
uv run python scripts/check_experiment_spec.py \
  --spec experiments/EXP-002-forced-replay-and-route-precision.json \
  --report /srv/storage/grove/evaluations/exp002.json
```

Route recall is measured on an **oracle-free** copy of each held-out task.
A held-out task still carries the gold family tag the clusterer used to build
the expert's routing profile, and `ProfileRouter.score` scores an exact tag
overlap at 1.0, so the old probe compared two copies of the same label and
called it recall. `experts[*].metrics.route_positive_source` is now
`heldout_forced_pass_oracle_free` and `route_probe_metadata` is `oracle_free`;
D7a and D7b pin both. The gold-tag number is still reported as
`route_recall_gold_tags`, alongside
`route_recall_gold_tags_independent: false`. It is a diagnostic. It is never
independent routing evidence and it gates nothing; the runner and checker reject
any decision rule that names either gold-tag diagnostic path.

There is no `min_route_precision`. Route precision's denominator mixes the
held-out positive cohort with the replay negative cohort, so resizing the
replay buffer moves it even when the router has not changed. The field used to
be accepted as a dataclass `InitVar` and silently dropped; `SleepPolicy` now
raises `TypeError`, and a spec declaring it exits 2.

**H3 is expected to fail** on an expert like the 2026-07-31 one. That is the
point of writing it down.

The 2026-07-31 artifact cannot be retro-judged under this spec, and the checker
says so rather than scoring it:

```
$ uv run python scripts/check_experiment_spec.py \
    --spec experiments/EXP-002-forced-replay-and-route-precision.json \
    --report /srv/storage/grove/evaluations/final-real-cycle.json
verdict: report is not bound to this spec: report records no
         experiment_spec.spec_sha256                              (exit 2)
```

That is the correct outcome: the run predates the protocol, so it was never
launched under the spec and cannot be graded by it. What the older report does
contain is still readable directly — a held-out rate of 0.75, a routed
regression of 0.0, and a replay cohort of **two** tasks against the 50 this spec
requires. The first two numbers are the narrow claim the audit accepted; the
third is why the stability claim was never supportable.

## EXP-003 — self-generated versus human corrections

Spec: `experiments/EXP-003-correction-source-ab.json`. Two arms, two fresh
databases, everything else held constant:

```bash
uv run grove --db /srv/storage/grove/grove-exp003-canonical.db real-cycle --reset \
  --spec experiments/EXP-003-correction-source-ab.json --arm control \
  --correction-source canonical --compare-corrections \
  --report /srv/storage/grove/evaluations/exp003-canonical.json
uv run grove --db /srv/storage/grove/grove-exp003-self.db real-cycle --reset \
  --spec experiments/EXP-003-correction-source-ab.json --arm primary \
  --correction-source self-repair --compare-corrections \
  --report /srv/storage/grove/evaluations/exp003-self.json
uv run python scripts/check_experiment_spec.py \
  --spec experiments/EXP-003-correction-source-ab.json \
  --report /srv/storage/grove/evaluations/exp003-self.json \
  --control-report /srv/storage/grove/evaluations/exp003-canonical.json
```

The control report is not optional. The sealed delta, `control_path`, and
`pair_on` rules derive that requirement; `requires_control_report` is redundant
metadata and cannot disable it. Omitting `--control-report` exits 2 instead of
grading the self-repair arm against an absolute threshold it could clear while
the human-reference arm. The checker also refuses arms that differ in anything
but the correction source, and rejects two arms that used the same source.

Each arm names its own setup profile. `required_setup` demands
`correction_source: self-repair`; `control_required_setup` demands
`canonical`. `--arm control` selects the second, which is what makes a
canonical control launchable at all — validating it against the primary
declaration produced `correction_source declared 'self-repair' but the run
uses 'canonical'` before anything ran. Omit `--arm` and the profile is inferred
from the declared correction source; an unmatched run falls back to the primary
profile so the contradiction is reported rather than hidden.

`--compare-corrections` answers only the cheap half, and only about
*corrections*: per source, how many failures produced a verifier-approved
correction, the mean number of attempts, and which task ids each source covered
alone. It does not establish that a self-repair-trained expert matches a
canonical-trained one. That is `H2`, it is evaluated by the `delta>=` rule
against the control arm, and it needs both MLX runs.

The arms are not matched on inference budget: self-repair gets up to three
generation attempts per failure, the canonical arm gets one lookup. Olausson et
al. (ICLR 2024) report GPT-4 self-repair feedback at 33.30% against human
feedback at 52.60%, so a shortfall would be unsurprising.

Self-repair is verifier-gated by construction: `SelfRepairSource` rewrites the
prompt into a repair instruction, keeps the task identity so the *same* hidden
suite grades the retry, and discards any attempt the verifier rejects. A failure
that never clears the verifier produces no training target at all.

The comparison's proposals are the proposals that get trained on. Previously
`compare_correction_sources` generated them, discarded them, and the training
loop asked the same source again; self-repair is nondeterministic, so the
reported verified yield described corrections nobody trained on. The runner now
reuses the selected source's exact proposal, records
`correction_comparison.training_proposal_reuse` and a per-proposal
`response_sha256`, and counts the comparison and training generation budgets
separately. D10 pins the reuse flag.

The report separates the failures the live capture attempted from the failures
that became trainable: `run_setup.attempted_training_failure_ids` (also retained
as `actual_training_failure_ids`) names the capture set, while
`run_setup.trained_failure_ids` names only failures with accepted corrections.
Their digests are bound into the run manifest. A difference in the trainable
subset is an outcome of correction generation, not a reason to pretend the
arms selected different clusters.

## EXP-004 — fair self-repair regime (executed 2026-08-09: unusable, exit 2)

Spec: `experiments/EXP-004-fair-self-repair-ab.json`, sealed
`f129e49a10cfb00f659420ca04fd6ae7fa4e54d61a1613690f47d92162795dd5`. Executed
2026-08-09: **unusable (exit 2)** — self-repair 0/20 verified in 160 sampled,
seeded calls, no primary expert, arms unpairable
(`research/2026-08-09-exp004-fair-self-repair.md`). The regime description
below is what was sealed and what ran.

The 2026-08-08 EXP-003 run was **unusable** (exit 2): the self-repair arm
produced 0 verified corrections in 60 generation calls, trained no expert, and
the checker refused to pair the arms
(`research/2026-08-08-exp003-ab.md`). That run was also not a fair test of
self-repair. Repair attempts were generated under the global greedy decoding
(temperature 0.0), so all three attempts per failure were near-identical, and
every repair prompt carried one generic failure sentence instead of the
verifier's reason.

EXP-004 retests the same two hypotheses under a fair, spec-driven regime:

- **Per-purpose decoding.** Repair-attempt generation samples at temperature
  0.8 with one recorded integer seed per attempt, derived from the declared
  `base_seed` 20260809. Grading, baseline, held-out and replay decoding stay
  greedy at 0.0; `run_setup.decoding_by_purpose` records all of it and rules
  F2–F4b bind it. Worker seeding was verified before sealing: `mx.random.seed`
  deterministically drives `make_sampler`'s categorical draws, and the worker
  echoes the temperature and seed each request actually ran.
- **Budget by configuration.** `self_repair_attempts: 8` comes from the spec's
  `required_setup` machine block — the runner resolves it from the selected
  arm profile, not from an edited policy default — and rule F1 pins it.
- **Honest feedback.** Repair prompts carry the verifier's failure detail
  through an allowlist: reason, case pass counts, exit/timeout/output class,
  and the stderr exception *class name*. Never hidden expected values, case
  payloads, or stderr message text; a shipped test asserts no hidden-suite
  expected value can appear in a repair prompt.
- **Declared threshold, with a written justification.** D3 requires a verified
  self-repair rate ≥ 0.25 — deliberately not EXP-003's 0.5, which was never
  calibrated for a 1.5B 4-bit model. 0.25 is the least rate that clears the
  `min_cluster_size` 3 demand gate on the 20-task cluster (so the least rate
  at which H2 is testable at all), and corresponds to a ~3.5% per-attempt
  repair probability under 8 sampled attempts. The full justification is
  sealed in the spec's background.
- **Same comparison design as EXP-003.** Two arms, canonical control, pairing
  on `experts[*].pairing_key` (attempted-failure identity), the same permitted
  provenance gap and required resolved worker-model identity, the same run
  manifest bindings, and the same unusable-versus-falsified semantics.

```bash
uv run grove --db /srv/storage/grove/grove-exp004-canonical.db real-cycle --reset \
  --spec experiments/EXP-004-fair-self-repair-ab.json --arm control \
  --correction-source canonical --compare-corrections \
  --report /srv/storage/grove/evaluations/exp004-canonical.json
uv run grove --db /srv/storage/grove/grove-exp004-self.db real-cycle --reset \
  --spec experiments/EXP-004-fair-self-repair-ab.json --arm primary \
  --correction-source self-repair --compare-corrections \
  --report /srv/storage/grove/evaluations/exp004-self.json
uv run python scripts/check_experiment_spec.py \
  --spec experiments/EXP-004-fair-self-repair-ab.json \
  --report /srv/storage/grove/evaluations/exp004-self.json \
  --control-report /srv/storage/grove/evaluations/exp004-canonical.json
```

What EXP-004 does not claim: the seal proves which declaration a future report
ran under, not when this spec was written (no timing attestation exists here),
and a pass would show verifier-approved self-repair under a generous sampled
budget, not parity with human corrections.

## EXP-005 — second-cycle coexistence (executed 2026-08-11: H1+H2 falsified, exit 1)

Spec: `experiments/EXP-005-second-cycle-coexistence.json`, sealed
`e6867048551b2df4885b41c8894f2d7aa02f4d5744002a9503765306f5ecd733`.
**Executed 2026-08-11 (exit 1): H1 and H2 falsified — 23/29 rules passed, 6
failed (D5, D6, D7, D8b, D19b, D20). The `path_restructure` candidate met
every competence gate (target 0.85, held-out 0.75, plasticity 0.85, recall
1.0) but with expert 1 deployed the router claimed 18/112 base-passing replay
tasks (false-positive rate 0.161 vs budget 0.0, route precision 0.143) and 12
regressed under routing (0.107 vs 0.0); the unchanged admission policy
rejected it and the pool ended with one expert. Full record:
`../research/2026-08-11-exp005-second-cycle.md`.**

Predeclared question: can Grove grow a **second** expert on a new failure
family while the first expert stays deployed — without router confusion,
without interference with the first expert's competence, and with
monotonically non-decreasing capability? Every routing and stability number so
far is a single-expert measurement; this is the first plural (README open
questions 2 and 4).

- **A real two-cycle mode.** `real-cycle --cycles 2` runs the complete first
  cycle, then — against the same store, with the first expert still admitted,
  deployed and routable, no reset in between — captures the second family,
  trains and gates a second candidate through the identical admission policy.
  One invocation, one fresh database. The `growth_cycles` count resolves from
  the sealed `required_setup` machine block, like EXP-004's repair budget, and
  a one-cycle launch under this spec refuses before any cost.
- **The second family.** `path_restructure`: move, copy, pick and drop over
  escaped-dot paths — 20 freshly authored training prompts and 4 fresh
  held-outs never read by any prior gate, all content-hash disjoint from every
  cycle-1 prompt. It anchors the declared transfer targets `path_rename` /
  `path_project` without touching them: those two stay archived in the FUTURE
  role, and the after-cycle-2 future probe reads transfer to them as a
  diagnostic only (they carry cycle-1's family tag, so the routed probe
  under-measures expert-2 transfer; no rule gates on it). The shared
  escaped-dot vocabulary is the router stressor, on purpose.
- **Multi-expert measurement per checkpoint** (baseline, after cycle 1, after
  cycle 2), recorded under `coexistence`: route recall/precision/false-positive
  rate across the whole deployed pool on oracle-free probes; routed replay over
  the prior-passing suite (≥ 50 tasks, cycle-1 training targets included after
  cycle 1 by passing routed during the second capture); expert 1's held-out
  rate re-measured routed and forced-on with expert 2 deployed (the
  interference probe); per-expert forced-on replay against the bare base so
  `forgetting_claim` stays honest per expert; capability over a cohort union
  fixed from baseline on; parameter growth.
- **Declared thresholds, justified in the sealed background.** Held-out bars
  0.75 (`min_heldout_fix_rate`), two-expert recall ≥ 0.5 (`min_route_recall`),
  precision ≥ 0.8 (at most one cross-family misroute among the expected
  expert-bound routes; sealed as a decision rule, deliberately not a policy
  field), false positives ≤ 0.0, replay ≥ 50 with ≥ 16 cycle-1 targets
  (`min_target_fix_rate` × 20), routed regression ≤ 0.0, capability deltas
  ≥ 0.0 per checkpoint step, added parameters in (0, 6 M].
- **The EXP-004 structural lesson, applied.** Single-arm, no delta, no
  `pair_on`, no control rules: every rule grades independently, so a
  measurable zero — no second expert, an absent per-family metric — records as
  a **falsification (exit 1)**, never as unusable. Exit 2 stays reserved for
  genuine grading blockers: a tampered or unbound report, an edited rule
  input, a setup that contradicts the sealed declaration, an infrastructure
  refusal. The semantics are stated in `preregistered_limitations`.
- **EXP-004's integrity rigor.** `requires_report_integrity`, gaps limited to
  `models.base.aggregate_sha256`, and
  `provenance.worker.model_manifest_sha256` under
  `required_resolved_identity`, so a worker that cannot name its model files
  refuses the run before any cost.

The complete procedure is one script, whose invocation the sealed `command`
field documents and whose real-cycle exit code — exit 2 included — propagates
unchanged:

```bash
scripts/run_exp005.sh /srv/storage/grove/grove-exp005.db \
  /srv/storage/grove/evaluations/exp005.json
```

which runs exactly:

```bash
uv run grove --db <db> real-cycle --reset --cycles 2 \
  --spec experiments/EXP-005-second-cycle-coexistence.json \
  --arm primary --correction-source canonical --report <report>
uv run python scripts/check_experiment_spec.py \
  --spec experiments/EXP-005-second-cycle-coexistence.json --report <report>
```

What EXP-005 did not claim, and its execution did not change: preregistration
timing (seal is self-consistent, not timestamped); N-expert scaling or learned
routing (two experts, auditable profile router); anything about self-repair,
whose 0/20-twice floor is a recorded result this experiment did not
relitigate. The falsified component is the tag/keyword router's same-domain
discrimination, not the candidate or the gates.

## Guardrails that run today, without MLX

These are deterministic and need no Apple worker:

```bash
uv run --extra dev pytest -q                    # full suite incl. LXD integration
uv run grove provenance                         # reproducibility record + gaps
uv run grove provenance --worker                # adds a live worker round trip
uv run python scripts/check_experiment_spec.py --spec experiments/EXP-002-forced-replay-and-route-precision.json --report <report.json>
uv run python scripts/audit_evaluation_report.py --db <db> --report <report.json>
uv run grove --db .grove/demo.db demo --reset   # deterministic control-plane demo
```

The demo is itself a live demonstration of the router-shield distinction: its
admitted expert records `regression_rate: 0.0` over four prior-passing replay
tasks and `forced_regression_rate: 1.0` over the same four, so
`forgetting_claim` is `router_shielded`. Tightening one policy field
(`SleepPolicy(max_forced_regression_rate=0.0)`) rejects that same expert. Both
behaviours are pinned in `tests/test_forgetting_gates.py`.

The zero-cost preflight refuses an inadmissible run before the sandbox exists.
An unknown correction source, a non-integer or non-positive
`self_repair_attempts`, a hollow sealed spec, an invalid setup schema, an
unknown `--arm`, an unsupported setup key, any preregistration timing claim, a
paired spec with no machine-checkable `control_required_setup`, and a
`requires_control_report: false` flag that contradicts the spec's own rules all
raise `ExperimentSetupError` with no database, no report and no sandbox.
`tests/test_experiment_spec.py` pins each one, including that no artifacts were
created.

`scripts/audit_evaluation_report.py --annotate` resolves the annotation path
against both inputs before opening anything, through symlinks and hard links,
and exits 2 rather than writing over `--db` or `--report`.

It also recomputes every evaluation-row digest, and reports three distinct
states rather than two:

| state | meaning | exit |
| --- | --- | --- |
| `clean` | every row verified against its stored digest | 0, or 1 if the report is stale |
| `unverified` | at least one row carries no digest, so nothing was checked | 2 |
| `tampered` | at least one row disagrees with its stored digest | 2 |

`intact = not tampered` used to collapse the first two, so a database whose rows
all predate digest recording reported `status: clean` with `checked: 0`. That is
exactly the 2026-07-31 store. An annotation produced over unverified rows now
carries `evidence_status: unverified` and a note saying it is historical
reconciliation only. `benchmark.curve()` labels each checkpoint the same way, so
an unverifiable legacy row cannot quietly support a capability claim.

## What this protocol does not claim

- No result here shows adapter-intrinsic non-forgetting. The one measurement
  that exists points the other way.
- No result here shows the model can learn from its own corrections at scale.
  The code path exists and is verifier-gated; the yield is unmeasured on the
  real model.
- Binding a report to a sealed spec prevents a threshold from being edited after
  the fact. It does not make a post-hoc *cohort design* blind, it is not a
  timestamp, and it is not a substitute for a held-out benchmark authored by
  someone else.
- `provenance_gaps: []` on the control host would be a red flag, not a
  reassurance: the base model lives on the worker and cannot be hashed from
  here. Expect `models.base.aggregate_sha256` to remain open. Whether
  `worker.model_manifest_sha256` may also remain open depends on the spec:
  **EXP-002** is single-arm and permits it; **EXP-003** is paired and does not,
  because it is the only evidence both arms loaded the same weights. EXP-003
  refuses before the database and the sandbox when the worker cannot supply it.
- Missing evidence is not clean evidence. An absent provenance digest, an
  unhashed evaluation row and a hole in the ledger chain are all `unverified`,
  and unverified records may be read as history but never quoted as proof. The
  2026-07-31 store's rows predate digest recording, so nothing derived from them
  is tamper-evident.
- Matching arm identity shows two arms were *configured* alike and reported the
  same worker identity. The worker's checkout is a self-report; nothing here
  proves an untrusted worker described honestly what it ran. That needs a
  worker-side signed attestation.
- Report and ledger digests detect edits. They are unsigned and locally
  anchored, so they do not defeat an attacker who can rewrite every artifact.
- Oracle-free routing removes one specific circularity. It does not fix
  post-hoc cohort design, prompt similarity, or generalisation beyond the
  held-out cohort.
- Grove has no timing attestation, so it may not claim preregistration timing —
  only that a report was bound to a named sealed spec version.
- **Executed so far: EXP-002 (2026-08-08 rerun; H3 falsified — forced
  regression 46/94, router-shielded), EXP-003 (2026-08-08; unusable, exit
  2 — self-repair 0/20 verified, no primary expert, arms unpairable; see
  `research/2026-08-08-exp003-ab.md`), EXP-004 (2026-08-09; unusable, exit
  2 — self-repair 0/20 verified again in 160 sampled seeded calls, control
  20/20 with an admitted expert, arms unpairable; see
  `research/2026-08-09-exp004-fair-self-repair.md`) and EXP-005 (2026-08-11;
  exit 1 — H1+H2 falsified: the second candidate met every competence gate but
  route precision measured 0.143 and routed regression 0.107 against zero
  budgets, so admission rejected it; see
  `research/2026-08-11-exp005-second-cycle.md`).** The broader
  continual-learning thesis remains unproven: no run has yet demonstrated an
  admissible expert trained on self-generated corrections, and no run has yet
  ended with two coexisting deployed experts — EXP-005 measured the attempt
  and recorded its falsification.
- The deterministic demo exercises the control plane, not the thesis. A toy
  arithmetic backend that reports `router_shielded` proves the metric is honest;
  it says nothing about a 1.5B coding model.
