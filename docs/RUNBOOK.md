# Grove experiment runbook

This runbook explains how to inspect, reproduce, and safely extend the first real MLX experiment. Read `EXPERIMENT_REPORT_2026-07-31.md` before interpreting new numbers; the report documents the failed candidates and known validity limits.

## Safety rules

1. Never execute model-written code directly on Agentbox or grove-worker-1's Mac Mini. Use `SandboxedPythonVerifier` and `LxdSandbox`.
2. Never train from an unverified response. A training role requires a matching accepted correction.
3. Never reuse a held-out or regression prompt as training input.
4. Never overwrite an existing evidence database, dataset directory, adapter directory, or report during research runs.
5. Never promote a rejected or removed expert by editing its status. Create a new candidate.
6. Never merge an adapter into the frozen base for this experiment.
7. Treat prompts, responses, corrections, databases, and logs as potentially sensitive data.

## Repository and storage layout

```text
~/grove/                    source repository
/srv/storage/grove/
  datasets/<candidate-id>/              immutable training JSONL + manifest
  experts/<candidate-id>/               downloaded adapter, config, log
  evaluations/                          JSON experiment reports
  models/                               reserved model storage
  manifests/                            reserved exported manifests
  sandbox-images/                       reserved sandbox exports
  logs/                                  operational logs
  backups/                               local backup staging
  tmp/                                   temporary work
  grove-real*.db                         SQLite evidence databases

/Users/grove-worker/grove/              Mac worker root
  runtime/repo/                          synchronized Grove source + virtualenv
  cache/huggingface/                     pinned model cache
  datasets/<candidate-id>/               transferred training dataset
  adapters/<candidate-id>/               trained MLX adapter
  jobs/<job-id>/                          input specification + result JSON
```

## Current pinned configuration

| Setting | Value |
|---|---|
| Model repository | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` |
| Model commit | `b3252a2f97102b1fb1571fec2c9b27219a8536be` |
| Model source on worker | Snapshot path defined by `BASE_MODEL_SOURCE` in `src/grove/experiment.py` |
| Decoding | Temperature 0.0, maximum 768 tokens, stop at `<|im_end|>` |
| Training | 60 iterations, batch size 1, 8 layers, sequence length 1024, learning rate `2e-5`, seed 17 |
| Sandbox | `grove-python-base` with profile `grove-sandbox` |
| Router | `profile-router-v1` |
| Verifier suite | `escaped-path-v2+python-core-v1` |

## Preflight

Run from the repository root:

```bash
cd ~/grove
scripts/preflight.sh
```

The script checks Git, uv, Tailscale, SSH, LXD, KVM, the data volume, the dedicated SSH key, free storage, and the remote MLX worker. A healthy remote result includes:

```json
{
  "status": "ok",
  "machine": "arm64",
  "default_device": "Device(gpu, 0)",
  "mlx": "0.32.0",
  "mlx_lm": "0.31.3"
}
```

The worker can also be checked directly:

```bash
uv run grove worker-preflight
```

Inspect the sandbox without executing model code:

```bash
lxc profile show grove-sandbox
lxc image info grove-python-base
lxc list --format csv -c n,s
```

No `grove-run-*` instance should remain after a completed verification.

## Install and test

Agentbox has no runtime Python dependencies beyond the standard library. Install the locked development environment and run all tests with:

```bash
uv sync --extra dev
uv run pytest
```

The final implementation collected 25 tests. Five integration tests require the local LXD service. Run only fast tests with:

```bash
uv run pytest -m 'not integration'
```

Run only sandbox integration tests with:

```bash
uv run pytest tests/test_sandbox_integration.py
```

Check source syntax and patch cleanliness:

```bash
uv run python -m compileall -q src tests
git diff --check
```

## Synchronize worker code

The worker checkout at `/Users/grove-worker/grove/runtime/repo` must be a real
git checkout pinned to a commit, not an rsync'd directory. A bare rsync leaves
the worker unable to self-report `checkout.revision`, `tree`, `dirty`,
`status_sha256`, and `worktree_sha256`, which failed decision rule D9 in the
2026-08-08 EXP-002 run as five undeclared provenance gaps. Sync with a git
bundle instead:

```bash
# On the control host: package full history for the current HEAD.
git bundle create /tmp/grove.bundle HEAD
scp -i ~/.ssh/grove_worker -o IdentitiesOnly=yes \
  /tmp/grove.bundle \
  grove-worker@grove-worker-1:/Users/grove-worker/grove/runtime/grove.bundle

# On the worker: fetch from the bundle and pin the checkout to the same HEAD.
ssh -i ~/.ssh/grove_worker -o IdentitiesOnly=yes \
  grove-worker@grove-worker-1 '
  cd /Users/grove-worker/grove/runtime/repo &&
  /usr/bin/git init -q 2>/dev/null;
  /usr/bin/git fetch -q ../grove.bundle HEAD &&
  /usr/bin/git checkout -qf FETCH_HEAD &&
  /usr/bin/git clean -fd &&
  /usr/bin/git status --porcelain'
```

The final `git status --porcelain` must print nothing: an empty status is what
lets `worker-preflight` report `dirty: false` and a `worktree_sha256`. `git
clean -fd` respects `.gitignore`, so the worker virtual environment
(`.venv/`) survives; never pass `-x`. Confirm from the control host with
`uv run grove worker-preflight` that `checkout.revision` equals the local
`git rev-parse HEAD`.

## Provision a replacement Mac worker

The checked-in provisioning script deliberately contains no machine-specific IP address or SSH key. On the target Mac, provide those values as environment variables and run the script as root:

```bash
sudo env \
  GROVE_AGENTBOX_TAILSCALE_IP='AGENTBOX_TAILSCALE_IP' \
  GROVE_WORKER_PUBLIC_KEY='ssh-ed25519 PUBLIC_KEY COMMENT' \
  GROVE_WORKER_UID='502' \
  bash scripts/provision_macos_worker.sh
```

`GROVE_WORKER_UID` is optional and defaults to 502. Confirm that the chosen UID is unused. Never place the private key in an environment variable, script, repository, or Mac authorized-keys file; only the public key belongs on the worker.

## Run a clean first-cycle experiment

> **Blocked, by design.** `REAL_CYCLE_POLICY` declares a 50-task prior-passing
> replay cohort while `coding_catalog()` supplies at most 24 captured tasks
> (20 train + 4 regression). A capacity preflight rejects the run before the
> sandbox or the worker is touched, so the command below currently exits 2
> with a one-document `setup_refused` message on stderr and creates nothing. Author the 50 predeclared,
> base-passing, family-independent replay tasks first, or reseal a spec and
> policy that declare a smaller pilot and label it as one.

Choose new, explicit database and report names. Verify they do not already exist:

```bash
test ! -e /srv/storage/grove/grove-real-next.db
test ! -e /srv/storage/grove/evaluations/real-cycle-next.json
```

Then run, naming the sealed spec so the report is bound to its declaration, and
the arm whose setup profile the run must satisfy:

```bash
uv run grove \
  --db /srv/storage/grove/grove-real-next.db \
  real-cycle --reset \
  --spec experiments/EXP-002-forced-replay-and-route-precision.json \
  --arm primary \
  --report /srv/storage/grove/evaluations/real-cycle-next.json
```

`--reset` deletes only the exact database path supplied to the command, and only
after the preflights below have all passed. Never point it at an existing
evidence database that must be preserved.

Without `--spec` the report carries no `experiment_spec` block and
`scripts/check_experiment_spec.py` will refuse to grade it (exit 2).

`--arm` matters for a paired spec. EXP-003 declares `required_setup` with
`correction_source: self-repair` and `control_required_setup` with `canonical`.
Run the control arm with `--arm control`; omitting it infers the profile from
the declared correction source. Getting this wrong is refused before anything
runs, not discovered afterwards.

### Preflight order — everything free happens first

Each step below is reached only if the previous one passed. Nothing is written
and no container is created until step 7.

1. **Sealed-spec load.** An unsealed or edited spec raises `ValueError`.
2. **Zero-cost preflight** (`preflight_experiment`), all pure:
   unknown correction source; non-integer or non-positive
   `self_repair_attempts`; hollow spec (no hypotheses, no decision rules, or a
   hypothesis with no `falsified_if`); invalid `required_setup` schema; unknown
   `--arm`; an unsupported setup key such as `min_route_precision`; any
   preregistration timing claim, with or without a `timing_attestation`,
   because no verifier for one exists.
3. **Arm setup conformance.** Every declared machine key must be present in the
   run setup manifest and equal. A missing key is a refusal, not a footnote.
4. **Replay capacity.** The declared minimum replay cohort must be reachable
   from the catalog at all.
5. **Required run identity** (`preflight_required_identity`). A spec that lists
   `required_resolved_identity` gets one worker round trip and a local git read
   here. EXP-003 lists `provenance.worker.model_manifest_sha256`, so a worker
   that returns no model manifest stops the run now rather than after two MLX
   trainings that could never have been compared. Paths this host cannot
   resolve before the run are reported as `deferred` and checked when the
   report is graded.
6. **Sandbox preflight.** The image and networkless profile are checked with
   LXD control calls only. A failure is a setup refusal and happens before any
   database reset.
7. Only now: database reset, model, training.

Every refusal raises `ExperimentSetupError`, which the CLI prints as one
`setup_refused` JSON document on stderr and exits **2** — no traceback, and the
same code the checker uses for "cannot be judged". Exit 1 is reserved for a
falsified prediction, which a refused setup is not. Verify with:

```bash
uv run grove --db /tmp/probe.db real-cycle --reset \
  --spec experiments/EXP-003-correction-source-ab.json --arm control \
  --correction-source canonical --compare-corrections \
  --report /tmp/probe.json
echo "exit=$?"
test ! -e /tmp/probe.db && test ! -e /tmp/probe.json && echo "refused, nothing created"
```

The command performs, in order:

0. the six setup preflight steps above — all before any state is written or any
   container is launched;

1. database reset;
2. frozen baseline manifest publication;
3. immutable split assignment;
4. baseline benchmark;
5. live base inference and failure capture;
6. canonical-correction verification in LXD;
7. candidate dataset materialization and hashing;
8. MLX LoRA training on the Mac;
9. adapter transfer and aggregate hash verification;
10. forced birth-task probation;
11. routed replay regression checks;
12. held-out probation;
13. admission or rejection;
14. post-growth benchmark;
15. future-stream probe;
16. rollback benchmark and restoration if admitted;
17. JSON report generation.

Those timings are historical, from the 2026-07-31 run: about 22.5 minutes end to
end, of which MLX training was 34.8 seconds. Fresh-container verification
dominated wall time; no step took 22.5 minutes on its own. They describe the
configuration of that run, not what the command above will do today — under the
current policy it stops at the capacity preflight in under a second. Quiet
terminal output during a real cycle is normal because the CLI prints the report
only at completion.

## Understand normal outcomes

Rejection is a successful pipeline outcome when a candidate misses any gate. Do not lower a gate merely to obtain an admitted adapter.

Inspect the resulting status:

```bash
uv run grove --db /srv/storage/grove/grove-real-next.db status
uv run grove --db /srv/storage/grove/grove-real-next.db ledger
uv run grove --db /srv/storage/grove/grove-real-next.db curve
```

Important distinctions:

- `active` is lifecycle state;
- current deployment membership determines whether an active expert is routable;
- rejected and removed experts remain in the ledger but cannot route;
- benchmark and probation attempts do not create new training failures;
- unresolved failures may remain after admission if a candidate fixes only the threshold-required fraction.

## Inspect the preserved final run

```bash
uv run grove --db /srv/storage/grove/grove-real-v3.db status
uv run grove --db /srv/storage/grove/grove-real-v3.db ledger
uv run grove --db /srv/storage/grove/grove-real-v3.db curve
```

Primary files:

```text
/srv/storage/grove/evaluations/final-real-cycle.json
/srv/storage/grove/grove-real-v3.db
/srv/storage/grove/datasets/expert_979511319695/
/srv/storage/grove/experts/expert_979511319695/
```

The current deployment should be sequence 7 with `expert_979511319695` and the exact base commit.

## Verify artifact integrity

Grove's recorded adapter hash is an aggregate over relative filenames and each file's SHA-256, not merely the `adapters.safetensors` hash.

```bash
uv run python - <<'PY'
from pathlib import Path
from grove.experiment import _artifact_hash
from grove.store import GroveStore

with GroveStore('/srv/storage/grove/grove-real-v3.db') as store:
    expert = store.get_expert('expert_979511319695')
    digest, size = _artifact_hash(Path(expert.artifact['local_adapter_path']))
    print('expected', expert.artifact['adapter_sha256'])
    print('actual  ', digest)
    print('bytes   ', size)
    assert digest == expert.artifact['adapter_sha256']
PY
```

Record ordinary file hashes before copying artifacts off-host:

```bash
sha256sum \
  /srv/storage/grove/evaluations/final-real-cycle.json \
  /srv/storage/grove/grove-real-v3.db \
  /srv/storage/grove/experts/expert_979511319695/adapters.safetensors
```

SQLite files change whenever new manifests or ledger entries are appended, so record a new database hash after every intentional state change.

### Never annotate over an input

`scripts/audit_evaluation_report.py --annotate` must be given a **new**
destination. Pointing it at `--db` used to succeed: it wrote JSON over the
SQLite file and left it unreadable. The command now resolves the annotation
path against both inputs first, through symlinks and hard links, and exits 2
before opening anything for writing.

```bash
uv run python scripts/audit_evaluation_report.py \
  --db /srv/storage/grove/grove-real-v3.db \
  --report /srv/storage/grove/evaluations/final-real-cycle.json \
  --annotate /srv/storage/grove/evaluations/final-real-cycle-annotated.json
```

Exit codes: `0` clean, `1` the report disagrees with the database, `2` an
annotation path collided with an input **or** the database evidence is not
authoritative. Evidence has three states, not two:

| `integrity.status` | meaning | exit |
| --- | --- | --- |
| `clean` | every row verified against its stored digest | 0, or 1 if stale |
| `unverified` | a row carries no digest, so nothing was checked | 2 |
| `tampered` | a row disagrees with its stored digest | 2 |

`intact = not tampered` used to collapse the first two, so a database whose rows
all predate digest recording reported `clean` with `checked: 0`. A row is not
authoritative simply because it lives in a database, and it is not authoritative
simply because nothing contradicted it. An annotation written over unverified
rows carries `evidence_status: unverified` and says in the file that it is
historical reconciliation only.

### Recompute the local integrity bindings

```bash
uv run python - <<'PY'
from grove.store import GroveStore

with GroveStore('/srv/storage/grove/grove-real-v3.db') as store:
    print('evaluations', store.verify_evaluations())
    print('ledger     ', store.verify_ledger())
PY
```

`verify_ledger` walks the hash chain: each entry commits to its payload and to
the previous entry's digest, so an edited or deleted row invalidates itself and
everything after it. An unhashed row breaks continuity rather than being stepped
over: it appears in `unhashed_sequences`, every row after it appears in
`unanchored_sequences`, and the verdict is `unverified`. The walk used to reset
the running digest to an empty string and carry on, so a stretch of legacy rows
verified as intact.

Both verifiers return `status` and `authoritative`. Only `clean` is
authoritative. `benchmark.curve()` labels each checkpoint `verified`,
`unverified` or `tampered` for the same reason: a row nobody hashed must not
quietly support a capability claim.

A report written by the runner carries `run_manifest` and
`run_manifest_sha256`. `scripts/check_experiment_spec.py` recomputes them,
`provenance.provenance_sha256`, every decision-rule input, and each expert's
metrics and adapter digest, and exits 2 when any of them was edited after the
run. A strict spec whose report carries *no* provenance digest is `unverified`
and equally unusable: absence is not agreement.

All of this is local and unsigned. It detects edits; it does not survive an
attacker who can rewrite every artifact. An external anchor — an RFC 3161
token, a signed in-toto/DSSE envelope, or a Sigstore Rekor entry for the
manifest digest — is the missing piece, and Grove does not have one.

### Worker source and model identity is a self-report

`grove worker-preflight` and `grove provenance --worker` record the worker's
own account of itself. Every one of these fields is compared across the two
arms of a paired experiment:

| field | what it pins |
| --- | --- |
| `worker.checkout.revision` / `.tree` | which commit the worker code is at |
| `worker.checkout.dirty` | whether anything is uncommitted |
| `worker.checkout.status_sha256` | which paths are dirty |
| `worker.checkout.worktree_sha256` | the **bytes** in them, tracked and untracked |
| `worker.framework_versions_sha256` | the Python, MLX and MLX-LM versions |
| `worker.model_manifest_sha256` | a digest over the base-model files the worker holds |

`worktree_sha256` used to hash `git diff HEAD`, so changing an untracked or
binary file left it unchanged. Both host and worker now call
`grove.provenance.worktree_digest`, which hashes file bytes, symlink targets,
executable bits and a tombstone per deleted path, from NUL-delimited
`git status` output. Ignored paths are excluded, and the digest payload says so.

Any field the worker does not return is recorded as an explicit `unavailable:`
marker and counted in `provenance_gaps`. An unresolved value in any of the
fields above blocks arm pairing.

`worker.model_manifest_sha256` needs the model path, so the control host sends
it:

```bash
uv run python -m grove.mlx_worker preflight --model /Users/grove-worker/grove/cache/...
```

`grove provenance --worker` and a real cycle pass `BASE_MODEL_SOURCE`
automatically. An absent or empty model directory is a named gap, never a
digest over zero files.

The gap policy differs by design:

- **EXP-002**, a single-arm report, may declare `worker.model_manifest_sha256`
  as a permitted partial-provenance gap.
- **EXP-003**, a paired A/B, may not. It lists the path under
  `required_resolved_identity`, so an unresolved manifest refuses the run
  before the database and the sandbox, and exits 2 at check time. When that
  path is unresolved, no other gap is waived either.

All of this is what the worker says about itself. Nothing here proves it ran
that code or loaded those weights; that needs a worker-side signed attestation.

## Back up the evidence

At minimum, copy these together:

- source tree and lockfile;
- evidence database plus `-wal`/`-shm` files if present while the database is open;
- final and original reports;
- candidate dataset and manifest;
- complete adapter directory;
- exact Hugging Face model snapshot or a manifest containing every model-file hash;
- LXD image fingerprint and profile;
- Python, MLX, and MLX-LM versions.

Close Grove processes before taking a simple SQLite file copy. For a live database, use SQLite's backup API rather than copying only the main file.

## Rollback and removal

The automatic experiment performs a temporary rollback by publishing a new manifest. It does not mutate expert lifecycle state.

Permanent operator removal changes an active expert to `removed` and publishes a replacement manifest:

```bash
uv run grove \
  --db /path/to/the/intended.db \
  remove EXPERT_ID \
  --reason 'precise operator reason'
```

This action is intentionally state-changing. Resolve and verify the database and expert ID before running it. Removed artifacts remain on disk and in the ledger; removal is not deletion.

The replacement manifest is the **current deployment minus the removed id**,
not the set of lifecycle-active experts. An expert can be active and
deliberately unplugged — that is what a rollback manifest does — and rebuilding
membership from lifecycle status plugged it back in, so removing one expert
silently deployed another. Removal never plugs anything in.

## Add a new workload safely

1. Define a new task ID, prompt, role, failure family, hidden cases, and canonical solution in `coding_tasks.py` or a new catalog module.
2. Keep expected values only in `PythonSuite`; leave `Task.expected` unset for sandboxed coding tasks.
3. Execute every canonical solution through the real LXD verifier before any model run.
4. Assert task IDs and prompts are unique across every role.
5. Define fresh held-outs before training or tuning.
6. Keep future-stream tasks untouched until their planned checkpoint.
7. Increase replay coverage before broadening an expert's routing profile.
8. Record suite and catalog versions in the deployment manifest.

A genuine second cycle is now implemented as a single-invocation two-cycle mode: `real-cycle --cycles 2` runs the complete first cycle and then, against the same store with the first expert still admitted, deployed and routable — no reset in between — captures, trains and gates the second family (`path_restructure`), recording multi-expert coexistence measurements at every checkpoint. It is sealed as EXP-005 and runs via `scripts/run_exp005.sh <db> <report>`, which propagates the real-cycle exit code (exit 2 included) and then grades the report against the sealed spec. The `real-cycle` command still requires an empty database at the start of an invocation: a *resumable continuation of an existing store* remains unimplemented, and a second cycle must never be faked by resetting or editing an existing evidence database such as `grove-real-v3.db`.

## Pause between experiments: shutdown checklist

The system is safe to leave idle by construction — the MLX worker spawns one process per job and exits (the model leaves the Mac's memory when the job ends), and every sandbox container is deleted in a `finally` block. Before walking away, verify rather than assume:

1. No sandbox containers remain on Agentbox (an empty table is the correct result):

   ```bash
   lxc list
   ```

2. No model or worker process remains on the Mac, and memory is free:

   ```bash
   ssh -i ~/.ssh/grove_worker \
     -o IdentitiesOnly=yes -o BatchMode=yes \
     grove-worker@grove-worker-1 \
     'ps aux | grep -iE "mlx|python" | grep -v grep; memory_pressure -Q | head -3'
   ```

   An empty process list is the correct result. Only standard macOS session daemons (`cfprefsd`, `distnoted`, `trustd`, ...) should be owned by `grove-worker`.

3. Leave `/Users/grove-worker/grove/jobs/` alone. The per-job `spec.json`/`result.json` files are provenance for recorded runs (including the admitted training job) and total under 1 MB.

4. Do not delete `/srv/storage/grove` databases, datasets, adapters, or evaluations; they are the evidence chain. Back them up per the backup section instead.

Nothing needs to be stopped, unloaded, or powered down: if steps 1 and 2 come back empty, the system is fully quiescent and ready to resume with `scripts/preflight.sh`.

Verified in this state on 2026-08-01 after the independent verification probe: zero LXD containers, zero worker processes, 93% of the Mac's 24 GB free.

## Troubleshooting

### SSH or worker preflight fails

Check Tailscale name resolution and the dedicated key:

```bash
tailscale status
ssh -i ~/.ssh/grove_worker \
  -o IdentitiesOnly=yes -o BatchMode=yes \
  grove-worker@grove-worker-1 true
```

Do not replace the restricted key with a personal unrestricted SSH credential.

### Model generation continues after the answer

Confirm `stop_tokens` contains `<|im_end|>` and keep host-side `extract_python()` truncation enabled. Do not solve this only by reducing `max_tokens`; text after the assistant turn is a trust-boundary violation regardless of length.

### Candidate exits with status 1

Inspect the verifier's `stderr_tail`. Exit 1 commonly means invalid syntax, a missing `solve(payload)`, or a runtime exception. It is a model failure unless `infrastructure_error` is set.

### LXD infrastructure error

```bash
lxc version
lxc profile show grove-sandbox
lxc image info grove-python-base
lxc list --format csv -c n,s
```

If a stale `grove-run-*` instance remains after an interrupted process, inspect it before deleting only that exact instance. Never use a broad wildcard or recursive host deletion.

### Output or time limit

`timed_out` and `output_limited` are verifier failures, not partial passes. Preserve their diagnostic metadata. Raise limits only when a workload's legitimate resource needs are understood and the sandbox profile remains bounded.

### Database is not empty

The first-cycle orchestrator refuses nonempty evidence. Choose a new database for another clean first-cycle reproduction. Preserve existing databases; do not use `--reset` to silence the error unless replacement is explicitly intended.

### Adapter directory already exists

Candidate directories are immutable. Generate a new candidate ID. Never train into or overwrite an existing adapter directory.

### Training log says validation is empty

This is expected in the first experiment but is a known limitation. External probation still controls admission. Before hyperparameter sweeps, implement a dedicated validation split rather than repeatedly consulting admission holdouts.

### MLX-LM deprecation warning

The current worker invokes `python -m mlx_lm.lora`. Migrate it to the supported `python -m mlx_lm lora` form and retest before upgrading MLX-LM.

## Required record for every future experiment

Every run should record:

- objective and predeclared hypotheses;
- source revision;
- hardware and dependency versions;
- exact model repository and commit;
- dataset/split IDs and hashes;
- verifier suite and sandbox image fingerprint;
- decoding and training configuration;
- candidate, job, cycle, artifact, and deployment IDs;
- training loss, duration, tokens, and peak memory;
- birth, held-out, replay, routing, future, and rollback metrics;
- all gate thresholds and decision reasons;
- unexpected defects and whether metrics were rerun;
- artifact locations and hashes;
- claims supported, unsupported, and deferred.
