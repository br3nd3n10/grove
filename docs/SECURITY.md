# Security model

Grove treats model output as untrusted data. It is never evaluated in the control-plane process or on the MLX worker.

## Trust boundaries

1. Agentbox owns orchestration, evidence, hidden expected values, admission, and deployment state.
2. grove-worker-1's Mac Mini loads the frozen model and isolated LoRA candidates. Its dedicated `grove-worker` account is non-admin and receives job specifications over a dedicated SSH key.
3. Generated Python crosses back to Agentbox as text and runs as UID/GID 1000 in a newly launched, unprivileged LXD container.

The sandbox profile has no network device, 512 MiB memory, two CPU cores, 64 processes, and no nesting. The process adds CPU-time, file-size, open-file, process-count, wall-time, and captured-output limits. The container is deleted after every program. Integration tests verify network denial, absence of host home/storage paths, timeout handling, and output-flood handling.

## SSH worker controls

The private key is stored only on Agentbox. The authorized-key entry on grove-worker-1's Mac is restricted to Agentbox's Tailscale address and disables agent, port, and X11 forwarding. The worker validates all dataset, adapter, job, and result paths beneath `/Users/grove-worker/grove`; remote adapter downloads are restricted to the adapter store.

## Evidence integrity

Reports and the SQLite store carry local integrity bindings. Every report the
runner writes contains a `run_manifest` and its digest.

The manifest binds **every value the sealed spec's decision rules can read**.
The path set comes from the rules themselves -- each rule's `path`,
`control_path` and `pair_on` -- so a rule cannot read a value nothing committed
to. Naming bound fields by hand is what left `correction_comparison`, `cycle`
and the top-level pairing key unbound, and editing any of those turned a
falsified run into a passing one. On top of that the manifest commits to the
spec digest, the run setup, the provenance digest, the actual training failure
set, the training proposals and each expert's metrics.

Every digest goes through one encoder, Grove canonical JSON v1: UTF-8, sorted
keys, preserved array order, no insignificant whitespace, and outright
rejection of NaN, the infinities and any value JSON cannot represent. The
previous encoder passed `default=str`, so an unrepresentable value was hashed
as its `repr`.

Every evaluation row stores a digest of its own metrics, and the ledger is a
hash chain in which each entry commits to its payload and to the previous
entry's digest.

Verification has three outcomes, and collapsing them is itself a defect:
`clean` (every required digest present and matching), `unverified` (a required
digest or chain link is absent), and `tampered` (a digest disagrees). Only
`clean` is authoritative. A database whose rows predate digest recording — which
is what the 2026-07-31 store is — reports `unverified` with `checked: 0`, and
both the audit CLI and the checker exit 2 rather than treating silence as
agreement. The same rule applies to a report sealed without a provenance
digest.

`scripts/check_experiment_spec.py` recomputes the report bindings and exits 2
on a mismatch. `scripts/audit_evaluation_report.py` recomputes the evaluation
digests and exits 2 on a mismatch, and refuses to write an annotation over
`--db` or `--report`, resolving symlinks and hard links before it opens
anything.

These are unsigned and locally anchored. They detect an edited artifact; they
do not defeat an actor who can rewrite every local artifact and recompute every
digest. An external anchor — an RFC 3161 timestamp token, a signed in-toto/DSSE
envelope over the run manifest, or a Sigstore Rekor entry for its digest — is
the missing control, and Grove does not have one.

Grove cannot evidence *when* a spec was written either, and no longer pretends
to. A self-reported `{"type": "rfc3161", "timestamp": "..."}` map is not
evidence: nothing here parses the timestamp, reads a token, checks a signature
or looks up a registration. Both shipped specs declare
`seal_self_consistent: true` with a null `timing_attestation`, the checker
reports timing as unverified, and any preregistration claim -- with or without
an attestation map -- is refused at check time and before the sandbox at run
time.

Worker provenance is a self-report. `worker.checkout.*`,
`worker.framework_versions_sha256` and `worker.model_manifest_sha256` record
what the worker said about its own code, libraries and model files. The
worktree digest hashes bytes rather than a textual diff, so an untracked or
binary change is visible; the model manifest is what stops two arms of a paired
experiment running different weights while reporting the same checkout.

For the paired EXP-003 the model manifest is a required resolved identity, not
a permitted gap: an unresolved manifest refuses the run before the database and
the sandbox. Nothing here proves an untrusted worker described honestly what it
ran; that needs a worker-side signed attestation.

## Residual risks

LXD containers share the host kernel, so they reduce risk but are not equivalent to a hardware VM against a kernel exploit. Keep LXD and the Ubuntu image patched; use a microVM or separate disposable host for hostile public submissions. Model and dataset files can also be malicious inputs to their parsers, so only approved model snapshots and internally generated JSONL should enter the worker.

The SQLite database contains prompts, responses, corrections, and verifier diagnostics. Do not capture credentials or secrets in tasks. A production deployment should add secret redaction, encryption at rest, retention limits, signed artifacts, and off-host backups.
