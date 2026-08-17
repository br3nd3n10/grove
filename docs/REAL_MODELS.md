# Real-model integration

The local demonstrator proves Grove's lifecycle. The repository now also contains a real MLX vertical slice that replaces inference, training, and verification while keeping explicit failure-family clustering for auditability.

## Implemented slice

| Concern | Implementation |
|---|---|
| Frozen base | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit@b3252a2f97102b1fb1571fec2c9b27219a8536be` |
| Compute worker | Dedicated non-admin `grove-worker` account on grove-worker-1's Mac Mini |
| Expert | Separate 8-layer LoRA adapter; base weights are never merged or updated |
| Evidence | 20 verified failures, four holdouts, four regressions, two future tasks |
| Verifier | Host-held cases in a fresh unprivileged, networkless LXD container |
| State | SQLite plus append-only ledger and deployment manifests |
| Artifacts | SHA-256-addressed datasets and adapters under `/srv/storage/grove` |

Generation explicitly adds the Qwen `<|im_end|>` token to MLX-LM's stop set. Host-side source extraction truncates at model sentinels as a second boundary, so content outside the assistant turn never reaches the code verifier.

The first completed run admitted one 2.64M-parameter adapter after three rejected candidates. Held-out success increased from 0/4 to 3/4 with no measured replay loss, and rollback restored the baseline. See the [complete experiment report](EXPERIMENT_REPORT_2026-07-31.md) and [runbook](RUNBOOK.md).

## Design of the first slice

Keep the experiment narrow enough that every result can be audited:

- one frozen 1–3B instruct model;
- one repository snapshot or self-contained Python function per task;
- deterministic unit tests as the verifier;
- one failure family with at least 20 corrected training cases;
- a held-out target set and a frozen regression set;
- one LoRA adapter at a time;
- an external route at task granularity, never token-level routing initially.

The first success criterion is not a good aggregate benchmark score. It is one admitted adapter with positive held-out plasticity, zero measured regression, a reproducible artifact, and a clean removal test.

## Inference adapter

Implement `ModelBackend.generate`:

```python
class PeftBackend:
    def __init__(self, base_model, tokenizer, adapter_registry):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.adapters = adapter_registry

    def generate(self, task, expert=None) -> str:
        # Disable all adapters for expert=None.
        # Otherwise load/select only expert.artifact["adapter_path"].
        # Use one pinned decoding configuration for live and evaluation runs.
        ...
```

Required provenance in `expert.artifact`:

```json
{
  "backend": "peft-lora",
  "base_model": "provider/model@immutable_revision",
  "adapter_path": "artifacts/experts/expert_...",
  "adapter_sha256": "...",
  "dataset_sha256": "...",
  "training_config_sha256": "...",
  "parameter_count": 1234567
}
```

The current implementation stores the full training configuration but does not yet store a separate `training_config_sha256` field. Add that digest before treating the example schema above as fully satisfied.

Do not let the inference adapter silently merge LoRA weights into the base. Selection must remain explicit so removal is a store state change, not a model reconstruction.

## Trainer adapter

Implement `ExpertTrainer.train(cluster, candidate_id)` with these steps:

1. Materialize only corrected cluster examples into a versioned JSONL dataset.
2. Record task IDs and content hashes; never include regression or held-out target examples.
3. Start from the pinned frozen base, not from another active expert.
4. Train into a candidate-specific output directory.
5. Hash the adapter and training configuration.
6. Return `ExpertStatus.CANDIDATE` plus a narrow routing profile.

The trainer must not call `store.save_expert(... active ...)`. Admission belongs exclusively to `SleepCycle`.

## Coding verifier

Model-generated code is untrusted. Run tests in an ephemeral sandbox with:

- no network;
- read-only base fixture;
- strict CPU, memory, process, and wall-clock limits;
- a new writable overlay per attempt;
- captured exit status and structured test results;
- secrets absent from the environment;
- sandbox image and test-suite hashes stored in verifier details.

Do not run candidate code in the Grove process or directly on the host. The verifier should translate the sandbox result to `Verification(passed, score, reason, details)`.

## Clusterer

Start with embeddings plus HDBSCAN or agglomerative clustering, but preserve explicit verifier labels when available. Save the embedding model revision and clustering parameters in the cycle report. Cluster quality is part of the evidence: mixed failure families produce weak experts and broad routes.

## Data split

Use four disjoint roles:

| Split | Purpose | Can train? |
|---|---|---:|
| failure/correction | candidate training | yes |
| target holdout | measure whether the failure family generalized | no |
| replay regression | protect already-working behavior and routes | no |
| longitudinal stream | future arriving work | only after it becomes a verified failure in a later cycle |

Content-hash every example before its first use and reject overlaps across roles. Split leakage would make the admission evidence meaningless.

## Operational sequence

1. Pin model, tokenizer, decoding, sandbox, and datasets.
2. Record a baseline longitudinal checkpoint.
3. Run live tasks until a correctable cluster clears the demand gate.
4. Train the candidate on a GPU worker.
5. Load it only in an isolated probation worker.
6. Run forced target evaluation, then full shadow-router replay.
7. Admit or reject from the stored metrics.
8. Record the next longitudinal checkpoint.
9. Run a removal drill and confirm traffic returns to the prior behavior.

For the first real backend, keep scheduling manual. Add a weekly job only after one full train/probation/admit/remove sequence is reproducible from a clean checkout.
