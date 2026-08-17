# Grove first real-model experiment report

## Record metadata

| Field | Value |
|---|---|
| Experiment date | 2026-07-31 UTC |
| Status | Completed; one expert admitted, rollback tested, expert restored |
| Frozen base | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` |
| Resolved base snapshot | `b3252a2f97102b1fb1571fec2c9b27219a8536be` |
| Admitted expert | `expert_979511319695` |
| Current deployment | `deployment_cfff527d63d2`, sequence 7 |
| Primary evidence database | `/srv/storage/grove/grove-real-v3.db` |
| Consolidated report | `/srv/storage/grove/evaluations/final-real-cycle.json` |
| Final outcome | Held-out capability increased from 0/4 to 3/4 with no measured loss on previously successful replay tasks |

This document records the complete first real-model Grove experiment, including the unsuccessful attempts and engineering defects discovered along the way. It distinguishes what the experiment demonstrated from what remains unproven.

For a diagram-first explanation of the topology, lifecycle, data boundaries, four candidates, benchmark, rollback, evidence lineage, and next steps, see `VISUAL_GUIDE.md`.

## Executive summary

Grove successfully completed a real failure-to-expert lifecycle using a frozen 1.5B coding model, deterministic host-held verification, LoRA training on an Apple M4 Mac Mini, and disposable networkless LXD sandboxes on Agentbox.

The final candidate learned from 20 verifier-backed failures, fixed 18 of them, passed 3 of 4 held-out paraphrases, preserved both behaviors the base had previously demonstrated, passed the admission policy, and became a separately removable 2.64M-parameter expert. The aggregate eight-task benchmark increased from 25.0% to 62.5%. A deployment rollback removed the expert and restored baseline behavior; a later manifest restored the expert and a smoke test reproduced the 3/4 held-out result.

This is evidence that the first narrow vertical slice works. It is not yet evidence that Grove can improve continuously for months, manage many interacting experts, or generalize to arbitrary new algorithms.

## Research questions and outcomes

| Question | Outcome | Evidence |
|---|---|---|
| Can the control plane run locally without a GPU? | Yes | Agentbox handled evidence, routing, sandboxing, admission, storage, and benchmarks on CPU. |
| Can a 24 GB Apple Silicon machine supply the training compute? | Yes | All four LoRA adapters trained through MLX on grove-worker-1's Mac Mini; the admitted run took 34.82 seconds and peaked near 1.86 GB MLX memory. |
| Can verified failures create a useful removable expert without editing the base? | Yes, for this failure family | The final adapter fixed 18/20 birth failures and 3/4 held-out paraphrases. The base model files were not updated or merged. |
| Can deterministic verification replace a larger judge model? | Yes, for these coding tasks | Expected results remained host-side and generated programs were scored by hidden cases in LXD. No judge API was used. |
| Will strict gates prevent weak candidates from deploying? | Yes | Three candidates were rejected before the fourth met all thresholds. |
| Can the expert be removed without reconstructing the model? | Yes | Rollback changed only the deployment manifest, returned expert routing to 0%, and restored baseline capability. |
| Did the expert learn the entire escaped-path domain? | No | It failed two training paraphrases partially and both harder future-stream algorithms. |

## System that was built

The implemented lifecycle is:

```text
task -> external route -> frozen base plus optional adapter -> verifier
     -> success replay or verified failure
     -> failure clustering -> demand gate -> isolated LoRA training
     -> birth-task probation + held-out probation + routed replay
     -> admit/reject -> append-only deployment manifest -> rollback/removal
```

### Component responsibilities

| Component | Implementation | Responsibility |
|---|---|---|
| Operational store | `src/grove/store.py` | Tasks, attempts, failures, corrections, dataset roles, artifacts, cycles, deployments, and ledger |
| Runtime | `src/grove/runtime.py` | Routing, batched generation, verification, and evidence capture |
| Router | `src/grove/routing.py` | Auditable tag/keyword/profile selection among deployed experts |
| Sleep cycle | `src/grove/sleep.py` | Demand gate, training request, probation, admission/rejection, deployment, removal |
| MLX backend | `src/grove/mlx_backend.py` | Remote adapter-aware inference and assistant-turn source extraction |
| MLX trainer | `src/grove/mlx_trainer.py` | Versioned JSONL creation, remote LoRA training, artifact transfer and provenance |
| Mac worker | `src/grove/mlx_worker.py` | Constrained MLX inference/training process under a dedicated account |
| Sandbox | `src/grove/sandbox.py` | Fresh LXD container, resource limits, output capture, and cleanup per program |
| Coding verifier | `src/grove/verifiers.py` | Host-held cases and structural result comparison |
| Benchmark | `src/grove/benchmark.py` | Capability, plasticity, stability, forgetting, routing, and growth curves |
| Real experiment | `src/grove/experiment.py` | Reproducible first-cycle orchestration and audit report generation |

## Hardware and software environment

### Agentbox control plane

| Property | Value |
|---|---|
| OS/kernel | Ubuntu, Linux `7.0.0-28-generic`, x86_64 |
| CPU | Intel Core i7-4770K, 4 cores / 8 threads |
| RAM | 30 GiB |
| Swap | 8 GiB |
| OS volume at audit | 193 GiB available |
| Data volume at audit | 868 GiB available at `/srv/storage` |
| GPU | None required |
| Container manager | LXD 6.9 |
| Sandbox image | Private Ubuntu 24.04 amd64 image, alias `grove-python-base` |
| Sandbox image fingerprint | `3c5997136b68613f330b4f635481769ba6c30a9d08c40bc83f9a024923162ca8` |

### grove-worker-1's Mac Mini ML worker

| Property | Value |
|---|---|
| Model | Mac Mini, Apple M4 |
| CPU/GPU | 10 CPU cores; integrated Apple GPU used by MLX |
| Unified memory | 24 GB |
| OS | macOS 26.4, Darwin 25.4.0 |
| Free storage at final preflight | Approximately 336 GiB |
| Worker account | Dedicated non-admin `grove-worker` |
| Python | 3.14.3 |
| MLX | 0.32.0 |
| MLX-LM | 0.31.3 |
| MLX default device | `Device(gpu, 0)` |

### Connection and worker boundary

- Agentbox reaches the Mac through Tailscale and a dedicated SSH key.
- The authorized key is source-restricted to Agentbox's Tailscale address and disables agent, port, and X11 forwarding.
- The worker account is non-admin.
- Worker-controlled paths are restricted beneath `/Users/grove-worker/grove`.
- Adapter downloads accept only one safe adapter directory name beneath the adapter store.
- Model output returns to Agentbox as text; it is never executed on the Mac worker.

## Sandbox and verifier design

Every generated Python program runs in a newly launched LXD container. The profile is unprivileged, has no network device, disables nesting, and limits the container to two CPU cores, 512 MiB memory, and 64 processes. Inside the container the candidate runs as UID/GID 1000 under additional `prlimit` constraints:

- 4 seconds CPU time;
- 1 MiB file-size limit;
- 64 open files;
- 32 processes;
- 5 seconds host-enforced wall time by default;
- 1 MB combined captured-output limit by default.

The verifier supplies JSON inputs through stdin and compares JSON outputs against expected values that remain on Agentbox. Candidate prompts do not contain expected results. Integration tests demonstrated:

- valid JSON execution;
- no usable network connection;
- no visibility of Agentbox SSH or Grove storage paths;
- termination of an infinite loop;
- termination and truncation of an output flood.

LXD shares the Agentbox kernel. This is appropriate for the controlled internal experiment, but a microVM or disposable machine is recommended for hostile public submissions.

## Workload and data governance

### Final workload split

| Role | Count | Purpose | Trains? |
|---|---:|---|---:|
| Regression | 4 | Existing general Python behavior | No |
| Training | 20 | Verified escaped-path failures and corrections | Yes |
| Held-out target | 4 | Unseen paraphrases and hidden cases for admission | No |
| Future stream | 2 | Harder unseen algorithms for the next cycle | No |

The 20 training prompts cover six operations: get, exists, set, delete, flatten, and unflatten. Multiple independently worded prompts prevent a one-prompt-per-operation adapter from appearing more general than it is.

The final four holdouts are unseen paraphrases of lookup, membership, assignment, and flattening with different hidden inputs. The future stream contains `path_rename` and `path_project`, which require new compositions rather than merely transferring escaped-path parsing.

### Governance rules

- A `train` assignment now requires a matching accepted correction tied to the same failure.
- Task identity and model-input content hashes prevent a task or identical prompt from silently crossing roles.
- Training materialization includes only corrected failures in the selected cluster.
- Regression, held-out, and future examples are never written into candidate JSONL.
- Dataset and adapter directories are immutable per candidate ID.

The final dataset hash shared by the 40-step and 60-step 20-example candidates is:

```text
e817012ccc8ee874162f224e940e02f62eecbdefd8f44f7b4636640cca97884e
```

## Admission policy

The real experiment declared these thresholds before the final candidate decision:

| Gate | Threshold |
|---|---:|
| Minimum corrected failure cluster | 3 failures |
| Birth-task fix rate | At least 80% |
| Plasticity gain over frozen behavior | At least 50 percentage points |
| Held-out fix rate | At least 75% |
| Routed replay regression | 0% allowed |
| Held-out set required | Yes |

A candidate remains non-routable until every check passes. Rejected candidates have no path back to active state; reconsideration requires a new candidate and new evidence.

## Chronological experiment log

### Phase 0: deterministic control-plane scaffold

Before attaching a real model, the lifecycle was implemented and tested with a deterministic arithmetic backend. That proved capture, clustering, demand gating, admission, router-regression rejection, benchmarking, removal, and audit-ledger behavior without claiming simulated training was a LoRA result.

### Phase 1: machine provisioning and real boundaries

Agentbox received the `/srv/storage/grove` artifact hierarchy, LXD image/profile, and integration tests. grove-worker-1's Mac received a dedicated account, directory hierarchy, Python environment, MLX/MLX-LM, model cache, and restricted SSH access. The control plane and worker passed end-to-end preflight.

### Attempt A: six examples, aggressive training

| Field | Value |
|---|---|
| Expert | `expert_f7e68790c1e8` |
| Dataset | 6 examples, hash `e84c98e25d796c48c946291fa86fddfa64c3878bfd7ee4f879f3d626a8e0cfd4` |
| Training | 120 iterations, learning rate `1e-4`, 8 LoRA layers |
| Duration | 73.52 seconds |
| Final logged loss | 0.000 |
| Peak MLX memory | 1.871 GB |
| Recorded probation | 0/6 birth tasks, 0/2 holdouts |
| Decision | Rejected |

The recorded 0/6 result exposed an inference defect rather than a total learning failure. The adapter often emitted the correct program, then emitted the textual Qwen `<|im_end|>` sentinel, then continued producing unrelated text until the token limit. Source extraction originally removed the sentinel instead of truncating at it, so correct code plus trailing junk reached the sandbox and failed syntax checks.

MLX-LM's loaded stop set contained `<|endoftext|>` but not the Qwen chat-turn token, even though the tokenizer exposed `<|im_end|>` separately. The repair had two layers:

1. add `<|im_end|>` to the MLX tokenizer's EOS set so generation stops promptly;
2. truncate host-side extraction at the first model sentinel so text outside the assistant turn can never reach the verifier.

The rejection was preserved in the audit record. A diagnostic replay after the repair showed the adapter passed all 6 training operations, but it passed 0/2 novel algorithm holdouts and, when forcibly applied outside its intended route, 0/4 regression tasks. The adapter had learned the six examples but was badly overfit.

### Attempt B: six examples, gentler training

| Field | Value |
|---|---|
| Expert | `expert_af9693b11576` |
| Dataset | Same 6 examples as Attempt A |
| Training | 40 iterations, learning rate `2e-5`, 8 LoRA layers |
| Duration | 24.17 seconds |
| Final logged loss | 0.002 |
| Birth-task result | 5/6, 83.33% |
| Original holdout result | 0/2 |
| Routed replay regression | 0% |
| Decision | Rejected for held-out failure |

Reducing update strength substantially improved retention and birth-task performance, but six examples were still not enough to transfer to the two held-out algorithms. This showed that lowering the learning rate alone would not solve the data-design problem.

### Workload redesign

The initial two holdouts, rename and project, tested both escaped-path parsing and wholly new algorithms. That confounded the capability being measured. They were retained as a future stream, while four new held-out paraphrases were created to isolate whether escaped-path behavior transferred to unseen wording and cases.

The training set was expanded from six prompts to 20 verified prompts across the same six operations. All 30 canonical implementations across every role were executed in real LXD containers and passed all hidden cases before new training began.

This design change improves construct validity but limits strict comparability between Attempts A/B and Attempts C/D. It was made before the final two candidates and is documented rather than hidden.

### Attempt C: 20 examples, two effective passes

| Field | Value |
|---|---|
| Expert | `expert_d26badfaf51d` |
| Dataset | 20 examples, final dataset hash |
| Training | 40 iterations, learning rate `2e-5`, 8 LoRA layers |
| Duration | 23.56 seconds |
| Final logged loss | 0.013 |
| Birth-task result | 15/20, 75% |
| Held-out result | 4/4, 100% |
| Routed replay regression | 0% |
| Decision | Rejected because 75% was below the 80% birth-task threshold |

This candidate generalized to every holdout but remained underfit on its own corrected failures. The gate rejected it by one required success. The threshold was not weakened after seeing the result.

### Attempt D: 20 examples, three effective passes

| Field | Value |
|---|---|
| Expert | `expert_979511319695` |
| Dataset | Same 20 examples as Attempt C |
| Training | 60 iterations, learning rate `2e-5`, 8 LoRA layers |
| Duration | 34.82 seconds |
| Trainable parameters | 2,637,824 of 1,543,714,000, or 0.171% |
| Final logged loss | 0.003 |
| Peak MLX memory | 1.860 GB |
| Birth-task result | 18/20, 90% |
| Plasticity gain | 90 percentage points |
| Held-out result | 3/4, 75% |
| Routed replay regression | 0% |
| Decision | Admitted |

Increasing from 40 to 60 iterations improved birth-task performance from 75% to 90%, while held-out performance fell from 100% to 75%. With only four holdouts this difference is not statistically stable, but it demonstrates why training loss alone is not an admission metric and why both birth and held-out gates are needed.

## Detailed final results

### Live capture

The frozen base processed 24 live tasks: four regression tasks and 20 training tasks.

- 2 passed;
- 22 failed;
- all 20 escaped-path failures received independently sandbox-verified corrections;
- the two failed regression tasks remained unresolved and were not used to train the escaped-path expert.

### Probation

The final candidate achieved:

```text
birth tasks:          18 / 20 = 90%
held-out targets:      3 /  4 = 75%
successful replay:     2 /  2 = 100% before and after
measured regression:               0%
```

A later deterministic replay identified the residual birth-task failures:

| Task | Result |
|---|---:|
| `path_delete_v2` | 2/3 hidden cases |
| `path_delete_v3` | 2/3 hidden cases |

All other 18 training tasks passed all hidden cases. The residual cluster is therefore not generic path parsing; it is deletion and pruning behavior under paraphrased instructions.

### Longitudinal benchmark

| Checkpoint | Capability | Held-out plasticity | Regression stability | Forgetting | Deployed experts | Added parameters |
|---|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 25.0% (2/8) | 0.0% (0/4) | 50.0% (2/4) | 0% | 0 | 0 |
| After growth | 62.5% (5/8) | 75.0% (3/4) | 50.0% (2/4) | 0% | 1 | 2,637,824 |
| Corrected rollback | 25.0% (2/8) | 0.0% (0/4) | 50.0% (2/4) | 0% | 0 | 0 |

The regression score did not become 100%; the base already failed two of the four regression tasks. The claim is no measured forgetting of previously successful behavior, not universal regression-suite competence.

After growth, all four held-out prompts routed to the expert. One of four regression prompts also routed to it. That unnecessary route did not change the pass count, but it reveals that the current tag/keyword route is broader than desirable.

### Restored deployment smoke test

After rollback restoration, all four holdouts again routed to `expert_979511319695`:

| Holdout | Result |
|---|---:|
| `holdout_path_lookup` | Pass |
| `holdout_path_membership` | Fail |
| `holdout_path_assign` | Pass |
| `holdout_path_flatten` | Pass |

This reproduced the 3/4 probation result and confirmed that restoration changed real routing behavior rather than only writing metadata.

### Future stream

The active expert did not solve the two harder future algorithms:

| Task | Result |
|---|---:|
| `path_rename` | 1/3 hidden cases |
| `path_project` | 0/2 hidden cases |

This is a useful negative result. The adapter learned the trained operation family and transferred to closely related paraphrases, but did not acquire open-ended compositional competence across all escaped-path algorithms.

### Remaining unresolved failures

The final database contains four unresolved live failures:

- the two partially unresolved escaped-path delete prompts;
- two core regression failures the base never passed and that did not have accepted corrections in this cluster.

## Deployment and rollback record

The final database contains seven append-only manifests:

| Sequence | Expert set | Purpose |
|---:|---|---|
| 1 | None | Frozen baseline |
| 2 | Final expert | Admission |
| 3 | None | First rollback drill |
| 4 | Final expert | Restore after first drill |
| 5 | None | Corrected deployment-aware rollback measurement |
| 6 | Final expert | Restore after corrected measurement |
| 7 | Final expert | Pin the exact resolved base snapshot after audit |

There is no rejected-to-active or removed-to-active state shortcut. The first three adapter artifacts remain rejected in their original databases.

## What went well

1. **The hybrid machine design worked.** Agentbox did not need a GPU, and the Mac Mini had ample memory and storage for this model and adapter size.
2. **Training was inexpensive.** The final adapter trained in under 35 seconds and used about 1.86 GB peak MLX memory.
3. **Verifier-backed learning worked without a judge API.** Deterministic hidden cases were enough to create and admit a useful coding expert.
4. **Admission gates behaved correctly.** Three weak or incomplete candidates were rejected; the 75% birth-task candidate was not admitted by lowering a threshold after the fact.
5. **Modularity was real.** The expert was a separate approximately 10.6 MB artifact. The base was never merged or modified.
6. **Rollback was behaviorally observable.** Expert routing and plasticity dropped to zero while regression behavior returned to baseline, then restoration recovered expert behavior.
7. **Negative results remained visible.** Failed adapters, reports, training logs, database ledgers, future failures, and metric repairs were retained.
8. **The sandbox caught malformed output safely.** Model code failures remained container failures rather than host incidents.

## What did not go well, and what changed

### 1. Chat stop-token mismatch

**Symptom:** correct source followed by `<|im_end|>` and unrelated continuation text failed syntax checks.

**Cause:** MLX-LM's stop set omitted the Qwen chat terminator for this model metadata combination.

**Repair:** add the chat terminator to the tokenizer EOS set and truncate host extraction at the first sentinel. Three unit tests cover the extraction boundary.

### 2. Six-example overfitting

**Symptom:** the first corrected replay passed 6/6 birth tasks but failed novel operations and performed poorly when forced outside its route.

**Cause:** only one canonical prompt per operation, combined with 120 iterations at `1e-4`.

**Repair:** lower learning rate, reduce iterations, then expand to 20 independently worded examples.

### 3. Initial holdouts mixed two capabilities

**Symptom:** rename and project remained at 0/2 even when the six learned operations improved.

**Cause:** the holdouts required new algorithms as well as escaped-path parsing.

**Repair:** retain them as future tasks and use unseen paraphrases for the admission holdout. This change is explicitly recorded as an experiment-design revision.

### 4. The 40-step 20-example candidate underfit

**Symptom:** 15/20 birth tasks despite 4/4 held-out success.

**Repair:** keep the gate unchanged and train a new 60-step candidate on the same dataset.

### 5. No MLX validation split

Every training log warns that the validation set is empty. Grove's external probation prevented deployment based on loss, but hyperparameter selection would be cleaner with a separate training-time validation split that remains distinct from admission targets.

### 6. Router breadth

The admitted profile routed one regression prompt to the expert. It caused no measured score loss, but larger deployments need a narrower or learned route plus substantially larger replay coverage.

### 7. Regression coverage was too small

Only two successful base behaviors were available for routed replay. Zero regression over two examples is encouraging but weak evidence. A production claim requires tens or hundreds of stable successes across diverse capabilities.

### 8. Rollback growth-counter defect

The first rollback report correctly showed baseline behavior and zero expert routing, but its `active_experts` and `added_parameters` fields counted lifecycle-active experts rather than experts in the current deployment manifest.

**Repair:** the benchmark now counts `routable_experts()`. A regression test covers an active-but-unplugged expert. The rollback was rerun and correctly recorded zero deployed experts and zero added parameters. The original report remains preserved, and the consolidated report marks the corrected measurement.

### 9. Split hash representation mismatch

Early experiment code hashed raw prompts for held-out roles but a prompt-plus-completion JSON object for training roles. An identical prompt serialized differently could therefore evade the content-hash collision.

No actual prompts overlapped; the catalog now tests all 30 prompts for uniqueness. The implementation was repaired so model input is the split identity, task IDs cannot receive a second role, and `train` requires an accepted matching correction.

### 10. Base revision was pinned after the final run

The run referred to the Hugging Face repository name. Inspection showed one cached snapshot, commit `b3252a...`, which was the model actually loaded. The code now uses its exact snapshot path and deployment sequence 7 records the full repository-plus-commit identifier. Future clean runs start pinned rather than adding the pin after audit.

### 11. Remote adapter path validation was initially prefix-only

The worker already constrained its own paths, but the host download check accepted any string beginning with the adapter prefix. It now accepts only one safe alphanumeric/dash/underscore adapter directory name. Four traversal and shell-like cases are tested.

### 12. Verification was slow

Fresh LXD launch and deletion dominate wall time. The final full experiment took roughly 22.5 minutes even though training took only 34.8 seconds. This is an intentional isolation cost for the first slice. Future optimization can maintain one fresh container per candidate program while parallelizing a bounded number of independent programs or using prewarmed microVM snapshots.

### 13. MLX-LM invocation deprecation

The training logs warn that `python -m mlx_lm.lora` is deprecated. It still works in MLX-LM 0.31.3, but the worker should migrate to the supported `python -m mlx_lm lora` invocation before upgrading dependencies.

## Design decisions and rationale

### Start from a small pretrained base

Training a useful language model from random initialization is a different, much larger experiment. A pretrained base isolates the continual-learning question: can verified experience add removable capability without rewriting existing weights?

### Keep experts as separate LoRA artifacts

The adapter is explicit, small, independently hashed, and removable. It is never silently merged into the base. This makes rollback a deployment-state operation rather than model reconstruction.

### Use task-level routing first

An external interpretable route is easier to audit and undo than learned token-level MoE routing. The router is knowingly simple; precision improvements belong after the lifecycle is proven.

### Let deterministic verifiers, not models, decide coding correctness

Hidden executable cases are cheaper, repeatable, and less ambiguous than another language model for this workload. This decision does not generalize to tasks without deterministic success criteria.

### Treat evaluation as non-learning

Probation and benchmark attempts are ephemeral and do not create new failure evidence. This prevents evaluation from silently feeding its own answers into later training.

### Preserve rejected artifacts and defective reports

Research traceability is more valuable than presenting a clean first try. Repairs append new evidence instead of deleting the evidence that exposed the defect.

## Interpretation

The strongest supported conclusion is:

> On one controlled escaped-path coding family, a frozen 1.5B model plus a separately trained 2.64M-parameter LoRA adapter improved held-out task success from 0/4 to 3/4 while preserving two previously successful replay behaviors, and the capability could be removed and restored through deployment manifests.

The experiment also supports several engineering conclusions:

- Apple MLX is sufficient for rapid small-model LoRA cycles on a 24 GB M4 Mac Mini.
- The expensive portion of this design is verification isolation, not training.
- A strict gate is useful only when birth, held-out, regression, and routing behavior are all measured separately.
- Low training loss is not evidence of safe or general learning.
- More examples improved the balance far more than aggressive training on six examples.

## Threats to validity and claims not made

1. There is one failure family and one admitted expert.
2. Four held-out tasks are too few for statistical confidence.
3. Hyperparameters were adjusted across sequential experiments, so this is engineering validation rather than a blinded benchmark result.
4. The holdout design was revised after the first two attempts exposed a construct-validity problem.
5. There is no naive full-fine-tune control condition yet.
6. The regression suite contains only four tasks, of which the base passes two.
7. The router is profile-based and has already shown one unnecessary regression route.
8. The harder future tasks remain unsolved.
9. The experiment does not test expert composition, conflict, capacity growth, pruning, or many-cycle storage behavior.
10. The system is manually scheduled and is not yet a production service.
11. LXD is not a hardware security boundary.
12. Results apply to deterministic Python-function tasks, not arbitrary agent work.

## Recommended next experiments

### Priority 0: preserve and stabilize this result

- Back up the final database, reports, datasets, adapters, and exact model snapshot off-host.
- Add a machine-readable environment manifest and hashes for source revision, sandbox image, model files, and dependencies.
- Switch to the supported MLX-LM CLI invocation.
- Add a small training-time validation split without consuming admission targets.

### Priority 1: run a second genuine growth cycle

- Treat `path_delete_v2` and `path_delete_v3` as a residual cluster only if more live failures confirm demand.
- Expand rename/project into at least 20 live, corrected examples across several related operations.
- Create entirely new held-outs before training the next expert.
- Keep the current expert frozen and train the second candidate from the same frozen base, not from the first adapter.
- Test both experts independently and together through the router.

### Priority 1: strengthen stability evidence

- Collect at least 50 to 100 successful base tasks for replay.
- Cover syntax, collections, strings, recursion, transformations, error handling, and multi-step functions.
- Report per-task route changes as well as aggregate score changes.
- Add a route precision/recall metric and a base-vs-expert abstention analysis.

### Priority 2: establish research controls

- Compare LoRA expert growth with naive full fine-tuning on the same corrections.
- Repeat each configuration over several seeds.
- Predeclare thresholds and fresh holdouts before each run.
- Measure confidence intervals rather than only point estimates.
- Track wall time, energy, memory, artifact size, and verification cost separately.

### Priority 2: operationalize repeated cycles

- Add an explicit resumable real-cycle command for nonempty databases.
- Add scheduling, locks, alerts, retention, backups, and interrupted-job recovery.
- Add artifact signing and a deployment health monitor.
- Move hostile execution to microVMs before accepting public workloads.

## Evidence and artifact index

### Final successful run

| Artifact | Location or hash |
|---|---|
| Consolidated report | `/srv/storage/grove/evaluations/final-real-cycle.json` |
| Report SHA-256 | `849f54c87bd71c7bbc60507cf4b3fa9d6c56db217a16cf98a6ee85eb52400c90` |
| Evidence database | `/srv/storage/grove/grove-real-v3.db` |
| Database SHA-256 after sequence 7 | `33fbb4e51a7afcca7e65b96175687d960f014379b7f8133a28528b28459faa0f` |
| Adapter directory | `/srv/storage/grove/experts/expert_979511319695` |
| Aggregate adapter artifact hash | `301519521c191776446e4ea23cd5aabf3546363d9903d1e57a5d78757c9f91e6` |
| Adapter weights SHA-256 | `1cf1e5114a9e8b46f89ddf2bb3cce174580cf6cf481f6ed21328405b6d799e33` |
| Adapter artifact size | 10,565,604 bytes |
| Dataset directory | `/srv/storage/grove/datasets/expert_979511319695` |
| Dataset SHA-256 | `e817012ccc8ee874162f224e940e02f62eecbdefd8f44f7b4636640cca97884e` |
| Training job | `job_5bd89f425646` |

The consolidated report hash above was recorded immediately after final deployment sequence 7. Any later annotation of that JSON will necessarily produce a new hash; the database and artifact ledgers remain the authoritative state records.

### Preserved unsuccessful runs

| Run | Evidence |
|---|---|
| Six-example attempts A and B | `/srv/storage/grove/grove-real.db`, `/srv/storage/grove/evaluations/first-real-cycle.json` |
| 20-example attempt C | `/srv/storage/grove/grove-real-v2.db`, `/srv/storage/grove/evaluations/first-real-cycle-v2.json` |
| Original successful report before audit annotations | `/srv/storage/grove/evaluations/first-real-cycle-v3.json` |

### Documentation

- `README.md`: project entry point and commands;
- `PLAN.md`: original research map;
- `docs/ARCHITECTURE.md`: ownership and lifecycle invariants;
- `docs/REAL_MODELS.md`: real-model integration design;
- `docs/SECURITY.md`: threat model and residual security risks;
- `docs/RUNBOOK.md`: reproduction, inspection, and recovery procedures;
- `docs/VISUAL_GUIDE.md`: Mermaid charts covering the full experiment;
- `docs/ORIGINAL_FILES.md`: hashes and links for the untouched initial README and plan;
- `docs/data/posthoc-training-replay.json`: machine-readable admitted-adapter replay results.

## Current state at close of report

- `expert_979511319695` is lifecycle-active and present in current deployment sequence 7.
- The current manifest pins the exact base snapshot, router version, verifier suite version, and decoding configuration.
- Three prior adapters remain rejected and non-routable.
- No generated program remains running, and no `grove-run-*` LXD container remains after verification.
- The complete automated suite passed 25/25 tests at the end of implementation.
- The next meaningful milestone is a fresh second learning cycle with broader replay, not additional tuning on the four current holdouts.
