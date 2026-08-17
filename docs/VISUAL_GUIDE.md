# Grove visual guide

These Mermaid diagrams summarize the architecture, experimental history, results, evidence, and remaining research work. Sections 1–3 describe the system; sections 4–8 describe the first real experiment (2026-07-31) and its independent verification; sections 9–12 describe the sealed experiments EXP-002 through EXP-005 (2026-08-08 through 2026-08-11); section 13 maps the open questions. The complete quantitative records are `EXPERIMENT_REPORT_2026-07-31.md`, `VERIFICATION_2026-08-01.md`, `EXPERIMENT_PROTOCOL.md`, and the dated notes in `../research/`.

## 1. Two-machine system and trust boundaries

```mermaid
flowchart LR
    USER[Task stream]

    subgraph AB[Agentbox control plane]
        direction TB
        RT[Runtime and router]
        DB[(SQLite evidence and ledger)]
        VF[Host-held verifier cases]
        LXD[Fresh networkless LXD container]
        ART[(Datasets adapters reports)]

        RT --> DB
        RT --> VF
        VF --> LXD
        DB --> ART
    end

    subgraph MAC[grove-worker-1s Mac Mini ML worker]
        direction TB
        SSH[Restricted non-admin SSH worker]
        BASE[Frozen Qwen 1.5B base]
        MLX[MLX inference and LoRA training]
        CAND[Isolated candidate adapter]

        SSH --> MLX
        BASE --> MLX
        MLX --> CAND
    end

    USER --> RT
    RT -->|Job specification only| SSH
    MLX -->|Generated text and artifact metadata| RT
    CAND -->|Hashed adapter transfer| ART
    LXD -->|Structured pass or fail| RT

    classDef trusted fill:#1d3557,color:#fff,stroke:#0b1f33
    classDef compute fill:#457b9d,color:#fff,stroke:#1d3557
    classDef untrusted fill:#7b2d26,color:#fff,stroke:#4d1713
    classDef store fill:#2d6a4f,color:#fff,stroke:#183b2c
    class RT,DB,VF,ART trusted
    class SSH,BASE,MLX,CAND compute
    class LXD untrusted
    class DB,ART store
```

The model output is untrusted. It crosses back to Agentbox as text and is executed only inside the disposable LXD boundary. Hidden expected values never go to the Mac or into model prompts.

## 2. Verified expert-growth lifecycle

```mermaid
flowchart TD
    T[Receive task] --> R{External router}
    R -->|No expert| B[Frozen base]
    R -->|Matching deployed expert| E[Frozen base plus adapter]
    B --> G[Generate response]
    E --> G
    G --> V{Deterministic verifier}

    V -->|Pass| S[(Successful replay evidence)]
    V -->|Fail| F[(Verified failure evidence)]
    F --> C[Cluster recurring failures]
    C --> D{Demand gate}
    D -->|Too little or unsafe evidence| SKIP[Log and wait]
    D -->|Enough verified corrections| TRAIN[Train isolated LoRA candidate]

    TRAIN --> P1{Fix at least 80 percent of birth tasks}
    P1 -->|No| REJECT[Reject and keep artifact non-routable]
    P1 -->|Yes| P2{Gain at least 50 percentage points}
    P2 -->|No| REJECT
    P2 -->|Yes| P3{Pass at least 75 percent of holdouts}
    P3 -->|No| REJECT
    P3 -->|Yes| P4{Zero routed replay regression}
    S --> P4
    P4 -->|No| REJECT
    P4 -->|Yes| ADMIT[Admit expert]

    ADMIT --> DEPLOY[Append deployment manifest]
    DEPLOY --> R
    DEPLOY --> ROLLBACK[Rollback or remove without editing base]

    classDef decision fill:#b07d2b,color:#fff,stroke:#704c14
    classDef pass fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef fail fill:#7b2d26,color:#fff,stroke:#4d1713
    classDef evidence fill:#1d3557,color:#fff,stroke:#0b1f33
    class R,V,D,P1,P2,P3,P4 decision
    class ADMIT,DEPLOY pass
    class REJECT fail
    class S,F evidence
```

Rejected candidates cannot be switched to active later. A reconsideration is a new candidate with a new artifact and a new probation decision.

## 3. Data isolation and evidence flow

```mermaid
flowchart LR
    LIVE[Live tasks] --> BASE[Frozen base execution]
    BASE --> VERIFY{Hidden-case verifier}

    VERIFY -->|Pass| REG[(Replay successes)]
    VERIFY -->|Fail| FAIL[(Failure log)]
    FAIL --> CORR{Canonical correction passes verifier}
    CORR -->|No| UNRES[Unresolved failure]
    CORR -->|Yes| TRAIN[(Training split)]

    TARGET[(Held-out target split)] --> PROB[Probation only]
    REG --> PROB
    TRAIN --> LORA[LoRA training]
    LORA --> PROB
    FUTURE[(Future stream)] --> FUTUREPROBE[Post-decision probe only]

    TRAIN -. never overlaps .-> TARGET
    TRAIN -. never overlaps .-> REG
    TRAIN -. never overlaps .-> FUTURE

    classDef train fill:#457b9d,color:#fff,stroke:#1d3557
    classDef eval fill:#b07d2b,color:#fff,stroke:#704c14
    classDef future fill:#6d597a,color:#fff,stroke:#3f3048
    classDef evidence fill:#2d6a4f,color:#fff,stroke:#183b2c
    class TRAIN,LORA train
    class TARGET,REG,PROB eval
    class FUTURE,FUTUREPROBE future
    class FAIL,CORR evidence
```

Final split sizes were 20 training prompts, 4 held-out targets, 4 regressions, and 2 future tasks. Only accepted corrections can receive the training role.

## 4. Four real candidate attempts

```mermaid
flowchart LR
    A[Attempt A<br/>6 examples<br/>120 steps at 1e-4<br/>Recorded 0 of 6<br/>Post-fix replay 6 of 6<br/>Holdout 0 of 2<br/>REJECTED]
    B[Attempt B<br/>6 examples<br/>40 steps at 2e-5<br/>Birth 5 of 6<br/>Holdout 0 of 2<br/>REJECTED]
    C[Attempt C<br/>20 examples<br/>40 steps at 2e-5<br/>Birth 15 of 20<br/>Holdout 4 of 4<br/>REJECTED]
    D[Attempt D<br/>20 examples<br/>60 steps at 2e-5<br/>Birth 18 of 20<br/>Holdout 3 of 4<br/>ADMITTED]

    A -->|Fix stop boundary<br/>reduce update strength| B
    B -->|Expand evidence<br/>separate transfer from novel algorithms| C
    C -->|Keep gate<br/>increase to three effective passes| D

    classDef reject fill:#7b2d26,color:#fff,stroke:#4d1713
    classDef admit fill:#2d6a4f,color:#fff,stroke:#183b2c
    class A,B,C reject
    class D admit
```

```mermaid
flowchart TB
    ALOSS[Attempt A<br/>Final loss 0.000<br/>73.52 seconds<br/>Overfit and stop-token defect]
    BLOSS[Attempt B<br/>Final loss 0.002<br/>24.17 seconds<br/>Birth improved but no holdout transfer]
    CLOSS[Attempt C<br/>Final loss 0.013<br/>23.56 seconds<br/>Perfect holdouts but underfit birth tasks]
    DLOSS[Attempt D<br/>Final loss 0.003<br/>34.82 seconds<br/>Passed every admission gate]

    ALOSS --> FINDING[Training loss alone<br/>does not determine admission]
    BLOSS --> FINDING
    CLOSS --> FINDING
    DLOSS --> FINDING

    classDef attempt fill:#457b9d,color:#fff,stroke:#1d3557
    classDef finding fill:#b07d2b,color:#fff,stroke:#704c14
    class ALOSS,BLOSS,CLOSS,DLOSS attempt
    class FINDING finding
```

Attempts A/B and C/D used different workload designs, so their holdout numbers are not directly comparable. The change is part of the recorded experiment history.

## 5. Final benchmark result

```mermaid
flowchart LR
    BASELINE[Frozen baseline<br/><br/>Capability 2 of 8 = 25%<br/>Holdouts 0 of 4<br/>Regression 2 of 4<br/>Deployed experts 0]
    GROWN[After growth<br/><br/>Capability 5 of 8 = 62.5%<br/>Holdouts 3 of 4<br/>Regression 2 of 4<br/>Deployed experts 1]
    ROLLED[Rollback<br/><br/>Capability 2 of 8 = 25%<br/>Holdouts 0 of 4<br/>Regression 2 of 4<br/>Deployed experts 0]
    RESTORED[Restored<br/><br/>Holdout smoke 3 of 4<br/>Expert routes 4 of 4 holdouts<br/>Deployed experts 1]

    BASELINE -->|Add 2.64M parameters<br/>plus 37.5 capability points| GROWN
    GROWN -->|Manifest-only rollback| ROLLED
    ROLLED -->|Restore expert manifest| RESTORED

    classDef base fill:#1d3557,color:#fff,stroke:#0b1f33
    classDef gain fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef rollback fill:#7b2d26,color:#fff,stroke:#4d1713
    class BASELINE base
    class GROWN,RESTORED gain
    class ROLLED rollback
```

The regression result stayed at 2/4 because the base already failed the other two tasks. The supported claim is zero measured loss on known successes, not perfect general Python performance.

## 6. What the final expert learned and did not learn

```mermaid
flowchart TB
    EXPERT[Admitted escaped-path expert]

    EXPERT --> LEARNED[Learned reliably]
    EXPERT --> PARTIAL[Learned partially]
    EXPERT --> FUTURE[Not yet learned]

    LEARNED --> GET[Get and lookup variants]
    LEARNED --> EXISTS[Exists variants]
    LEARNED --> SET[Set and assign variants]
    LEARNED --> FLAT[Flatten and unflatten variants]
    LEARNED --> DELETE1[Canonical delete]

    PARTIAL --> DELETE2[path_delete_v2<br/>2 of 3 cases]
    PARTIAL --> DELETE3[path_delete_v3<br/>2 of 3 cases]
    PARTIAL --> MEMBER[Held-out membership<br/>failed]

    FUTURE --> RENAME[path_rename<br/>1 of 3 cases]
    FUTURE --> PROJECT[path_project<br/>0 of 2 cases]

    classDef root fill:#1d3557,color:#fff,stroke:#0b1f33
    classDef learned fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef partial fill:#b07d2b,color:#fff,stroke:#704c14
    classDef missing fill:#7b2d26,color:#fff,stroke:#4d1713
    class EXPERT root
    class LEARNED,GET,EXISTS,SET,FLAT,DELETE1 learned
    class PARTIAL,DELETE2,DELETE3,MEMBER partial
    class FUTURE,RENAME,PROJECT missing
```

The residual corrected failures form a specific deletion/pruning cluster. The future failures show that learning escaped parsing does not automatically produce new path algorithms.

## 7. Append-only deployment history

```mermaid
flowchart LR
    S1[Sequence 1<br/>Frozen baseline<br/>No experts]
    S2[Sequence 2<br/>Admission<br/>Final expert]
    S3[Sequence 3<br/>Rollback drill<br/>No experts]
    S4[Sequence 4<br/>Restore<br/>Final expert]
    S5[Sequence 5<br/>Corrected metric drill<br/>No experts]
    S6[Sequence 6<br/>Restore<br/>Final expert]
    S7[Sequence 7<br/>Pin exact base commit<br/>Final expert<br/>CURRENT]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7

    classDef off fill:#7b2d26,color:#fff,stroke:#4d1713
    classDef on fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef current fill:#1d3557,color:#fff,stroke:#0b1f33,stroke-width:4px
    class S1,S3,S5 off
    class S2,S4,S6 on
    class S7 current
```

Rollback appends a new manifest; it does not erase history or mutate adapter weights. Sequence 5 corrected a reporting defect where growth cost counted lifecycle-active rather than currently deployed experts.

## 8. Evidence and artifact lineage

```mermaid
flowchart TD
    TASK[Live task and prompt hash] --> ATTEMPT[Base response and verification]
    ATTEMPT --> FAILURE[Failure ID]
    FAILURE --> CORRECTION[Accepted correction ID]
    CORRECTION --> DATASET[Candidate JSONL<br/>Dataset SHA-256]
    DATASET --> JOB[MLX training job ID]
    JOB --> ADAPTER[Adapter directory<br/>Aggregate SHA-256]
    ADAPTER --> CANDIDATE[Candidate expert record]
    CANDIDATE --> PROBATION[Birth holdout and replay metrics]
    PROBATION --> DECISION{Admission decision}
    DECISION -->|Pass| ACTIVE[Active expert state]
    DECISION -->|Fail| REJECTED[Rejected expert state]
    ACTIVE --> MANIFEST[Append-only deployment manifest]
    MANIFEST --> REPORT[Longitudinal report and curve]

    LEDGER[(SQLite ledger)]
    ATTEMPT --> LEDGER
    FAILURE --> LEDGER
    CORRECTION --> LEDGER
    CANDIDATE --> LEDGER
    DECISION --> LEDGER
    MANIFEST --> LEDGER

    classDef evidence fill:#457b9d,color:#fff,stroke:#1d3557
    classDef decision fill:#b07d2b,color:#fff,stroke:#704c14
    classDef success fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef failure fill:#7b2d26,color:#fff,stroke:#4d1713
    class TASK,ATTEMPT,FAILURE,CORRECTION,DATASET,JOB,ADAPTER,CANDIDATE,PROBATION,REPORT,LEDGER evidence
    class DECISION decision
    class ACTIVE,MANIFEST success
    class REJECTED failure
```

This lineage makes every deployed expert traceable to the exact failures, corrections, dataset, training job, artifact, probation metrics, and deployment decision that created it.

## 9. Sealed-experiment history and outcomes

Everything after the first experiment ran under sealed, predeclared specs (`../experiments/`) graded by `scripts/check_experiment_spec.py`. A run either passes, fails a named rule, or is refused as unusable (exit 2) — a refusal binds the run to its declaration but grades nothing.

```mermaid
flowchart TB
    E1["2026-07-31 — First real growth cycle<br/>ADMITTED after three rejections<br/>held-out 0/4 → 3/4, rollback clean"]
    V1["2026-08-01 — Independent verification<br/>exact reproduction; fresh holdouts 0/6 → 5/6<br/>forced adapter breaks reg_dedupe"]
    E2["2026-08-08 — EXP-002 rerun (exit 1, usable)<br/>3 of 13 rules failed — H3 FALSIFIED<br/>forced regression 46/94, forgetting_claim router_shielded<br/>D9 also failed: five undeclared worker checkout gaps"]
    E3["2026-08-08 — EXP-003 (exit 2, unusable)<br/>self-repair 0/20 verified under 3 greedy attempts<br/>no primary expert, arms unpairable"]
    E4["2026-08-09 — EXP-004 (exit 2, unusable)<br/>self-repair 0/20 verified in 160 sampled seeded calls<br/>control arm 20/20, expert admitted; arms still unpairable"]
    E5["2026-08-11 — EXP-005 (exit 1, usable)<br/>H1+H2 FALSIFIED — second candidate competent<br/>but router claimed 18/112 base-passing tasks (FP 0.161)<br/>12 regressed routed; admission rejected it; pool stayed at one"]

    E1 --> V1 --> E2 --> E3 --> E4 --> E5

    classDef pos fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef mixed fill:#b07d2b,color:#fff,stroke:#704c14
    classDef neg fill:#7b2d26,color:#fff,stroke:#4d1713
    class E1 pos
    class V1 mixed
    class E2,E3,E4,E5 neg
```

The first EXP-002 attempt (2026-08-08 morning) is also on record: every LXD launch timed out, live capture graded 0/123, and the run measured nothing (`../research/2026-08-08-exp002-run.md`). The rerun the same day had healthy infrastructure and produced the falsification.

## 10. EXP-002: routed versus forced replay

The question H3 asked: does the adapter itself preserve prior competence, or does the router merely keep prior traffic away from it?

```mermaid
flowchart TB
    CAPTURE[123 live-capture tasks<br/>94 passed at baseline, 29 failed] --> COHORT[(94 prior-passing<br/>replay tasks<br/>spec floor 50)]

    COHORT --> ROUTED[Routed replay<br/>router decides adapter use]
    COHORT --> FORCED[Forced replay<br/>adapter always on]

    ROUTED --> RPASS[94 of 94 pass<br/>regression_rate 0.0<br/>route false positives 0 of 94]
    FORCED --> FFAIL[46 of 94 break<br/>forced_regression_rate 0.489<br/>lists strings dicts arithmetic]

    RPASS --> CLAIM[forgetting_claim resolves to<br/>router_shielded not adapter_intrinsic]
    FFAIL --> CLAIM
    CLAIM --> H3[H3 FALSIFIED<br/>stability belongs to the router<br/>not the adapter]

    classDef evidence fill:#1d3557,color:#fff,stroke:#0b1f33
    classDef pass fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef fail fill:#7b2d26,color:#fff,stroke:#4d1713
    classDef decision fill:#b07d2b,color:#fff,stroke:#704c14
    class CAPTURE,COHORT evidence
    class ROUTED,RPASS pass
    class FORCED,FFAIL fail
    class CLAIM,H3 decision
```

This is a scientific result, not a bug: the deployed system's zero regression is real, but it depends on router precision (measured 1.0 on this single-expert probe) holding as experts multiply. No threshold was loosened to accommodate the failure.

## 11. EXP-003 and EXP-004: the self-repair zero

Both experiments asked whether the model can supply its own verified corrections instead of human-written canonical ones — the decisive input for a self-improving loop. Both arms of each run captured the same 20 verified `escaped_path` training failures.

```mermaid
flowchart TB
    FAILS[(20 verified training failures<br/>identical set in every arm)]

    FAILS --> R3["EXP-003 regime<br/>3 greedy attempts per failure, temperature 0.0<br/>near-identical retries, generic feedback<br/>60 generation calls"]
    FAILS --> R4["EXP-004 fair regime<br/>8 sampled attempts per failure, temperature 0.8<br/>per-attempt recorded seeds, honest verifier feedback<br/>160 generation calls"]
    FAILS --> CTRL["Control arm<br/>canonical human-written corrections"]

    R3 --> Z3[0 of 20 verified]
    R4 --> Z4[0 of 20 verified]
    CTRL --> C20[20 of 20 verified]

    Z3 --> CHAIN[No verified corrections<br/>→ no candidate trained<br/>→ no pairing key]
    Z4 --> CHAIN
    CHAIN --> EXIT2["Checker refuses run as UNUSABLE exit 2<br/>the preregistered outcome for unpairable arms<br/>nothing confirmed, nothing formally falsified"]

    C20 --> ADMIT["EXP-004 control expert admitted<br/>held-out 0.75, plasticity 0.90<br/>routed regression 0.0, forced 46/94 again"]

    classDef evidence fill:#1d3557,color:#fff,stroke:#0b1f33
    classDef regime fill:#457b9d,color:#fff,stroke:#1d3557
    classDef fail fill:#7b2d26,color:#fff,stroke:#4d1713
    classDef pass fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef decision fill:#b07d2b,color:#fff,stroke:#704c14
    class FAILS evidence
    class R3,R4,CTRL regime
    class Z3,Z4,CHAIN,EXIT2 fail
    class C20,ADMIT pass
```

Zero successes in 160 roughly independent sampled attempts bounds the per-attempt repair probability at roughly 1.9% (95%, rule of three) — below the ~3.5% needed to reach the sealed 0.25 verified-rate bar under 8 attempts. The rejections were substantive, not infrastructural: exit-status failures, hidden-case failures, one malformed result. The measurement stands even though the A/B comparison could not be graded.

## 12. EXP-005: the second cycle and the router over-claim

The question: can a second expert grow on a new same-domain family (`path_restructure`) while the first stays deployed? The candidate was competent; the tag/keyword router was not precise enough to deploy it.

```mermaid
flowchart TB
    FAM[(20 path_restructure training failures<br/>canonical corrections 20/20<br/>4 fresh held-outs, content-hash disjoint)] --> CAND["Cycle-2 candidate expert_bb821fb6d00f<br/>target 0.85, held-out 0.75<br/>plasticity 0.85, route recall 1.0<br/>every competence gate met"]

    CAND --> POOL{Two-expert routing probe<br/>expert 1 deployed}
    POOL --> FP["Router claims 18/112 base-passing<br/>replay tasks for the new expert<br/>FP rate 0.161 vs budget 0.0<br/>precision 0.143"]
    FP --> REG[12/112 regress under routing<br/>0.107 vs budget 0.0]
    REG --> REJ["Admission policy rejects candidate<br/>unchanged gates, working as sealed<br/>pool ends with ONE expert"]

    REJ --> SAFE["Expert 1 untouched: held-out 0.75 routed and forced<br/>routed replay 112/112, capability delta 0.0"]
    REJ --> H12["H1+H2 FALSIFIED (exit 1, 23/29 rules passed)<br/>failed component: tag/keyword router's<br/>same-domain discrimination — not the candidate,<br/>not the training, not the gates"]

    classDef evidence fill:#1d3557,color:#fff,stroke:#0b1f33
    classDef pass fill:#2d6a4f,color:#fff,stroke:#183b2c
    classDef fail fill:#7b2d26,color:#fff,stroke:#4d1713
    classDef decision fill:#b07d2b,color:#fff,stroke:#704c14
    class FAM evidence
    class CAND,SAFE pass
    class FP,REG,REJ fail
    class POOL,H12 decision
```

The safety property held (nothing regressed in deployment; the bad pairing was refused); the growth property did not (the pool has still never held two experts). The transfer diagnostic was also negative: `path_project` 0.0, `path_rename` 0.33, both misrouted to expert 1. Full record: `../research/2026-08-11-exp005-second-cycle.md`.

## 13. Open questions after EXP-005

Ranked by how much each threatens the core claim; details in `../PLAN.md` and the README.

```mermaid
flowchart TD
    NOW["Current state 2026-08-11<br/>One expert family admitted, reproduced across four cycles<br/>Stability shown to be router-shielded, not adapter-intrinsic<br/>Self-repair: 0 verified corrections in 220 attempts<br/>Two-expert coexistence falsified: router over-claims, FP 0.161"]

    NOW --> Q1["1 — Router discrimination<br/>can any router at this representational budget<br/>meet the sealed FP 0.0 / precision bar<br/>on same-domain families?"]
    NOW --> Q2["2 — Self-generated corrections<br/>stronger proposer, or near-miss failure families,<br/>or accept that yield is floored at this model scale"]
    NOW --> Q3["3 — Router shielding under many experts<br/>blocked behind Q1: no pool has ever held two"]
    NOW --> Q4["4 — Transfer beyond paraphrases<br/>path_rename 0.33, path_project 0.0 in the EXP-005 probe<br/>both misrouted to expert 1"]
    NOW --> Q5["5 — Remaining research controls<br/>seeds, per-cycle fresh holdouts, full fine-tune baseline"]
    NOW --> Q6["6 — Non-deterministic verifiers<br/>judges and user feedback untested"]
    NOW --> Q7["7 — Persistent serving<br/>resident base with hot-swapped multi-LoRA adapters"]

    Q2 --> BLOCK["External trust anchors Grove lacks:<br/>worker-side signed attestation, verifiable timestamps,<br/>signed or transparency-logged manifests"]
    Q5 --> BLOCK

    classDef now fill:#1d3557,color:#fff,stroke:#0b1f33
    classDef work fill:#457b9d,color:#fff,stroke:#1d3557
    classDef blocked fill:#b07d2b,color:#fff,stroke:#704c14
    class NOW now
    class Q1,Q2,Q3,Q4,Q5,Q6,Q7 work
    class BLOCK blocked
```

The 50-task replay floor that an earlier version of this diagram listed as future work has been met and measured (94 prior-passing tasks): routed replay is stable and forced replay is falsified. Replay stability is now a result, not a gap.
