# grove

**A continually-learning MoE system with verified expert admission.**

Today's LLMs are frozen the day training ends. grove is an attempt to build the thing the 2026 continual-learning literature has only built in pieces: a deployed agent that captures its own verifier-checked failures, periodically grows new *removable* experts to fix them (base weights stay frozen), gates every addition through demand and probation checks, and demonstrably improves month over month without regressing.

Two intended artifacts:

1. **The system** — the assembled grow-loop, running on real production agent workloads, not toy benchmarks.
2. **The benchmark** — a longitudinal evaluation harness tracking the three curves that matter: plasticity up, regression flat, growth sublinear. No standard benchmark for this exists yet.

See [PLAN.md](PLAN.md) for the full research plan: background, prior art (FLEX-MoE, GoD-MoE, CP-MoE, BAR), system architecture diagrams, the three-stage roadmap, budget, and risks.

## Status

**Stage 0 — planning.** Currently working through foundations (Stage 1 of the plan): building intuition from scratch via nanoGPT, toy MoE routing, and first LoRA fine-tunes.

## Why this might work

The hard part of this project is not exotic ML — it's orchestration: verification harnesses, automated eval gates, scheduled consolidation jobs, monitoring. The academic groups working on the components are strong at the ML half and haven't built the systems half. I'm approaching from the opposite side.
