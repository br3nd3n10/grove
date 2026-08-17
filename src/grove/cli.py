from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from grove.benchmark import LongitudinalBenchmark
from grove.demo import (
    DemoMathBackend,
    DemoMathTrainer,
    demo_benchmark_cohorts,
    demo_live_tasks,
)
from grove.runtime import GroveRuntime
from grove.sleep import SleepCycle
from grove.store import GroveStore


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _components(database: str):
    store = GroveStore(database)
    runtime = GroveRuntime(store, DemoMathBackend())
    sleep = SleepCycle(store, runtime, DemoMathTrainer())
    benchmark = LongitudinalBenchmark(store, runtime)
    return store, runtime, sleep, benchmark


def run_demo(database: str, reset: bool = False) -> dict[str, Any]:
    path = Path(database)
    if reset and path.exists():
        path.unlink()
    store, runtime, sleep, benchmark = _components(database)
    try:
        baseline = benchmark.evaluate(demo_benchmark_cohorts(), label="baseline")
        live = runtime.run(demo_live_tasks(), run_id="demo_live_capture")
        cycle = sleep.run()
        grown = benchmark.evaluate(demo_benchmark_cohorts(), label="after_growth")
        return {
            "baseline": baseline,
            "live_capture": {
                "tasks": len(live),
                "passed": sum(result.verification.passed for result in live),
                "failures": sum(not result.verification.passed for result in live),
            },
            "sleep_cycle": asdict(cycle),
            "after_growth": grown,
            "curve": benchmark.curve(),
            "store": store.summary(),
        }
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grove", description="Verified expert-growth control plane"
    )
    parser.add_argument("--db", default=".grove/grove.db", help="SQLite evidence store")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="run the complete local grow-loop")
    demo.add_argument(
        "--reset", action="store_true", help="replace the selected demo database"
    )
    subcommands.add_parser("status", help="show operational state")
    subcommands.add_parser("ledger", help="show the append-only expert audit ledger")
    subcommands.add_parser("curve", help="show longitudinal benchmark checkpoints")
    subcommands.add_parser("worker-preflight", help="check the remote Apple MLX worker")
    remove = subcommands.add_parser("remove", help="unplug an active expert")
    remove.add_argument("expert_id")
    remove.add_argument("--reason", required=True)
    real = subcommands.add_parser(
        "real-cycle", help="run the first real MLX learning cycle"
    )
    real.add_argument("--reset", action="store_true")
    real.add_argument(
        "--report",
        default="/srv/storage/grove/evaluations/first-real-cycle.json",
    )
    real.add_argument(
        "--correction-source",
        choices=("canonical", "self-repair"),
        default="canonical",
        help="where verified training targets come from",
    )
    real.add_argument(
        "--compare-corrections",
        action="store_true",
        help="also record both correction sources' verified yield",
    )
    real.add_argument(
        "--self-repair-attempts",
        type=int,
        default=None,
        help=(
            "repair attempts per failure for the self-repair source; omitted, "
            "the sealed spec's required_setup for the selected arm decides "
            "(legacy default 3)"
        ),
    )
    real.add_argument(
        "--cycles",
        type=int,
        default=None,
        help=(
            "growth cycles to run (1 or 2). Two runs the complete first cycle, "
            "then a second capture-train-gate cycle against the same store with "
            "the first expert still deployed, plus multi-expert coexistence "
            "measurements. Omitted, the sealed spec's required_setup for the "
            "selected arm decides (legacy default 1)."
        ),
    )
    real.add_argument(
        "--spec",
        help=(
            "sealed predeclared experiment spec to bind this run to; the seal is "
            "verified before any model work and the digest goes into the report"
        ),
    )
    real.add_argument(
        "--arm",
        choices=("primary", "control"),
        help=(
            "which arm's required_setup profile this run must satisfy. A paired "
            "spec declares one per arm; without this the control arm is graded "
            "against the primary declaration and cannot launch. Omitted, the arm "
            "is inferred from the declared correction source."
        ),
    )
    provenance = subcommands.add_parser(
        "provenance", help="show reproducibility metadata for this checkout"
    )
    provenance.add_argument(
        "--worker",
        action="store_true",
        help=(
            "also SSH to the MLX worker for its identity and framework versions; "
            "without this the worker section is reported as an explicit gap"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        _print(run_demo(args.db, reset=args.reset))
        return 0
    if args.command == "real-cycle":
        from grove.experiment import ExperimentSetupError, run_first_real_cycle

        try:
            report = run_first_real_cycle(
                args.db,
                args.report,
                reset=args.reset,
                correction_source=args.correction_source,
                self_repair_attempts=args.self_repair_attempts,
                compare_corrections=args.compare_corrections,
                spec_path=args.spec,
                arm=args.arm,
                growth_cycles=args.cycles,
            )
        except ExperimentSetupError as error:
            # A refused setup is not a crash and not an experimental result.
            # A traceback made a deliberate, correct refusal look like a bug,
            # and exit 1 is the code for a falsified prediction. Nothing ran
            # here, so this is the same exit 2 the checker uses for "cannot be
            # judged". No database and no report were created.
            print(
                json.dumps(
                    {"status": "setup_refused", "error": str(error)},
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        _print(report)
        return 0
    if args.command == "provenance":
        from grove.experiment import checkout_provenance

        _print(checkout_provenance(collect_worker=args.worker))
        return 0
    if args.command == "worker-preflight":
        from grove.remote import MlxSshWorker

        _print(MlxSshWorker().preflight())
        return 0
    store, runtime, sleep, benchmark = _components(args.db)
    try:
        if args.command == "status":
            _print(store.summary())
        elif args.command == "ledger":
            _print(store.ledger())
        elif args.command == "curve":
            _print(benchmark.curve())
        elif args.command == "remove":
            current = store.current_deployment()
            if current is not None:
                # Removal is model-agnostic, but its replacement manifest must
                # retain the exact deployed base/router/verifier/decoding pins.
                sleep = SleepCycle(
                    store,
                    runtime,
                    DemoMathTrainer(),
                    base_model_revision=current.base_model_revision,
                    router_version=current.router_version,
                    verifier_suite_version=current.verifier_suite_version,
                    decoding_config=current.decoding_config,
                )
            _print(asdict(sleep.remove_expert(args.expert_id, args.reason)))
        return 0
    finally:
        store.close()
