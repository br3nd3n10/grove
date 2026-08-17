"""Independent verification probe for the Grove experiment.

Runs entirely read-only against the original evidence: no store writes,
fresh verifier instances, results written to /tmp/grove-probe/results.json.

Probes:
  A. Reproduce headline: base vs adapter on the 4 original holdouts.
  B. Fresh holdouts: 6 brand-new paraphrases with brand-new hidden cases
     (never seen by training or the admission gate). Base vs adapter.
  C. Forced-adapter forgetting: regression tasks with the adapter FORCED on
     (bypassing the router), which the original experiment never measured.
  D. Future stream: base vs adapter on path_rename / path_project.
  E. Routing: which expert the router picks for an oracle-free copy of each
     task, and for the original task with its gold family tags left in. The
     gold-tag number is a diagnostic only. The routing profile is built out of
     those very tags, so scoring the router on them measures label equality,
     not routing, and it is never independent routing evidence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from grove.coding_tasks import coding_catalog
from grove.experiment import BASE_MODEL_SOURCE
from grove.mlx_backend import MlxRemoteBackend
from grove.models import Expert, ExpertStatus, Task
from grove.routing import ProfileRouter
from grove.sandbox import LxdSandbox
from grove.sleep import oracle_free
from grove.verifiers import PythonCase, PythonSuite, SandboxedPythonVerifier

OUT = Path("/tmp/grove-probe/results.json")

catalog = coding_catalog()
by_id = {item.task.id: item for item in catalog}

ADAPTER_EXPERT = Expert(
    id="expert_979511319695",
    name="escaped_path-lora",
    status=ExpertStatus.ACTIVE,
    artifact={"adapter_path": "/Users/grove-worker/grove/adapters/expert_979511319695"},
    routing_profile={},
    born_from=(),
)


def reference_fn(task_id: str):
    namespace: dict = {}
    exec(by_id[task_id].reference_solution, namespace)  # noqa: S102 — trusted repo code
    return namespace["solve"]


def fresh_task(new_id, base_task_id, prompt, payloads):
    solve = reference_fn(base_task_id)
    cases = []
    for payload in payloads:
        expected = solve(copy.deepcopy(payload))
        cases.append(PythonCase(payload, expected))
    task = Task(
        id=new_id,
        prompt=prompt,
        verifier="sandboxed_python",
        cohort="probe_fresh",
        tags=("python", "escaped_path"),
    )
    suite = PythonSuite(new_id, tuple(cases), version="probe-fresh-v1")
    return task, suite


FRESH = [
    fresh_task(
        "fresh_lookup",
        "path_get",
        "Write solve(payload). payload carries keys document (nested dicts), path (a string), and sometimes default. "
        "The path is period-delimited, except a period immediately after a backslash is part of the key name, "
        "and a doubled backslash means one literal backslash. Walk the document along the path and return the value; "
        "if any step is absent or not a dictionary, return the default field (or None when absent).",
        [
            {"document": {"srv": {"log.dir": "/var/log"}}, "path": r"srv.log\.dir"},
            {"document": {"a\\b": {"c": 7}}, "path": r"a\\b.c"},
            {"document": {"root": {"leaf": 1}}, "path": "root.leaf.deeper", "default": "missing"},
            {"document": {"": {"": 5}}, "path": "."},
        ],
    ),
    fresh_task(
        "fresh_member",
        "path_exists",
        "Implement solve(payload) that answers true or false: does payload['document'] contain the nested key "
        "sequence written in payload['path']? Periods split the sequence unless preceded by a backslash, which "
        "quotes the next character. A key whose stored value is null still exists.",
        [
            {"document": {"cfg": {"timeout.ms": None}}, "path": r"cfg.timeout\.ms"},
            {"document": {"cfg": {}}, "path": "cfg.retries"},
            {"document": {"x\\y": 3}, "path": r"x\\y"},
        ],
    ),
    fresh_task(
        "fresh_assign",
        "path_set",
        "Create solve(payload) that returns a modified deep copy of payload['document']: store payload['value'] at "
        "the location payload['path'] describes. Path components are separated by unescaped periods; a backslash "
        "makes the following character literal. Intermediate dictionaries must be created when missing, and any "
        "non-dictionary encountered along the way is replaced by a dictionary. Never mutate the input.",
        [
            {"document": {}, "path": r"app.window\.title", "value": "Grove"},
            {"document": {"a": {"b": 1}}, "path": "a.b.c", "value": 2},
            {"document": {"k": {"old": 1}}, "path": "k.new", "value": [1, 2]},
        ],
    ),
    fresh_task(
        "fresh_remove",
        "path_delete",
        "Define solve(payload): produce a deep copy of payload['document'] with the entry named by "
        "payload['path'] removed. Periods separate path components unless backslash-escaped. After removing the "
        "entry, delete any ancestor dictionaries that are now empty. When the path does not resolve, hand back "
        "an unchanged copy.",
        [
            {"document": {"a": {"b": {"c": 1}}, "keep": 2}, "path": "a.b.c"},
            {"document": {"x.y": 1, "z": 2}, "path": r"x\.y"},
            {"document": {"a": {"b": 1}}, "path": "a.missing"},
        ],
    ),
    fresh_task(
        "fresh_flatten",
        "path_flatten",
        "Write solve(payload) that turns the nested dictionary payload into a single-level dictionary whose keys "
        "are period-joined paths. When an original key itself contains a period or backslash, prefix that "
        "character with a backslash in the output key. A dictionary with no entries stays in the result as a "
        "leaf value.",
        [
            {"web": {"host.name": "a", "port": 80}},
            {"a\\b": {"c": {}}},
            {"top": 1},
        ],
    ),
    fresh_task(
        "fresh_expand",
        "path_unflatten",
        "Implement solve(payload). payload maps escaped period-delimited path strings to values. Rebuild and "
        "return the nested dictionary those paths describe: unescaped periods introduce a new nesting level, and "
        "a backslash quotes the character after it (so backslash-period is a literal period inside a key). "
        "Create every intermediate dictionary.",
        [
            {r"db.conn\.str": "x", "db.pool": 4},
            {r"one\\two.three": 9},
            {"a.b": 1, "a.c": 2},
        ],
    ),
]

repro_tasks = [by_id[i].task for i in (
    "holdout_path_lookup", "holdout_path_membership", "holdout_path_assign", "holdout_path_flatten")]
regression_tasks = [by_id[i].task for i in (
    "reg_sum_even", "reg_dedupe", "reg_run_lengths", "reg_balanced")]
future_tasks = [by_id[i].task for i in ("path_rename", "path_project")]
fresh_tasks = [t for t, _ in FRESH]

all_tasks = repro_tasks + fresh_tasks + regression_tasks + future_tasks


def routing_probe(tasks: list[Task]) -> dict[str, dict[str, object]]:
    """Route each task twice: oracle-free, and with its gold tags left in.

    Only the oracle-free column is routing evidence. The gold column is kept
    because the gap between the two is the size of the leak.
    """
    router = ProfileRouter()
    pool = [ADAPTER_EXPERT]
    return {
        task.id: {
            "oracle_free_expert": router.route(oracle_free(task), pool).expert_id,
            "oracle_free_score": router.route(oracle_free(task), pool).score,
            "gold_tag_expert": router.route(task, pool).expert_id,
            "gold_tag_score": router.route(task, pool).score,
            "gold_tag_routing_is_independent_evidence": False,
        }
        for task in tasks
    }


def main() -> None:
    verifier = SandboxedPythonVerifier(
        LxdSandbox(),
        [item.suite for item in catalog] + [s for _, s in FRESH],
    )
    backend = MlxRemoteBackend(model=BASE_MODEL_SOURCE, max_tokens=768)

    print(f"[probe] generating {len(all_tasks)} responses with BASE model...", flush=True)
    base_responses = backend.generate_batch(all_tasks, None)
    print("[probe] base generation done", flush=True)
    print(f"[probe] generating {len(all_tasks)} responses with ADAPTER...", flush=True)
    adapter_responses = backend.generate_batch(all_tasks, ADAPTER_EXPERT)
    print("[probe] adapter generation done", flush=True)

    results: dict[str, dict[str, object]] = {}
    for label, responses in (("base", base_responses), ("adapter", adapter_responses)):
        for task, response in zip(all_tasks, responses, strict=True):
            print(f"[probe] verifying {label}:{task.id}", flush=True)
            verification = verifier.verify(task, response)
            results.setdefault(task.id, {})[label] = {
                "passed": verification.passed,
                "score": verification.score,
                "reason": verification.reason,
                "response": response,
            }

    routing = routing_probe(all_tasks)
    for task_id, decision in routing.items():
        results[task_id]["routing"] = decision

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print("[probe] wrote", OUT, flush=True)

    for task in all_tasks:
        record = results[task.id]
        base = record["base"]
        adapter = record["adapter"]
        print(
            f"{task.id:28s} base={'PASS' if base['passed'] else 'fail'}"
            f"({base['score']:.2f})  adapter={'PASS' if adapter['passed'] else 'fail'}"
            f"({adapter['score']:.2f})"
        )


if __name__ == "__main__":
    main()
