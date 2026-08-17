from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from grove.models import Expert, Task
from grove.remote import MlxSshWorker

PYTHON_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You write small, dependency-free Python functions for a verification harness.
Return only Python source defining solve(payload).
payload is already a Python value decoded from JSON, usually a dict or list; do not call json.loads on it.
Do not read stdin, print, use the network, access files, spawn processes, or include markdown.
The function's return value must be JSON serializable.
"""


def extract_python(text: str) -> str:
    # Some MLX/tokenizer combinations do not stop generation when the chat
    # template emits its textual end-of-turn sentinel.  Anything after that
    # sentinel is outside the assistant turn and must never reach a verifier.
    for sentinel in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(sentinel, 1)[0]
    text = text.strip()
    match = PYTHON_FENCE.search(text)
    if match:
        return match.group(1).strip()
    if text.lower().startswith("python\n"):
        return text.split("\n", 1)[1].strip()
    return text


class MlxRemoteBackend:
    def __init__(
        self,
        worker: MlxSshWorker | None = None,
        *,
        model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        max_tokens: int = 768,
    ) -> None:
        self.worker = worker or MlxSshWorker()
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        task: Task,
        expert: Expert | None = None,
        *,
        decoding: Mapping[str, object] | None = None,
    ) -> str:
        return self.generate_batch([task], expert, decoding=decoding)[0]

    def generate_batch(
        self,
        tasks: Sequence[Task],
        expert: Expert | None = None,
        *,
        decoding: Mapping[str, object] | None = None,
    ) -> list[str]:
        """Generate one answer per task.

        ``decoding`` overrides the per-purpose decoding for these requests only:
        ``temperature`` (float), ``seed`` (int) and ``max_tokens`` (int). The
        default is, and must remain, greedy temperature 0.0 -- evaluation,
        baseline, held-out and replay calls all take this path. Only a caller
        that explicitly asks (repair-attempt generation) samples, and the worker
        must echo the temperature and seed it ran so the override is verified,
        not assumed.
        """
        adapter_path = expert.artifact.get("adapter_path") if expert else None
        overrides = dict(decoding or {})
        temperature = float(overrides.get("temperature", 0.0))
        seed = overrides.get("seed")
        seed = int(seed) if seed is not None else None
        max_tokens = int(overrides.get("max_tokens", self.max_tokens))
        requests = []
        for task in tasks:
            request: dict[str, object] = {
                "id": task.id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task.prompt},
                ],
            }
            if decoding is not None:
                request["temperature"] = temperature
                if seed is not None:
                    request["seed"] = seed
            requests.append(request)
        result = self.worker.run_job(
            "infer",
            {
                "model": self.model,
                "adapter_path": adapter_path,
                "temperature": 0.0 if decoding is None else temperature,
                "max_tokens": max_tokens,
                "stop_tokens": ["<|im_end|>"],
                "requests": requests,
            },
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"MLX inference failed: {result}")
        outputs = {item["id"]: item for item in result["outputs"]}
        if decoding is not None:
            # The worker echoes what it actually ran. A worker that ignored the
            # override would silently turn a sampled, seeded repair regime back
            # into EXP-003's three identical greedy attempts.
            for task in tasks:
                echoed = outputs[task.id]
                if echoed.get("temperature") != temperature or echoed.get(
                    "seed"
                ) != seed:
                    raise RuntimeError(
                        "MLX worker did not honor the requested decoding for "
                        f"{task.id}: asked temperature={temperature} seed={seed}, "
                        f"echoed temperature={echoed.get('temperature')} "
                        f"seed={echoed.get('seed')}"
                    )
        return [extract_python(outputs[task.id]["text"]) for task in tasks]
