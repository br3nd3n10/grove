import pytest

from grove.mlx_backend import MlxRemoteBackend, extract_python
from grove.models import Task


def test_extract_python_stops_at_chat_end_token() -> None:
    generated = "def solve(payload):\n    return payload['value']\n<|im_end|>\nnot python"

    assert extract_python(generated) == (
        "def solve(payload):\n    return payload['value']"
    )


def test_extract_python_handles_fence_before_end_token() -> None:
    generated = (
        "```python\ndef solve(payload):\n    return 1\n```"
        "<|im_end|>hallucinated continuation"
    )

    assert extract_python(generated) == "def solve(payload):\n    return 1"


def test_extract_python_stops_at_end_of_text_token() -> None:
    generated = "def solve(payload):\n    return 2<|endoftext|>garbage"

    assert extract_python(generated) == "def solve(payload):\n    return 2"


# --------------------------------------------------------------------------
# EXP-004: per-purpose decoding on the control-host side of the wire
# --------------------------------------------------------------------------


class EchoWorker:
    """Records the job spec and echoes each request's decoding, like the worker."""

    def __init__(self, *, echo_temperature=None, echo_seed=None, honest=True):
        self.specs: list[dict] = []
        self.echo_temperature = echo_temperature
        self.echo_seed = echo_seed
        self.honest = honest

    def run_job(self, command: str, spec: dict) -> dict:
        self.specs.append(spec)
        outputs = []
        for request in spec["requests"]:
            item = {"id": request["id"], "text": "def solve(payload):\n    return 1"}
            if self.honest:
                item["temperature"] = request.get("temperature", 0.0)
                item["seed"] = request.get("seed")
            else:
                item["temperature"] = self.echo_temperature
                item["seed"] = self.echo_seed
            outputs.append(item)
        return {"status": "ok", "outputs": outputs}


def _task(task_id: str = "t1") -> Task:
    return Task(task_id, "prompt", verifier="sandboxed_python")


def test_backend_default_decoding_stays_greedy_with_no_per_request_fields():
    """Evaluation calls must be byte-identical to the pre-EXP-004 protocol."""
    worker = EchoWorker()
    backend = MlxRemoteBackend(worker, model="m")

    backend.generate(_task())

    spec = worker.specs[0]
    assert spec["temperature"] == 0.0
    assert "temperature" not in spec["requests"][0]
    assert "seed" not in spec["requests"][0]


def test_backend_sends_sampled_decoding_per_request_and_accepts_the_echo():
    worker = EchoWorker()
    backend = MlxRemoteBackend(worker, model="m")

    text = backend.generate(
        _task(), decoding={"temperature": 0.8, "seed": 99, "max_tokens": 128}
    )

    request = worker.specs[0]["requests"][0]
    assert request["temperature"] == 0.8
    assert request["seed"] == 99
    assert worker.specs[0]["max_tokens"] == 128
    assert text == "def solve(payload):\n    return 1"


def test_backend_rejects_a_worker_that_ignores_the_requested_decoding():
    """An ignored override silently reruns EXP-003's broken greedy regime."""
    worker = EchoWorker(honest=False, echo_temperature=0.0, echo_seed=None)
    backend = MlxRemoteBackend(worker, model="m")

    with pytest.raises(RuntimeError, match="did not honor"):
        backend.generate(_task(), decoding={"temperature": 0.8, "seed": 99})
