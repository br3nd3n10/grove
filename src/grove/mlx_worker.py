from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from grove.provenance import git_runner, worktree_digest

WORKER_ROOT = Path.home() / "grove"


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _inside_worker(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = WORKER_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"worker path must remain under {root}: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    files = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = str(path.relative_to(directory))
        file_hash = _sha256_file(path)
        size = path.stat().st_size
        aggregate.update(relative.encode())
        aggregate.update(file_hash.encode())
        files.append({"path": relative, "sha256": file_hash, "size_bytes": size})
    parameter_count = 0
    adapter_weights = directory / "adapters.safetensors"
    if adapter_weights.exists():
        import mlx.core as mx

        parameter_count = sum(
            array.size for array in mx.load(str(adapter_weights)).values()
        )
    return {
        "path": str(directory),
        "sha256": aggregate.hexdigest(),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "parameter_count": parameter_count,
        "files": files,
    }


WORKER_REPOSITORY = WORKER_ROOT / "runtime" / "repo"


def _worker_checkout(root: str | Path | None = None) -> dict[str, Any]:
    """Which code this worker is running, reported by the worker itself.

    The worker command is launched after the caller changes into its configured
    ``remote_repository``. Using the process working directory by default keeps
    provenance tied to the checkout that actually ran the job, rather than to a
    hard-coded path that may differ from the job's repository. Tests and direct
    callers can provide an explicit root.

    The control host cannot see the worker's checkout, so without this a paired
    experiment cannot tell whether both arms were trained by the same worker
    code. This is a self-report: it is evidence of what the worker claims, not
    proof, and the control host records it as such.

    The content digest comes from ``grove.provenance.worktree_digest``, the same
    function the control host uses, so the two sides are comparable. The worker
    previously hashed only ``git diff HEAD``, which left an untracked or binary
    change invisible.
    """
    checkout_root = Path.cwd() if root is None else Path(root)
    git = git_runner(checkout_root)
    revision = git("rev-parse", "HEAD")
    if revision is None:
        return {}
    status = git("status", "--porcelain")
    record: dict[str, Any] = {
        "revision": revision,
        "tree": git("rev-parse", "HEAD^{tree}"),
    }
    if status is not None:
        record["dirty"] = bool(status)
        record["status_sha256"] = hashlib.sha256(status.encode()).hexdigest()
    record["worktree_sha256"] = worktree_digest(git, checkout_root)
    return record


MODEL_MANIFEST_SCHEMA = "grove-worker-model-manifest-v1"


def model_manifest_digest(model_path: str | Path | None) -> dict[str, Any]:
    """Digest over the base-model files this worker actually holds.

    Framework versions and a checkout revision say nothing about which model
    weights were loaded. Without this a paired experiment cannot tell two arms
    apart when the worker swapped the snapshot underneath them.

    An absent or empty directory is a named gap, never a digest over nothing:
    hashing an empty file set produces a perfectly stable value that means
    "no evidence".
    """
    if model_path is None:
        return {"path": None, "sha256": None, "reason": "no model path requested"}
    root = Path(model_path)
    if not root.is_dir():
        return {"path": str(root), "sha256": None, "reason": "not a directory"}
    files = []
    aggregate = hashlib.sha256()
    aggregate.update(MODEL_MANIFEST_SCHEMA.encode())
    try:
        members = sorted(entry for entry in root.rglob("*") if entry.is_file())
        for item in members:
            relative = str(item.relative_to(root))
            digest = _sha256_file(item)
            aggregate.update(relative.encode())
            aggregate.update(b"\0")
            aggregate.update(digest.encode())
            files.append({"path": relative, "sha256": digest})
    except OSError as error:
        # A partial digest would look like evidence. Refuse instead.
        return {
            "path": str(root),
            "sha256": None,
            "reason": f"{type(error).__name__} reading model files",
        }
    if not files:
        return {"path": str(root), "sha256": None, "reason": "no model files found"}
    return {
        "path": str(root),
        "sha256": aggregate.hexdigest(),
        "file_count": len(files),
    }


def preflight(
    model_path: str | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    import mlx.core as mx

    checkout_root = Path.cwd() if repository is None else Path(repository)
    manifest = model_manifest_digest(model_path)
    return {
        "status": "ok",
        "hostname": platform.node(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "default_device": str(mx.default_device()),
        "worker_root": str(WORKER_ROOT),
        "checkout": _worker_checkout(checkout_root),
        "model_manifest": manifest,
        "model_manifest_sha256": manifest["sha256"],
        "free_bytes": os.statvfs(WORKER_ROOT).f_bavail
        * os.statvfs(WORKER_ROOT).f_frsize,
    }


def infer(spec: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model_id = str(spec["model"])
    adapter = spec.get("adapter_path")
    adapter_path = str(_inside_worker(adapter)) if adapter else None
    started = time.monotonic()
    model, tokenizer = load(model_id, adapter_path=adapter_path)
    # Qwen chat templates finish assistant turns with <|im_end|>.  Some model
    # metadata exposes only <|endoftext|> to MLX-LM's stop set, which otherwise
    # lets generation continue into a hallucinated next turn until max_tokens.
    for token in spec.get("stop_tokens", ["<|im_end|>"]):
        tokenizer.add_eos_token(str(token))
    default_temperature = float(spec.get("temperature", 0.0))
    samplers: dict[float, Any] = {}

    def sampler_for(temperature: float) -> Any:
        if temperature not in samplers:
            samplers[temperature] = make_sampler(temp=temperature)
        return samplers[temperature]

    outputs = []
    for item in spec["requests"]:
        # Per-request decoding, so one job can mix greedy evaluation with
        # sampled repair attempts. The values are echoed back per output:
        # the caller must be able to verify what decoding actually ran.
        temperature = float(item.get("temperature", default_temperature))
        seed = item.get("seed")
        if seed is not None:
            # ``make_sampler`` draws through ``mx.random.categorical`` from the
            # global MLX PRNG, so seeding that PRNG per request is what makes a
            # sampled attempt reproducible. Refuse rather than silently run
            # unseeded if the runtime cannot do it.
            if not hasattr(mx.random, "seed"):
                raise RuntimeError(
                    "request asked for a sampling seed but mx.random.seed is "
                    "unavailable in this MLX runtime"
                )
            seed = int(seed)
            mx.random.seed(seed)
        messages = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=int(item.get("max_tokens", spec.get("max_tokens", 512))),
            sampler=sampler_for(temperature),
            verbose=False,
        )
        outputs.append(
            {
                "id": item["id"],
                "text": text,
                "temperature": temperature,
                "seed": seed,
            }
        )
    return {
        "status": "ok",
        "model": model_id,
        "adapter_path": adapter_path,
        "outputs": outputs,
        "duration_seconds": time.monotonic() - started,
    }


def train(spec: dict[str, Any]) -> dict[str, Any]:
    data_path = _inside_worker(spec["data_path"])
    adapter_path = _inside_worker(spec["adapter_path"])
    if adapter_path.exists() and any(adapter_path.iterdir()):
        raise ValueError(f"candidate adapter directory is not empty: {adapter_path}")
    adapter_path.mkdir(parents=True, exist_ok=True)
    log_path = adapter_path / "training.log"
    command = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        str(spec["model"]),
        "--train",
        "--data",
        str(data_path),
        "--adapter-path",
        str(adapter_path),
        "--fine-tune-type",
        "lora",
        "--batch-size",
        str(int(spec.get("batch_size", 1))),
        "--iters",
        str(int(spec.get("iterations", 100))),
        "--num-layers",
        str(int(spec.get("num_layers", 8))),
        "--max-seq-length",
        str(int(spec.get("max_seq_length", 1024))),
        "--learning-rate",
        str(float(spec.get("learning_rate", 1e-5))),
        "--seed",
        str(int(spec.get("seed", 0))),
        "--mask-prompt",
        "--grad-checkpoint",
    ]
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=float(spec.get("timeout_seconds", 7200)),
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"MLX training failed with status {process.returncode}; see {log_path}"
        )
    return {
        "status": "ok",
        "model": spec["model"],
        "duration_seconds": time.monotonic() - started,
        "training_config": spec,
        "artifact": _artifact_manifest(adapter_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constrained Grove MLX worker")
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight_command = subcommands.add_parser("preflight")
    preflight_command.add_argument(
        "--model",
        help=(
            "base-model directory to hash into a model manifest; without it "
            "the manifest is reported as an explicit gap"
        ),
    )
    preflight_command.add_argument(
        "--repository",
        help=(
            "worker checkout to hash; defaults to the process working directory, "
            "which is the remote repository used by the job"
        ),
    )
    for command in ("infer", "train"):
        item = subcommands.add_parser(command)
        item.add_argument("--spec", required=True)
        item.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        _json(preflight(args.model, args.repository))
        return 0
    spec_path = _inside_worker(args.spec)
    output_path = _inside_worker(args.output)
    spec = json.loads(spec_path.read_text())
    try:
        result = infer(spec) if args.command == "infer" else train(spec)
    except Exception as error:
        result = {
            "status": "error",
            "type": type(error).__name__,
            "message": str(error),
        }
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        raise
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    _json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
