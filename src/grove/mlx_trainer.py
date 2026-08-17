from __future__ import annotations

import hashlib
import json
from pathlib import Path

from grove.mlx_backend import SYSTEM_PROMPT
from grove.models import Cluster, Expert, ExpertStatus
from grove.remote import MlxSshWorker


class MlxLoraTrainer:
    """Materializes verified corrections and trains one isolated MLX QLoRA adapter."""

    def __init__(
        self,
        worker: MlxSshWorker | None = None,
        *,
        model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        base_model_revision: str | None = None,
        artifact_root: str | Path = "/srv/storage/grove/datasets",
        iterations: int = 60,
        learning_rate: float = 2e-5,
        num_layers: int = 8,
        seed: int = 17,
    ) -> None:
        self.worker = worker or MlxSshWorker()
        self.model = model
        self.base_model_revision = base_model_revision or model
        self.artifact_root = Path(artifact_root)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.seed = seed

    def training_config(self) -> dict[str, object]:
        """Hyperparameters that define this trainer, for provenance hashing.

        Per-candidate paths are deliberately excluded so the hash identifies the
        training recipe rather than the run directory.
        """
        return {
            "trainer": "mlx-qlora-v1",
            "model": self.model,
            "base_model_revision": self.base_model_revision,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "num_layers": self.num_layers,
            "seed": self.seed,
            "batch_size": 1,
            "max_seq_length": 1024,
        }

    def train(self, cluster: Cluster, candidate_id: str) -> Expert:
        examples = [failure for failure in cluster.failures if failure.correction]
        if not examples:
            raise ValueError("MLX trainer requires verifier-approved corrections")
        dataset_dir = self.artifact_root / candidate_id
        dataset_dir.mkdir(parents=True, exist_ok=False)
        records = [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": failure.task.prompt},
                    {"role": "assistant", "content": failure.correction},
                ]
            }
            for failure in examples
        ]
        train_path = dataset_dir / "train.jsonl"
        train_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        dataset_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
        manifest = {
            "candidate_id": candidate_id,
            "cluster": cluster.label,
            "failure_ids": [failure.id for failure in examples],
            "dataset_sha256": dataset_hash,
            "examples": len(records),
        }
        (dataset_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        remote_data = self.worker.upload_directory(dataset_dir, candidate_id)
        remote_adapter = f"/Users/grove-worker/grove/adapters/{candidate_id}"
        training_spec = {
            "model": self.model,
            "data_path": remote_data,
            "adapter_path": remote_adapter,
            "iterations": self.iterations,
            "batch_size": 1,
            "num_layers": self.num_layers,
            "max_seq_length": 1024,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "timeout_seconds": 7200,
        }
        result = self.worker.run_job("train", training_spec)
        if result.get("status") != "ok":
            raise RuntimeError(f"MLX candidate training failed: {result}")
        artifact = result["artifact"]
        local_adapter = self.worker.download_adapter(
            remote_adapter, Path("/srv/storage/grove/experts") / candidate_id
        )
        specific_tags = sorted(
            {
                tag
                for failure in examples
                for tag in failure.task.tags
                if tag not in {"python", "coding"}
            }
        )
        return Expert(
            id=candidate_id,
            name=f"{cluster.label}-lora",
            status=ExpertStatus.CANDIDATE,
            artifact={
                "backend": "mlx-lora",
                "base_model": self.base_model_revision,
                "model_source": self.model,
                "adapter_path": remote_adapter,
                "local_adapter_path": str(local_adapter),
                "adapter_sha256": artifact["sha256"],
                "dataset_sha256": dataset_hash,
                "parameter_count": artifact["parameter_count"],
                "size_bytes": artifact["size_bytes"],
                "training_job_id": result["job_id"],
                "training_config": training_spec,
            },
            routing_profile={
                "tags": specific_tags,
                "keywords": ["escaped", "path", "backslash", "nested"],
                "tokens": {},
            },
            born_from=tuple(failure.id for failure in examples),
            metrics={
                "training_examples": len(examples),
                "training_duration_seconds": result["duration_seconds"],
            },
        )
