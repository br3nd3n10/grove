from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SshWorkerConfig:
    host: str = field(
        default_factory=lambda: os.environ.get("GROVE_WORKER_HOST") or "grove-worker-1"
    )
    user: str = "grove-worker"
    identity_file: str = field(
        default_factory=lambda: (
            os.environ.get("GROVE_WORKER_IDENTITY")
            or str(Path.home() / ".ssh" / "grove_worker")
        )
    )
    remote_repository: str = "/Users/grove-worker/grove/runtime/repo"
    remote_jobs: str = "/Users/grove-worker/grove/jobs"


class MlxSshWorker:
    def __init__(self, config: SshWorkerConfig | None = None) -> None:
        self.config = config or SshWorkerConfig()

    def preflight(self, model_path: str | None = None) -> dict[str, Any]:
        """Worker identity, framework versions, checkout and model manifest.

        ``model_path`` names the base-model directory the worker should hash.
        The control host cannot reach that path, so without this the run has no
        evidence at all about which model files the worker holds.
        """
        model_argument = f" --model {shlex.quote(model_path)}" if model_path else ""
        repository = shlex.quote(self.config.remote_repository)
        result = self._ssh(
            f"cd {repository} && "
            f"{self._environment()} .venv/bin/python -m grove.mlx_worker preflight"
            f"{model_argument} --repository {repository}"
        )
        return json.loads(result.stdout)

    def run_job(self, command: str, spec: dict[str, Any]) -> dict[str, Any]:
        if command not in {"infer", "train"}:
            raise ValueError(f"unsupported MLX worker command: {command}")
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        remote_dir = f"{self.config.remote_jobs}/{job_id}"
        create = self._ssh(f"umask 077 && mkdir -p {remote_dir}")
        if create.returncode != 0:
            raise RuntimeError(create.stderr)
        upload = self._ssh(
            f"dd of={remote_dir}/spec.json status=none",
            input_text=json.dumps(spec, sort_keys=True),
        )
        if upload.returncode != 0:
            raise RuntimeError(upload.stderr)
        run = self._ssh(
            f"cd {self.config.remote_repository} && {self._environment()} "
            f".venv/bin/python -m grove.mlx_worker {command} "
            f"--spec {remote_dir}/spec.json --output {remote_dir}/result.json"
        )
        fetch = self._ssh(f"dd if={remote_dir}/result.json status=none")
        if fetch.returncode != 0:
            raise RuntimeError(fetch.stderr or run.stderr)
        result = json.loads(fetch.stdout)
        result["job_id"] = job_id
        return result

    def upload_directory(self, local_path: str | Path, remote_name: str) -> str:
        if not remote_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "remote dataset name must be alphanumeric with dashes/underscores"
            )
        remote_path = f"/Users/grove-worker/grove/datasets/{remote_name}"
        create = self._ssh(f"umask 077 && mkdir -p {remote_path}")
        if create.returncode != 0:
            raise RuntimeError(create.stderr)
        transfer = subprocess.run(
            [
                "rsync",
                "-az",
                "--delete",
                "-e",
                f"ssh -i {self.config.identity_file} -o IdentitiesOnly=yes",
                f"{Path(local_path).resolve()}/",
                f"{self.config.user}@{self.config.host}:{remote_path}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if transfer.returncode != 0:
            raise RuntimeError(transfer.stderr)
        return remote_path

    def download_adapter(self, remote_path: str, local_path: str | Path) -> Path:
        allowed = "/Users/grove-worker/grove/adapters/"
        if not remote_path.startswith(allowed) or remote_path == allowed:
            raise ValueError("remote adapter path is outside the worker adapter store")
        adapter_name = remote_path.removeprefix(allowed)
        if (
            "/" in adapter_name
            or not adapter_name.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("remote adapter name contains unsafe characters")
        destination = Path(local_path).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        transfer = subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh -i {self.config.identity_file} -o IdentitiesOnly=yes",
                f"{self.config.user}@{self.config.host}:{remote_path}/",
                f"{destination}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if transfer.returncode != 0:
            raise RuntimeError(transfer.stderr)
        return destination

    @staticmethod
    def _environment() -> str:
        return (
            "HF_HOME=/Users/grove-worker/grove/cache/huggingface "
            "TOKENIZERS_PARALLELISM=false"
        )

    def _ssh(
        self, command: str, *, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "ssh",
                "-i",
                self.config.identity_file,
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "BatchMode=yes",
                f"{self.config.user}@{self.config.host}",
                command,
            ],
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
