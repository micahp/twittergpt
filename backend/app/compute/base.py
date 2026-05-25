"""The ComputeBackend interface — the seam between the web app and where
finetuning actually runs (local desktop GPU vs Daytona cloud). Implementations
must be safe to call from the request thread (submit returns immediately)."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainRequest:
    job_id: str
    dataset_dir: Path        # contains train.jsonl + meta.json
    adapter_out: Path        # where the LoRA adapter should land
    profile: dict            # base model + sizing (see config.PROFILES)


@dataclass
class JobStatus:
    status: str              # queued|running|done|failed
    log_tail: str = ""
    error: str | None = None
    extra: dict | None = None


class ComputeBackend(abc.ABC):
    """One base model, per-user LoRA adapters. Backends differ only in *where*
    the training subprocess/workspace executes."""

    name: str = "base"

    @abc.abstractmethod
    def submit_training(self, req: TrainRequest) -> None:
        """Kick off training for req.job_id. Must not block until completion."""

    @abc.abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Return current status; reads process state + log tail."""

    @abc.abstractmethod
    def fetch_artifact(self, job_id: str) -> Path:
        """Return the local path to the produced adapter (after status == done)."""
