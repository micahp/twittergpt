"""DaytonaBackend — provision a Daytona GPU workspace, sync the dataset, run the
same train_qlora.py there, pull back the adapter. Stubbed for M3; wired in M2.

The shape mirrors LocalBackend so the rest of the app is unchanged when you flip
the profile from local-3b to daytona-7b. Implementation notes for M2:

  1. Create/start a workspace from the Daytona SDK/CLI with a GPU image that has
     CUDA + the training deps (or run `pip install -r requirements-train.txt`).
  2. Upload dataset_dir + train_qlora.py to the workspace.
  3. Exec the same command LocalBackend builds; stream logs back to the job log.
  4. On completion, download the adapter dir to req.adapter_out and stop the
     workspace so credits aren't burned idle.
"""
from __future__ import annotations

from pathlib import Path

from .base import ComputeBackend, JobStatus, TrainRequest

_NOT_WIRED = (
    "DaytonaBackend is a stub (M3 skeleton). Implement workspace provisioning in M2, "
    "or use the local backend (profile 'local-3b') for now."
)


class DaytonaBackend(ComputeBackend):
    name = "daytona"

    def submit_training(self, req: TrainRequest) -> None:
        raise NotImplementedError(_NOT_WIRED)

    def status(self, job_id: str) -> JobStatus:
        return JobStatus(status="failed", error=_NOT_WIRED)

    def fetch_artifact(self, job_id: str) -> Path:
        raise NotImplementedError(_NOT_WIRED)
