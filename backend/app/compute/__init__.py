"""Compute backends. `get_backend(name)` returns the impl; the rest of the app
talks only to the ComputeBackend interface and never cares where a job runs."""
from __future__ import annotations

from .base import ComputeBackend
from .local import LocalBackend
from .daytona import DaytonaBackend

_REGISTRY = {"local": LocalBackend, "daytona": DaytonaBackend}


def get_backend(name: str) -> ComputeBackend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; choices: {list(_REGISTRY)}")
    return _REGISTRY[name]()
