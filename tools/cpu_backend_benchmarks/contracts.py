"""Common fixture contract; importing it performs no numerical work."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Case:
    name: str
    reference: Callable[[], Any]
    compiled: Callable[[], Any]
    work_units: int
    unit: str
    metadata: dict[str, Any]
