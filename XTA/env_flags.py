"""Shared lenient parsers for environment switches and numeric knobs.

Standard library only, so dependency-light modules (workspace, geometry_quality,
experimental_features) can share one contract. An unset variable returns the
default. A set boolean is False for an integer zero ("0", "00", "-0") or one of
FALSE_VALUES, and True for any other value. Numeric parsers return the default
for an empty or invalid value.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional


FALSE_VALUES = frozenset(("", "0", "false", "no", "off", "disabled"))


def env_flag(
    name: str,
    default: bool = False,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    try:
        return int(value) != 0
    except ValueError:
        return value not in FALSE_VALUES


def env_int(
    name: str,
    default: int,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    source = os.environ if environ is None else environ
    raw = str(source.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def env_float(
    name: str,
    default: float,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> float:
    source = os.environ if environ is None else environ
    raw = str(source.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["FALSE_VALUES", "env_flag", "env_float", "env_int"]
