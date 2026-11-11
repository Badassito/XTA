"""Per-input-directory PTA option overrides.

An input directory may hold ``pta_input_options.json``: a JSON object with
only ``fraction`` (number, 0 < fraction <= 1) and ``enable_cpr`` (boolean).
Its values override the command line for every volume discovered in that
directory. Discovery validates the file before any volume is probed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

PTA_INPUT_OPTIONS_FILENAME = "pta_input_options.json"
PTA_INPUT_OPTION_KEYS = ("fraction", "enable_cpr")


@dataclass(frozen=True)
class PtaInputOptions:
    path: Path
    sha256: str
    fraction: Optional[float] = None
    enable_cpr: Optional[bool] = None

    def overrides(self) -> Dict[str, object]:
        values = {"fraction": self.fraction, "enable_cpr": self.enable_cpr}
        return {key: value for key, value in values.items() if value is not None}


def _unique_object(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"duplicate key {key!r}")
        values[key] = value
    return values


def read_pta_input_options(input_dir: Path) -> Optional[PtaInputOptions]:
    """Return validated overrides, or None when the directory has no options file."""
    path = Path(input_dir) / PTA_INPUT_OPTIONS_FILENAME
    if not os.path.lexists(path):
        return None
    try:
        payload = path.read_bytes()
        values = json.loads(payload, object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid PTA input options {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"PTA input options {path} must be a JSON object")
    unknown = sorted(set(values) - set(PTA_INPUT_OPTION_KEYS))
    if unknown:
        raise ValueError(
            f"PTA input options {path} has unknown key(s) {unknown}; "
            f"allowed keys: {list(PTA_INPUT_OPTION_KEYS)}"
        )
    fraction = values.get("fraction")
    if "fraction" in values:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            raise ValueError(f"PTA input options {path}: fraction must be a number in (0,1], got {fraction!r}")
        fraction = float(fraction)
    enable_cpr = values.get("enable_cpr")
    if "enable_cpr" in values and not isinstance(enable_cpr, bool):
        raise ValueError(f"PTA input options {path}: enable_cpr must be true or false, got {enable_cpr!r}")
    return PtaInputOptions(path=path.resolve(), sha256=hashlib.sha256(payload).hexdigest(),
                           fraction=fraction, enable_cpr=enable_cpr)


def resolve_volume_options(
    spec: object, args: argparse.Namespace, *, fraction: float,
) -> Tuple[float, bool, argparse.Namespace]:
    """Return one volume's (fraction, enable_cpr, args); CLI values are the defaults."""
    options = getattr(spec, "input_options", None)
    cli_cpr = bool(getattr(args, "enable_cpr", False))
    if options is None:
        return float(fraction), cli_cpr, args
    volume_fraction = float(fraction) if options.fraction is None else float(options.fraction)
    volume_cpr = cli_cpr if options.enable_cpr is None else bool(options.enable_cpr)
    if volume_cpr != cli_cpr:
        args = argparse.Namespace(**{**vars(args), "enable_cpr": volume_cpr})
    return volume_fraction, volume_cpr, args


__all__ = [
    "PTA_INPUT_OPTIONS_FILENAME",
    "PTA_INPUT_OPTION_KEYS",
    "PtaInputOptions",
    "read_pta_input_options",
    "resolve_volume_options",
]
