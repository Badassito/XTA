"""Build and authenticate the native scheduler cache for source launches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


CACHE_ENV = "XTA_NATIVE_SCHEDULER_CACHE"
_MODULE = "XTA._prepared_scheduler"
_RECEIPT = "native_scheduler_build.json"
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_BUILD_INPUTS = (
    "setup.py", "pyproject.toml", "native/prepared_scheduler.cpp", "native/prepared_scheduler.hpp",
    "native/device_stage.hpp", "native/prepared_scheduler_handles.hpp",
)


class NativeSchedulerBuildError(RuntimeError):
    """The mandatory native scheduler could not be supplied for a source launch."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_identity() -> dict:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if sys.implementation.name != "cpython" or not suffix:
        raise NativeSchedulerBuildError("The native TTA scheduler requires a CPython extension ABI.")
    try:
        sources = {name: _sha256(_SOURCE_ROOT / name) for name in _BUILD_INPUTS}
    except OSError as error:
        raise NativeSchedulerBuildError(
            "The native TTA scheduler is missing. Install a platform XTA wheel or use a complete source checkout "
            f"with a C++17 compiler. Build input unavailable: {error.filename or error}"
        ) from error
    return {
        "schema": "xta.native_scheduler_build/1",
        "module": _MODULE,
        "filename": "_prepared_scheduler" + suffix,
        "python_abi": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "sources": sources,
    }


def _cache_root(cache_dir) -> Path:
    configured = cache_dir if cache_dir is not None else os.environ.get(CACHE_ENV)
    if configured:
        root = Path(configured)
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "XTA" / "native-scheduler"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "xta" / "native-scheduler"
    root = root.expanduser().resolve()
    if root.is_relative_to(_SOURCE_ROOT):
        raise NativeSchedulerBuildError(f"{CACHE_ENV} must be outside the XTA source checkout: {root}")
    return root


def validate_native_scheduler_binary(path: str | Path) -> Path:
    """Check an explicit source-built artifact against this checkout and interpreter."""
    return _validate_binary(Path(path).expanduser().resolve(), _build_identity())


def _validate_binary(path: Path, identity: dict) -> Path:
    try:
        if path.is_relative_to(_SOURCE_ROOT) or path.name != identity["filename"] or path.parent.name != "XTA":
            raise ValueError("unexpected location or CPython extension filename")
        receipt = json.loads((path.parent.parent / _RECEIPT).read_text(encoding="utf-8"))
        if receipt != {**identity, "binary_sha256": _sha256(path)}:
            raise ValueError("source, ABI or binary digest differs")
    except (OSError, UnicodeError, ValueError) as error:
        raise NativeSchedulerBuildError(
            f"Invalid native TTA scheduler artifact {path}: {error}. "
            "Build a fresh cache with tools/build_native_scheduler.py --cache-dir OUTSIDE_CHECKOUT."
        ) from error
    return path


def _compile_extension(directory: Path, identity: dict) -> Path:
    output = directory / "out"
    temporary = directory / "temp"
    temporary.mkdir()
    environment = {
        **os.environ,
        "XTA_BUILD_EXTENSION": "scheduler",
        "TMPDIR": str(temporary),
        "TMP": str(temporary),
        "TEMP": str(temporary),
    }
    command = [
        sys.executable, "-X", "utf8", "-B", "setup.py", "build_ext",
        "--build-lib", str(output), "--build-temp", str(temporary / "objects"),
    ]
    result = subprocess.run(
        command, cwd=_SOURCE_ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    binary = output / "XTA" / identity["filename"]
    if result.returncode or not binary.is_file():
        raise NativeSchedulerBuildError(
            "The required native TTA scheduler could not be built. Install an XTA wheel, or build from source "
            "with setuptools, CPython development headers, and a C++17 toolchain "
            "(MSVC Build Tools and Windows SDK on Windows, or a C++ compiler and pthread on Linux).\n"
            + result.stdout
        )
    check_import = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2]); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)"
    )
    loaded = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-B", "-c", check_import, _MODULE, str(binary)],
        cwd=directory, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    if loaded.returncode:
        raise NativeSchedulerBuildError(f"The built native TTA scheduler failed its import check:\n{loaded.stdout}")
    return binary


def build_native_scheduler(*, cache_dir: str | Path | None = None) -> Path:
    """Reuse or build the source scheduler in a private external cache."""
    identity = _build_identity()
    root = _cache_root(cache_dir)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    destination = root / key
    binary = destination / "XTA" / identity["filename"]
    if destination.exists():
        return _validate_binary(binary, identity)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as stage:
            built = _compile_extension(Path(stage), identity)
            if _build_identity() != identity:
                raise NativeSchedulerBuildError("Native scheduler sources changed during the build. Retry the build.")
            publish = built.parent.parent
            (publish / _RECEIPT).write_text(
                json.dumps({**identity, "binary_sha256": _sha256(built)}, sort_keys=True) + "\n", encoding="utf-8",
            )
            try:
                publish.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
                # A concurrent preflight may have already published the same key.
    except OSError as error:
        raise NativeSchedulerBuildError(f"Native TTA scheduler build/cache failed at {root}: {error}") from error
    return _validate_binary(binary, identity)
