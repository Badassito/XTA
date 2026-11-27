"""Lazy binding to the installed or explicitly prepared CPU kernel extension."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import sysconfig
import threading


_native_cpu_module = None
_native_cpu_binary_digest = None
_native_cpu_load_lock = threading.Lock()


def _after_fork() -> None:
    # A forked render worker can inherit a lock owned by a vanished thread.
    global _native_cpu_load_lock
    _native_cpu_load_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def load_native_cpu():
    """Return the compiled kernel module without build or scheduler side effects."""
    global _native_cpu_module, _native_cpu_binary_digest
    if _native_cpu_module is not None:
        return _native_cpu_module
    with _native_cpu_load_lock:
        if _native_cpu_module is not None:
            return _native_cpu_module
        from .native_cpu_build import (
            ABI_VERSION, ARTIFACTS_ENV, BINARY_ENV, MODULE, NativeCpuBuildError, SOURCE_ROOT,
            binary_sha256, cpu_source_identity, find_native_cpu_binary, is_source_checkout,
        )

        checkout = is_source_checkout()
        try:
            if os.environ.get(BINARY_ENV) and not checkout:
                raise NativeCpuBuildError(f"{BINARY_ENV} requires a matching source checkout")
            source_sha256 = cpu_source_identity() if checkout else None
            artifact = find_native_cpu_binary(source_sha256=source_sha256) if checkout else None
            if artifact is not None:
                spec = importlib.util.spec_from_file_location(MODULE, artifact.path)
            elif checkout:
                spec = importlib.util.find_spec(MODULE)
            else:
                # An installed wheel keeps its compiled code beside the XTA
                # package. Ignore same-name modules in earlier sys.path entries.
                suffix = sysconfig.get_config_var("EXT_SUFFIX")
                spec = importlib.util.spec_from_file_location(MODULE, SOURCE_ROOT / (MODULE + suffix)) if suffix else None
            if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
                raise ImportError("The native CPU extension binary was not found")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if getattr(module, "ABI_VERSION", None) != ABI_VERSION:
                raise NativeCpuBuildError("Unsupported native CPU kernel ABI")
            if checkout and getattr(module, "SOURCE_SHA256", None) != source_sha256:
                raise NativeCpuBuildError("The native CPU extension does not match this source checkout")
            digest = artifact.sha256 if artifact is not None else binary_sha256(Path(module.__file__))
        except (ImportError, NativeCpuBuildError, OSError) as error:
            instruction = "Install a matching platform XTA wheel."
            if checkout:
                tool = SOURCE_ROOT / "tools" / "build_native_cpu.py"
                instruction += (
                    f' For this source checkout, run "{Path(sys.executable)}" -B "{tool}".'
                    f" Use {ARTIFACTS_ENV} to share an outside-checkout artifact directory."
                )
            raise NativeCpuBuildError(f"Native CPU kernels unavailable. {instruction}\n{error}") from error
        sys.modules[MODULE] = module
        _native_cpu_binary_digest = digest
        _native_cpu_module = module
        return module


def native_cpu_provenance(domain: str) -> dict[str, str]:
    """Identify one native implementation without reading its artifact per call."""
    module = load_native_cpu()
    source_sha256 = getattr(module, domain + "_SOURCE_SHA256")
    return {
        "source_sha256": source_sha256,
        "binary_sha256": _native_cpu_binary_digest,
        "compiler": module.COMPILER,
        "fp_contract": module.FP_CONTRACT,
    }


class _KernelBindings:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(load_native_cpu(), name)


kernels = _KernelBindings()
