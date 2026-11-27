"""Record the authenticated native CPU binary used by development probes."""

import hashlib
import inspect
from pathlib import Path


def native_cpu_provenance(*symbols: str) -> dict:
    """Load the native extension and verify any required compiled exports."""
    from XTA.native_cpu import load_native_cpu

    module = load_native_cpu()
    for name in symbols:
        if not inspect.isbuiltin(getattr(module, name, None)):
            raise RuntimeError(f"Native CPU extension does not export {name!r}")
    path = Path(module.__file__).resolve()
    return {
        "path": str(path),
        "binary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": module.SOURCE_SHA256,
        "abi_version": module.ABI_VERSION,
        "verified_exports": list(symbols),
    }
