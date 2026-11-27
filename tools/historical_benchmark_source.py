"""Load exact historical comparison code with its own package dependencies."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys


def load_historical_module(source: Path, package: Path, name: str, *, evidence_dir: Path):
    """Keep the archived package separate from the active XTA runtime."""
    package = package.resolve()
    source = source.resolve()
    initializer = package / "__init__.py"
    if not initializer.is_file():
        raise ValueError(f"Historical comparison requires the matching XTA source package: {package}")
    if not (package / "native_cpu.py").is_file():
        os.environ["NUMBA_CACHE_DIR"] = str(evidence_dir / "historical-numba-cache")
    prefix = "_xta_historical_" + hashlib.sha256(str(package).encode()).hexdigest()[:16]
    if prefix not in sys.modules:
        _load(prefix, initializer, submodule_search_locations=[str(package)])
    fullname = f"{prefix}.{name}"
    prior = sys.modules.get(fullname)
    if prior is not None and Path(prior.__file__).resolve() == source:
        return prior
    return _load(fullname, source)


def _load(name: str, source: Path, **kwargs):
    spec = importlib.util.spec_from_file_location(name, source, **kwargs)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load historical benchmark source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
    return module
