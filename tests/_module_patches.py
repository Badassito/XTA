"""Patch explicit lazy dependency entries without rolling back real imports."""
from contextlib import contextmanager
import sys


@contextmanager
def patch_module_entries(entries):
    """Restore only the requested keys; unrelated imports remain initialized.

    Use for dependencies imported inside the tested call. Modules whose globals
    would retain these replacements require a separate process instead.
    """
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in entries}
    sys.modules.update(entries)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
