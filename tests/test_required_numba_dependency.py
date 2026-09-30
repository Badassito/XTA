"""Numba is a required compiled runtime dependency, while CLI discovery stays light."""

from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from XTA import _deps


DEPS_PATH = Path(_deps.__file__)
ROOT = DEPS_PATH.parent.parent


@pytest.mark.parametrize("failure", [ImportError("numba missing"), RuntimeError("llvmlite broken")])
def test_numba_import_failure_is_actionable_and_preserves_cause(failure: Exception) -> None:
    original_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "numba":
            raise failure
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("xta_required_numba_probe", DEPS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch("builtins.__import__", side_effect=failing_import):
        with pytest.raises(RuntimeError, match="Numba is required for compiled CPU kernels") as error:
            spec.loader.exec_module(module)
    assert error.value.__cause__ is failure


def test_numba_compiler_can_compile_and_execute_required_kernel() -> None:
    @_deps._numba.njit(cache=False)
    def add(left: int, right: int) -> int:
        return left + right

    assert add(7, 5) == 12
    assert add.signatures


def test_disabled_jit_fails_runtime_import_but_keeps_cli_version_light() -> None:
    env = {**os.environ, "NUMBA_DISABLE_JIT": "1"}
    runtime = subprocess.run(
        [sys.executable, "-c", "import XTA._deps"], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert runtime.returncode != 0
    assert "unset NUMBA_DISABLE_JIT" in runtime.stdout

    version = subprocess.run(
        [sys.executable, "-m", "XTA", "--version"], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert version.returncode == 0, version.stdout
    assert "24.0.5" in version.stdout
