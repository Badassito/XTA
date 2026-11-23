"""Isolate process-wide test state."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import pytest


_DEPENDENCY_MODULES = ('cv2', 'scipy', 'scipy.ndimage', 'tifffile', 'tqdm', 'numba')
_REPO_SETTINGS = Path(__file__).resolve().parents[1] / 'Ultralytics' / 'settings.json'

# On Windows, importing Ultralytics replaces cv2.imread/imwrite/imshow for the
# whole process, and its imread returns grayscale as (H, W, 1). Linux runs never
# see that patch. Capture OpenCV's own functions before any test can import
# Ultralytics, and restore them after each test so decode shapes do not depend
# on whether an earlier test happened to load a YOLO model.
try:
    import cv2 as _cv2
except ImportError:  # OpenCV is optional for some test environments.
    _cv2 = None
_OPENCV_IO = ({name: getattr(_cv2, name) for name in ('imread', 'imwrite', 'imshow')}
              if _cv2 is not None else {})


def _is_smoke_stub(module):
    kind = type(module)
    return kind.__module__ == 'tools.smoke_import' and kind.__name__ == '_StubModule'


def _tf32_flags_if_loaded():
    torch = sys.modules.get('torch')
    if torch is None:
        return None
    try:
        return (bool(torch.backends.cuda.matmul.allow_tf32),
                bool(torch.backends.cudnn.allow_tf32))
    except AttributeError:
        return None


@pytest.fixture(autouse=True)
def _guard_process_wide_test_state():
    settings_existed = _REPO_SETTINGS.exists()
    previous = {name: sys.modules[name] for name in _DEPENDENCY_MODULES
                if name in sys.modules}
    initial_tf32 = _tf32_flags_if_loaded()
    yield
    for name, function in _OPENCV_IO.items():
        if getattr(_cv2, name) is not function:
            setattr(_cv2, name, function)
    assert settings_existed or not _REPO_SETTINGS.exists(), (
        f'Ultralytics settings appeared in the repository during this test: {_REPO_SETTINGS}; '
        'create YOLO_CONFIG_DIR before importing Ultralytics in child processes'
    )
    for name, original in previous.items():
        assert sys.modules.get(name) is original, f'{name} was replaced in sys.modules by a test'
    for name in _DEPENDENCY_MODULES:
        assert not _is_smoke_stub(sys.modules.get(name)), f'{name} stub leaked into sys.modules'
    deps = sys.modules.get('XTA._deps')
    if deps is not None:
        for name in ('cv2', 'ndi', 'tifffile', 'tqdm'):
            assert not _is_smoke_stub(getattr(deps, name, None)), f'XTA._deps.{name} retained a stub'
    if initial_tf32 is not None:
        assert _tf32_flags_if_loaded() == initial_tf32, 'Torch TF32 flags changed during a test'


def _reset_cgroup_layout_cache_if_loaded():
    reset = getattr(sys.modules.get('XTA.workspace'), '_reset_cgroup_layout_cache', None)
    if callable(reset):
        reset()


@pytest.fixture(autouse=True)
def _fresh_cgroup_layout_cache():
    """Each test resolves its own (usually fake) cgroup membership and mounts."""
    _reset_cgroup_layout_cache_if_loaded()
    yield
    _reset_cgroup_layout_cache_if_loaded()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    yield
    restore = getattr(item.config, '_xta_restore_windows_empty_environment', None)
    if restore is not None:
        restore()


def pytest_configure(config):
    if os.name == 'nt':
        inherited_empty = tuple(key for key, value in os.environ.items() if value == '')
        if inherited_empty:
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            read = kernel.GetEnvironmentVariableW
            read.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
            read.restype = wintypes.DWORD
            write = kernel.SetEnvironmentVariableW
            write.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
            write.restype = wintypes.BOOL

            def restore_empty_environment():
                # The Windows CRT drops empty values when a test restores os.environ.
                for key in inherited_empty:
                    if (os.environ.get(key) == '' and read(key, None, 0) == 0
                            and ctypes.get_last_error() == 203):  # ERROR_ENVVAR_NOT_FOUND
                        if not write(key, ''):
                            raise ctypes.WinError(ctypes.get_last_error())

            config._xta_restore_windows_empty_environment = restore_empty_environment

    previous = os.environ.get('YOLO_CONFIG_DIR')
    config._xta_previous_yolo_config_dir = previous
    if previous:
        config_dir = Path(previous).expanduser().resolve()
        # Ultralytics falls back to CWD when the configured directory's parent
        # does not exist yet, so create the base before test-module imports.
        config_dir.mkdir(parents=True, exist_ok=True)
    else:
        scratch_temp = Path(__file__).resolve().parents[2] / 'Scratch' / 'Temp'
        temporary = tempfile.TemporaryDirectory(
            prefix='xta-pytest-ultralytics-',
            dir=str(scratch_temp) if scratch_temp.is_dir() else None,
        )
        config._xta_yolo_config_temp = temporary
        config_dir = Path(temporary.name)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)


def pytest_unconfigure(config):
    previous = getattr(config, '_xta_previous_yolo_config_dir', None)
    if previous is None:
        os.environ.pop('YOLO_CONFIG_DIR', None)
    else:
        os.environ['YOLO_CONFIG_DIR'] = previous
    temporary = getattr(config, '_xta_yolo_config_temp', None)
    if temporary is not None:
        temporary.cleanup()
