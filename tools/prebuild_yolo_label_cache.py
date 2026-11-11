#!/usr/bin/env python3
"""Prebuild Ultralytics' YOLO label cache before ``yolo train``.

``yolo train`` first verifies every image/label pair and writes
``<split>/labels.cache``. It does this with an 8-thread pool whose label parsing
holds the GIL, so a freshly generated PTA dataset costs minutes before epoch 1
(365,577 pairs took 5 min 9 s in job 166605).

This tool runs Ultralytics' own ``YOLODataset.cache_labels`` with two changes:
- its ThreadPool is replaced by a process pool whose ``imap`` keeps the same
  result order;
- the cache is written to a sibling file and renamed into place, so a stopped
  run never leaves a partial ``labels.cache``. Ultralytics cannot load a
  partial cache file and would stop with an error instead of rescanning.

Everything else is the installed Ultralytics' code: the image list, per-pair
verification, label layout, cache version and hash. ``yolo train`` with the
same arguments therefore loads the cache instead of rescanning.

Two checks then run before the tool reports success:
- a second dataset, built with the original code and with rescanning disabled,
  must accept the cache;
- a sample of pairs, verified again serially, must match the cached labels.

If anything fails, the tool removes the cache (including an existing one it could
not load) so that training does its own scan, and exits with status 1. Status 2
means nothing was written: this Ultralytics version lacks a hook the tool uses,
the task does not use YOLODataset, or the arguments are invalid.

Pass the same key=value arguments as ``yolo train``; only the dataset-related
ones matter (data, task, single_cls, fraction, classes). Two differences from
the yolo CLI: task= is required, because yolo train takes the task from a .pt
checkpoint rather than from task= (it must be the model's task), and cfg=<yaml>
is not expanded, so it is rejected:

    python tools/prebuild_yolo_label_cache.py -- task=segment data=/tmp/ds/data.yaml

On a new machine or Ultralytics version, ``--self-test`` builds a small fixture
with stock Ultralytics and with this tool and requires byte-identical caches.
"""
from __future__ import annotations

import argparse
import contextvars
import functools
import inspect
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

EXIT_OK, EXIT_FAILED, EXIT_UNSUPPORTED = 0, 1, 2
DEFAULT_SAMPLE = 2000
CHUNKSIZE = 64
YOLO_DATASET_TASKS = {'detect', 'segment', 'pose', 'obb'}


class CacheCheckError(RuntimeError):
    """The prebuilt cache differs from what yolo train would build or accept."""


def log(message: str) -> None:
    print(f'label cache: {message}', flush=True)


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def run_in_fresh_context(func, args):
    """Call func as a new ThreadPool thread would: in an empty contextvars context.

    Ultralytics sets NumPy print options at import. In NumPy 2 they live in a
    context variable, so its ThreadPool threads format arrays in messages with
    NumPy's defaults while a pool process's main thread would not.
    """
    return contextvars.Context().run(func, args)


class ProcessImapPool:
    """Stands in for ThreadPool inside cache_labels: same imap order, separate processes."""

    def __init__(self, processes: int, chunksize: int = CHUNKSIZE):
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context('fork' if 'fork' in methods else 'spawn')
        self._pool = context.Pool(processes, initializer=send_dtypes_by_name)
        self._chunksize = chunksize

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._pool.terminate()  # what ThreadPool.__exit__ does
        self._pool.join()
        return False

    def imap(self, func, iterable, chunksize=None):
        return self._pool.imap(functools.partial(run_in_fresh_context, func), iterable,
                               chunksize or self._chunksize)


def reduce_dtype(dtype):
    return np.dtype, (dtype.str,)


def send_dtypes_by_name():
    """Pool initializer: workers send NumPy dtypes by name.

    By default an array unpickled in the parent carries its own copy of, e.g.,
    float32. The values are the same, but the saved cache then stores every
    copy separately and is not byte-identical to the one yolo train writes,
    where all arrays share NumPy's built-in dtype. np.dtype('<f4') returns
    that shared object.
    """
    from multiprocessing.reduction import ForkingPickler

    for code in '?bBhHiIlLqQefdgFDG':
        ForkingPickler.register(type(np.dtype(code)), reduce_dtype)


def missing_hooks(yolo_dataset) -> list[str]:
    """Names this tool replaces or calls that the installed Ultralytics lacks."""
    missing = [name for name in ('ThreadPool', 'save_dataset_cache_file', 'load_dataset_cache_file',
                                 'DATASET_CACHE_VERSION')
               if not hasattr(yolo_dataset, name)]
    dataset = getattr(yolo_dataset, 'YOLODataset', None)
    if dataset is None:
        return missing + ['YOLODataset']
    missing += [f'YOLODataset.{name}' for name in ('cache_labels', 'get_label_files', 'verify_args',
                                                   'result_to_label', 'get_img_files')
                if not hasattr(dataset, name)]
    if 'YOLODataset.cache_labels' not in missing:
        source = inspect.getsource(dataset.cache_labels)
        missing += [f'cache_labels calling {call}' for call in ('ThreadPool(', '.imap(', 'save_dataset_cache_file(')
                    if call not in source]
    return missing


def atomic_save(original, saved: list[Path]):
    """Wrap Ultralytics' save so the cache appears under its final name only when complete."""

    def save_dataset_cache_file(prefix, path, x, version):
        path = Path(path)
        partial = path.with_name(f'{path.name}.prebuild-{os.getpid()}')
        try:
            original(prefix, partial, x, version)
            if not partial.is_file():  # Ultralytics only warns when the save fails
                raise CacheCheckError(f'Ultralytics did not write {partial} (see its warning above)')
            os.replace(partial, path)
            saved.append(path)
        finally:
            partial.unlink(missing_ok=True)

    return save_dataset_cache_file


def same(a, b) -> bool:
    """Exact equality for label dicts: arrays by dtype, shape and value."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def sample_indices(count: int, sample: int) -> set[int]:
    if count <= 0 or sample <= 0:
        return set()
    step = max(1, count // sample)
    return set(range(0, count, step)) | {count - 1}


def compare_sample(dataset, cache: dict, sample: int) -> int:
    """Verify sampled pairs again with the original code and compare with the cache."""
    im_files = dataset.get_img_files(dataset.img_path)
    nf, nm, ne, nc, total = cache['results']
    if total != len(im_files) or len(cache['labels']) + nc != total:
        raise CacheCheckError(f'cache counts {cache["results"]} with {len(cache["labels"])} labels do not match '
                              f'{len(im_files)} images')
    labels = {label['im_file']: label for label in cache['labels']}
    msgs = set(cache['msgs'])
    dataset.im_files = im_files  # the loaded dataset dropped corrupt images; verify_args needs the full list
    dataset.get_label_files()
    func, iterable = dataset.verify_args()
    picked = sample_indices(len(im_files), sample)
    mismatches = []
    for index, args in enumerate(iterable):
        if index not in picked:
            continue
        label, *_, msg = dataset.result_to_label(run_in_fresh_context(func, args))
        cached = labels.get(im_files[index])
        if label is None:
            matches = cached is None  # corrupt pairs are left out of the cache
        else:
            matches = cached is not None and same(label, cached)
        if not matches:
            mismatches.append(im_files[index])
        elif msg and msg not in msgs:  # a restored JPEG reports nothing the second time
            mismatches.append(f'{im_files[index]} (message)')
    if mismatches:
        raise CacheCheckError(f'{len(mismatches)} of {len(picked)} sampled pairs differ from a serial '
                              f're-verify, e.g. {mismatches[:3]}')
    return len(picked)


def prebuild(cfg, data: dict, split: str, processes: int, sample: int) -> None:
    import ultralytics.data.dataset as yolo_dataset
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import load_dataset_cache_file

    # The arguments DetectionTrainer.build_dataset passes. batch and stride only
    # affect rect shapes in memory, not the cache.
    mode = 'train' if split == 'train' else 'val'
    img_path = data['train' if split == 'train' else cfg.split]
    batch = cfg.batch if isinstance(cfg.batch, int) and cfg.batch > 0 else 16

    def build():
        return build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=mode == 'val', stride=32)

    saved: list[Path] = []
    looked_up: list[Path] = []  # the cache path, known even if loading an existing file fails
    cache_path = None
    started = time.monotonic()
    originals = yolo_dataset.ThreadPool, yolo_dataset.save_dataset_cache_file, yolo_dataset.load_dataset_cache_file

    def recording_load(path):
        looked_up.append(Path(path))
        return originals[2](path)

    yolo_dataset.ThreadPool = lambda *args, **kwargs: ProcessImapPool(processes)
    yolo_dataset.save_dataset_cache_file = atomic_save(originals[1], saved)
    yolo_dataset.load_dataset_cache_file = recording_load
    try:
        try:
            built = build()
        finally:
            (yolo_dataset.ThreadPool, yolo_dataset.save_dataset_cache_file,
             yolo_dataset.load_dataset_cache_file) = originals
        cache_path = Path(built.label_files[0]).parent.with_suffix('.cache')
        built_s = time.monotonic() - started
        log(f'{split}: {"built" if saved else "found an existing"} {cache_path} '
            f'({len(built.labels)} labelled images, {built_s:.1f} s, {processes} processes)')
        del built

        # The trainer's path, unmodified, but a rescan is an error: it must load the file.
        started = time.monotonic()
        original_cache_labels = yolo_dataset.YOLODataset.cache_labels

        def refuse_rescan(self, path=Path('./labels.cache')):
            raise CacheCheckError(f'Ultralytics rejected {path} and would rescan')

        yolo_dataset.YOLODataset.cache_labels = refuse_rescan
        try:
            check = build()
        finally:
            yolo_dataset.YOLODataset.cache_labels = original_cache_labels
        accepted_s = time.monotonic() - started

        started = time.monotonic()
        compared = compare_sample(check, load_dataset_cache_file(cache_path), sample)
        log(f'{split}: Ultralytics accepts the cache without rescanning ({accepted_s:.1f} s); '
            f'{compared} sampled pairs match a serial re-verify ({time.monotonic() - started:.1f} s)')
    except BaseException:
        # Also an existing cache that could not be loaded: Ultralytics would fail on it too.
        for path in {*saved, *looked_up, *([cache_path] if cache_path else [])}:
            try:
                Path(path).unlink()
                log(f'{split}: removed {path}')
            except FileNotFoundError:
                pass
            except OSError as error:
                log(f'{split}: could not remove {path}: {error}')
        raise


# Builds the cache exactly as DetectionTrainer.build_dataset does, with stock code.
STOCK_BUILD = '''
import sys
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
cfg = get_cfg(DEFAULT_CFG, {"task": "segment", "data": sys.argv[1]})
data = check_det_dataset(cfg.data)
build_yolo_dataset(cfg, data["train"], 16, data, mode="train", rect=False, stride=32)
'''


def write_fixture(root: Path, count: int = 40, unterminated_jpeg: bool = False) -> Path:
    """A small segment dataset covering the label cases verify_image_label handles."""
    import cv2

    def polygon(vertices):
        angles = np.sort(rng.uniform(0, 2 * np.pi, vertices))
        center, radius = rng.uniform(0.35, 0.65, 2), rng.uniform(0.05, 0.3)
        points = np.clip(center + radius * np.c_[np.cos(angles), np.sin(angles)], 0, 1)
        return '0 ' + ' '.join(f'{value:.6f}' for value in points.ravel())

    rng = np.random.default_rng(7)
    for split, n in (('train', count), ('valid', 4)):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'labels').mkdir(parents=True)
        for index in range(n):
            stem = f'{split}_{index:04d}'
            image = rng.integers(0, 255, (40 + index % 7, 48), dtype=np.uint8)
            cv2.imwrite(str(root / split / 'images' / f'{stem}.jpg'), image)
            lines = [polygon(3 + (index * 37) % 200) for _ in range(1 + index % 4)]
            if split == 'train' and index == 1:
                continue  # missing label: background
            if split == 'train' and index == 2:
                lines = []  # empty label: background
            if split == 'train' and index == 3:
                lines = lines[:1] * 3  # duplicate rows are removed with a message
            if split == 'train' and index == 4:
                lines = ['0 0.1 0.1 1.5 0.2 0.3 0.4']  # out of bounds: corrupt; the message prints an array
            if split == 'train' and index == 5:
                lines = [lines[0].replace('0 ', '3 ', 1)]  # class 3 with nc=1: corrupt
            text = '\n'.join(lines) + ('\n' if lines else '')
            (root / split / 'labels' / f'{stem}.txt').write_text(text, encoding='utf-8')
    if unterminated_jpeg:
        image = root / 'train' / 'images' / 'train_0006.jpg'
        image.write_bytes(image.read_bytes() + b'\0\0')  # not ending in FFD9: Ultralytics rewrites it
    yaml = root / 'data.yaml'
    yaml.write_text(f'path: {root.as_posix()}\ntrain: train/images\nval: valid/images\n\n'
                    "channels: 1\nnc: 1\nnames: ['0']\n", encoding='utf-8')
    return yaml


def throwaway_ultralytics_env(root: Path) -> dict:
    """Environment for a fixture run: its own Ultralytics config folder, holding a
    placeholder Arial.ttf so that check_det_dataset does not download the font."""
    config = root / 'ultralytics-config'
    (config / 'Ultralytics').mkdir(parents=True, exist_ok=True)  # Ultralytics adds this subfolder
    (config / 'Ultralytics' / 'Arial.ttf').write_bytes(b'')
    return dict(os.environ, YOLO_CONFIG_DIR=str(config))


def self_test(processes: int) -> int:
    """Build a fixture's cache with stock Ultralytics and with this tool; the bytes must match."""
    import hashlib
    import subprocess
    import tempfile

    count = 3 * CHUNKSIZE + 7  # several pool chunks
    with tempfile.TemporaryDirectory(prefix='label-cache-self-test-') as temp:
        root = Path(temp)
        env = throwaway_ultralytics_env(root)
        yaml = write_fixture(root / 'dataset', count)
        cache = root / 'dataset' / 'train' / 'labels.cache'

        def run(*args):
            return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace')

        stock = run('-c', STOCK_BUILD, str(yaml))
        if stock.returncode or not cache.is_file():
            log(f'self-test: the stock Ultralytics build failed:\n{stock.stdout[-2000:]}{stock.stderr[-2000:]}')
            return EXIT_FAILED
        expected = hashlib.sha256(cache.read_bytes()).hexdigest()
        cache.unlink()
        built = run(os.path.abspath(__file__), '--processes', str(processes), '--sample', '50',
                    '--', 'task=segment', f'data={yaml}')
        print('\n'.join(line for line in built.stdout.splitlines() if line.startswith('label cache:')), flush=True)
        if built.returncode:
            log(f'self-test: the prebuild failed:\n{built.stdout[-2000:]}{built.stderr[-2000:]}')
            return built.returncode
        actual = hashlib.sha256(cache.read_bytes()).hexdigest()
    if actual != expected:
        log(f'self-test: FAILED: the cache differs from stock Ultralytics ({actual[:16]} vs {expected[:16]})')
        return EXIT_FAILED
    log(f'self-test: {count} pairs; labels.cache is byte-identical to stock Ultralytics ({actual[:16]})')
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Prebuild Ultralytics\' YOLO label cache with a process pool (see the module docstring).')
    parser.add_argument('--processes', type=int, default=0,
                        help='worker processes (default: the CPUs available to this process; '
                             'at most 4 for --self-test)')
    parser.add_argument('--split', action='append', choices=('train', 'val'),
                        help='split to prebuild; repeatable (default: train)')
    parser.add_argument('--sample', type=int, default=DEFAULT_SAMPLE,
                        help=f'pairs to re-verify serially after the build (default: {DEFAULT_SAMPLE}; 0 skips)')
    parser.add_argument('--check-hooks', action='store_true',
                        help='only report whether the installed Ultralytics has the hooks this tool uses')
    parser.add_argument('--self-test', action='store_true',
                        help='build a small fixture with stock Ultralytics and with this tool, and compare the bytes')
    parser.add_argument('overrides', nargs='*', metavar='key=value',
                        help='yolo train arguments, e.g. task=segment data=/path/data.yaml')
    args = parser.parse_args(argv)
    if args.processes < 0 or args.sample < 0:
        parser.error('--processes and --sample must not be negative')

    import ultralytics
    import ultralytics.data.dataset as yolo_dataset

    missing = missing_hooks(yolo_dataset)
    log(f'Ultralytics {ultralytics.__version__}, cache version {getattr(yolo_dataset, "DATASET_CACHE_VERSION", "?")}')
    if missing:
        log(f'unsupported Ultralytics version, missing: {", ".join(missing)}; nothing written')
        return EXIT_UNSUPPORTED
    if args.check_hooks:
        log('hooks present')
        return EXIT_OK
    if args.self_test:
        return self_test(args.processes or min(4, available_cpus()))

    from ultralytics.cfg import get_cfg, parse_key_value_pair
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    overrides = dict(parse_key_value_pair(pair) for pair in args.overrides)
    if 'data' not in overrides or 'task' not in overrides:
        parser.error('data=<data.yaml> and task=<the model\'s task> are required')
    if 'cfg' in overrides:
        parser.error('cfg=<yaml> is not supported; pass its arguments directly')
    cfg = get_cfg(DEFAULT_CFG, overrides)
    if cfg.task not in YOLO_DATASET_TASKS:
        log(f'task={cfg.task} does not use YOLODataset; nothing written')
        return EXIT_UNSUPPORTED
    # As BaseTrainer.get_dataset does for a data YAML.
    data = check_det_dataset(cfg.data)
    if cfg.single_cls:
        data['names'] = {0: 'item'}
        data['nc'] = 1

    processes = args.processes or available_cpus()
    status = EXIT_OK
    for split in args.split or ['train']:
        try:
            prebuild(cfg, data, split, processes, args.sample)
        except Exception as error:  # report, keep going with other splits
            log(f'{split}: FAILED: {type(error).__name__}: {error}')
            status = EXIT_FAILED
    return status


if __name__ == '__main__':
    sys.exit(main())
