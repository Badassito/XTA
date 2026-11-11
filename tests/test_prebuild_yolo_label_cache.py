"""The label cache prebuilt with a process pool is the one yolo train would write."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

import numpy as np

from tools import prebuild_yolo_label_cache as prebuild

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / 'tools' / 'prebuild_yolo_label_cache.py'
HAS_ULTRALYTICS = importlib.util.find_spec('ultralytics') is not None

STOCK_BUILD = prebuild.STOCK_BUILD
make_dataset = prebuild.write_fixture


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAS_ULTRALYTICS, 'ultralytics is not installed')
class PrebuildMatchesUltralyticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='xta-label-cache-')
        self.root = Path(self.temp.name)
        self.env = dict(prebuild.throwaway_ultralytics_env(self.root), PYTHONDONTWRITEBYTECODE='1')
        self.dataset = self.root / 'dataset'
        self.cache = self.dataset / 'train' / 'labels.cache'

    def tearDown(self):
        self.temp.cleanup()

    def python(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, *args], cwd=REPO, env=self.env, capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=600)

    def tool(self, yaml: Path, *options: str) -> subprocess.CompletedProcess:
        return self.python(str(TOOL), '--processes', '2', *options, '--', 'task=segment', f'data={yaml}')

    def driver(self, setup: str, yaml: Path) -> subprocess.CompletedProcess:
        """Run main() after `setup` modifies the tool module `t` in the child process."""
        code = (f'import sys; sys.path.insert(0, {str(REPO)!r})\n'
                'import tools.prebuild_yolo_label_cache as t\n'
                f'{setup}\n'
                f'raise SystemExit(t.main(["--processes", "2", "--", "task=segment", "data={yaml.as_posix()}"]))\n')
        return self.python('-c', code)

    def assert_ok(self, result: subprocess.CompletedProcess):
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])

    def test_self_test_matches_stock_ultralytics_byte_for_byte(self):
        result = self.python(str(TOOL), '--self-test', '--processes', '2')
        self.assert_ok(result)
        self.assertIn('byte-identical to stock Ultralytics', result.stdout)
        self.assertIn('accepts the cache without rescanning', result.stdout)

    def test_cache_bytes_match_stock_ultralytics_and_an_existing_cache_is_kept(self):
        # Several pool chunks (64 pairs each), so arrays arrive in separate messages.
        yaml = make_dataset(self.dataset, count=3 * prebuild.CHUNKSIZE + 7)
        self.assert_ok(self.python('-c', STOCK_BUILD, str(yaml)))
        stock = sha256(self.cache)
        self.cache.unlink()

        result = self.tool(yaml)
        self.assert_ok(result)
        self.assertEqual(sha256(self.cache), stock)
        self.assertIn('accepts the cache without rescanning', result.stdout)
        self.assertFalse(list(self.cache.parent.glob('labels.cache.prebuild-*')))

        again = self.tool(yaml)  # an existing valid cache is kept and checked
        self.assert_ok(again)
        self.assertIn('found an existing', again.stdout)
        self.assertEqual(sha256(self.cache), stock)

    def test_restored_jpeg_matches_stock_ultralytics(self):
        from ultralytics.data.utils import load_dataset_cache_file

        pristine = self.root / 'pristine'
        make_dataset(pristine, unterminated_jpeg=True)
        shutil.copytree(pristine, self.dataset)
        yaml = self.dataset / 'data.yaml'
        yaml.write_text(yaml.read_text(encoding='utf-8').replace(pristine.as_posix(), self.dataset.as_posix()),
                        encoding='utf-8')
        self.assert_ok(self.python('-c', STOCK_BUILD, str(yaml)))
        stock = load_dataset_cache_file(self.cache)
        self.cache.unlink()
        shutil.copy2(pristine / 'train' / 'images' / 'train_0006.jpg', self.dataset / 'train' / 'images')

        self.assert_ok(self.tool(yaml))
        built = load_dataset_cache_file(self.cache)
        self.assertTrue(any('corrupt JPEG restored' in msg for msg in built['msgs']))
        # The hash covers file times in newer Ultralytics, and the restore rewrote one file.
        for key in ('labels', 'results', 'msgs', 'version'):
            self.assertTrue(prebuild.same(built[key], stock[key]), key)

    def test_a_cache_ultralytics_rejects_is_removed(self):
        yaml = make_dataset(self.dataset)
        setup = textwrap.dedent('''
            original = t.atomic_save
            def tampered(save, saved):
                def wrapper(prefix, path, x, version):
                    x["hash"] = "not the dataset hash"
                    return original(save, saved)(prefix, path, x, version)
                return wrapper
            t.atomic_save = tampered
        ''')
        result = self.driver(setup, yaml)
        self.assertEqual(result.returncode, prebuild.EXIT_FAILED, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertIn('rejected', result.stdout)
        self.assertFalse(self.cache.exists())

    def test_a_sample_mismatch_removes_the_cache(self):
        yaml = make_dataset(self.dataset)
        result = self.driver('t.same = lambda a, b: False', yaml)
        self.assertEqual(result.returncode, prebuild.EXIT_FAILED, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertIn('sampled pairs differ', result.stdout)
        self.assertFalse(self.cache.exists())

    def test_an_unreadable_existing_cache_is_removed(self):
        yaml = make_dataset(self.dataset)
        self.cache.write_bytes(b'\x93NUMPY truncated')  # e.g. a stock save that was killed
        result = self.tool(yaml)
        self.assertEqual(result.returncode, prebuild.EXIT_FAILED, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertIn(f'removed {self.cache}', result.stdout)
        self.assertFalse(self.cache.exists())

    def test_task_is_required_and_cfg_is_rejected(self):
        yaml = make_dataset(self.dataset)
        for overrides in ([f'data={yaml}'], ['task=segment', f'data={yaml}', 'cfg=custom.yaml']):
            result = self.python(str(TOOL), '--', *overrides)
            self.assertEqual(result.returncode, prebuild.EXIT_UNSUPPORTED, result.stderr[-2000:])
        self.assertFalse(self.cache.exists())

    def test_a_failure_before_any_cache_reports_no_removal(self):
        yaml = make_dataset(self.dataset)
        shutil.rmtree(self.dataset / 'train' / 'labels')  # 'No labels found' before anything is saved
        result = self.tool(yaml)
        self.assertEqual(result.returncode, prebuild.EXIT_FAILED, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertNotIn('removed', result.stdout)
        self.assertIn('FAILED', result.stdout)

    def test_the_font_placeholder_is_where_ultralytics_looks(self):
        code = 'from ultralytics.utils.checks import check_font; print(check_font("Arial.ttf"))'
        result = self.python('-c', code)
        self.assert_ok(result)
        expected = Path(self.env['YOLO_CONFIG_DIR']) / 'Ultralytics' / 'Arial.ttf'
        self.assertEqual(Path(result.stdout.strip().splitlines()[-1]), expected)

    def test_unsupported_ultralytics_writes_nothing(self):
        yaml = make_dataset(self.dataset)
        result = self.driver('t.missing_hooks = lambda module: ["ThreadPool"]', yaml)
        self.assertEqual(result.returncode, prebuild.EXIT_UNSUPPORTED, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertFalse(self.cache.exists())

    def test_installed_ultralytics_has_the_hooks(self):
        import ultralytics.data.dataset as yolo_dataset

        self.assertEqual(prebuild.missing_hooks(yolo_dataset), [])


class PrebuildHelperTests(unittest.TestCase):
    def test_an_interrupted_save_leaves_no_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'labels.cache'

            def failing(prefix, partial, x, version):
                Path(partial).write_bytes(b'partial')
                raise OSError('disk full')

            saved = []
            with self.assertRaises(OSError):
                prebuild.atomic_save(failing, saved)('', path, {}, '1')
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), [])
            self.assertEqual(saved, [])

    def test_a_save_ultralytics_skipped_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'labels.cache'
            with self.assertRaises(prebuild.CacheCheckError):
                prebuild.atomic_save(lambda *args: None, [])('', path, {}, '1')
            self.assertFalse(path.exists())

    def test_a_complete_save_is_renamed_into_place(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'labels.cache'
            saved = []
            prebuild.atomic_save(lambda prefix, partial, x, version: Path(partial).write_bytes(b'cache'),
                                 saved)('', path, {}, '1')
            self.assertEqual(path.read_bytes(), b'cache')
            self.assertEqual(saved, [path])
            self.assertEqual([p.name for p in Path(temp).iterdir()], ['labels.cache'])

    def test_missing_hooks_names_what_is_absent(self):
        module = types.SimpleNamespace(ThreadPool=object, DATASET_CACHE_VERSION='1',
                                       YOLODataset=type('YOLODataset', (), {}))
        missing = prebuild.missing_hooks(module)
        self.assertIn('save_dataset_cache_file', missing)
        self.assertIn('YOLODataset.cache_labels', missing)
        self.assertIn('YOLODataset.result_to_label', missing)

    def test_same_is_exact(self):
        a = np.array([[0.5, 0.25]], dtype=np.float32)
        self.assertTrue(prebuild.same({'x': [a], 's': (3, 4)}, {'x': [a.copy()], 's': (3, 4)}))
        self.assertFalse(prebuild.same(a, a.astype(np.float64)))
        self.assertFalse(prebuild.same(a, a.reshape(2, 1)))
        self.assertFalse(prebuild.same([a], (a,)))
        self.assertFalse(prebuild.same({'s': (3, 4)}, {'s': [3, 4]}))
        self.assertFalse(prebuild.same(None, a))

    def test_sample_indices_cover_both_ends(self):
        self.assertEqual(prebuild.sample_indices(0, 10), set())
        self.assertEqual(prebuild.sample_indices(5, 0), set())
        self.assertEqual(prebuild.sample_indices(5, 10), {0, 1, 2, 3, 4})
        picked = prebuild.sample_indices(1001, 10)
        self.assertIn(0, picked)
        self.assertIn(1000, picked)
        self.assertLessEqual(len(picked), 12)


if __name__ == '__main__':
    unittest.main()
