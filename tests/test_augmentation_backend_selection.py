"""Backend-grouped policy selection stays independent of numerical runtimes."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from XTA.augmentation_policy import resolve_augmentation_definitions
from XTA.tta_augmentation_config import resolve_tta_augmentation

POLICIES = Path(__file__).resolve().parents[1] / 'XTA/examples/external_augmentations'
CPU = POLICIES / 'CPU_light.py'
GPU = POLICIES / 'GPU_light.py'


class AugmentationBackendSelectionTests(unittest.TestCase):
    def resolve(self, entries, *, gpu=False, cpu=False, ratio=3):
        return resolve_tta_augmentation(SimpleNamespace(augmentation=entries, augmentation_ratio=ratio),
                                        gpu_devices=('cuda:0',) if gpu else (), cpu_enabled=cpu)

    def test_each_inference_backend_selects_its_policy(self):
        cpu = self.resolve([f'cpu:{CPU}'], cpu=True)
        gpu = self.resolve([f'gpu:{GPU}'], gpu=True)
        self.assertEqual(cpu.for_backend('cpu').path, str(CPU.resolve()))
        self.assertEqual(cpu.for_backend('cpu').export_name, 'build_augmentation')
        self.assertEqual(gpu.for_backend('gpu').path, str(GPU.resolve()))
        cpu.assert_unchanged()
        gpu.assert_unchanged()

    def test_hybrid_requires_and_records_both_backends(self):
        for entries, missing in (([f'cpu:{CPU}'], 'gpu'), ([f'gpu:{GPU}'], 'cpu')):
            with self.assertRaisesRegex(ValueError, f'requires a {missing}:'):
                self.resolve(entries, cpu=True, gpu=True)
        settings = self.resolve([f'gpu:{GPU}', f'cpu:{CPU}'], cpu=True, gpu=True)
        inverse = self.resolve([f'cpu:{CPU}', f'gpu:{GPU}'], cpu=True, gpu=True)
        self.assertEqual(settings, inverse)
        self.assertEqual(set(settings.record()['policies']), {'cpu', 'gpu'})
        self.assertEqual(settings.record()['schema'], 'xta.tta.external-augmentation/2')
        self.assertEqual(settings.for_backend('cpu').content_sha256, settings.cpu_sha256)
        self.assertEqual(settings.for_backend('gpu').content_sha256, settings.gpu_sha256)
        settings.assert_unchanged()

    def test_missing_active_entry_and_duplicate_tags_fail(self):
        with self.assertRaisesRegex(ValueError, 'requires a cpu:'):
            self.resolve([f'gpu:{GPU}'], cpu=True)
        with self.assertRaisesRegex(ValueError, 'requires a gpu:'):
            self.resolve([f'cpu:{CPU}'], gpu=True)
        for entries in ([f'cpu:{CPU}', f'cpu:{CPU}'], [f'gpu:{GPU}', f'gpu:{GPU}']):
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                resolve_augmentation_definitions(entries)

    def test_tag_requires_appropriate_factory(self):
        for value in (f'cpu:{GPU}', f'gpu:{CPU}'):
            with self.assertRaisesRegex(ValueError, 'policy export'):
                resolve_augmentation_definitions([value])

    def test_windows_drive_and_spaces_are_preserved(self):
        with patch('XTA.augmentation_policy.inspect_augmentation_definition') as inspect:
            inspect.return_value = SimpleNamespace(export_name='build_augmentation')
            resolve_augmentation_definitions([r'cpu:C:\data files\policy.py'])
            inspect.assert_called_once_with(r'C:\data files\policy.py')

    def test_parser_accepts_grouped_and_repeated_groups(self):
        from XTA.config import build_argparser
        parser = build_argparser()
        for tail in ([f'cpu:{CPU}', f'gpu:{GPU}'], [f'cpu:{CPU}', '--augmentation', f'gpu:{GPU}']):
            parsed = parser.parse_args(['--input', 'source.mkv', '--model', 'cpu:model.xml',
                                        '--augmentation', *tail])
            self.assertEqual(parsed.augmentation, [f'cpu:{CPU}', f'gpu:{GPU}'])

    def test_pta_uses_selected_group(self):
        from XTA.pta_config import parse_pta_args
        for backend, path in (('cpu', CPU), ('gpu', GPU)):
            resolved = parse_pta_args(['--input', 'input', '--enable_cartesian', 'transverse',
                '--augmentation', f'{backend}:{path}', '--augmentation_execution', 'offline'])
            self.assertEqual(resolved.args.augmentation, str(path.resolve()))
            self.assertEqual(resolved.args.offline_augmentation_backend, backend)


if __name__ == '__main__':
    unittest.main()
