from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from XTA.lta_telemetry import LtaExecutionTrace, lta_source_fingerprint


class LtaTelemetryTests(unittest.TestCase):
    def test_phases_preserve_failures_and_carry_mutable_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = LtaExecutionTrace(temporary, 'coordinator')
            with trace.phase('reduce', work_id='window-1') as fields:
                fields['stored_bytes'] = 128
            with self.assertRaisesRegex(ValueError, 'bad shape'):
                with trace.phase('prepare'):
                    raise ValueError('bad shape')
            trace.close()
            records = [json.loads(line) for line in trace.path.read_text().splitlines()]
            self.assertEqual([record['event'] for record in records], ['phase_start', 'phase_end'] * 2)
            self.assertEqual(records[1]['stored_bytes'], 128)
            self.assertEqual(records[3]['status'], 'failed')
            self.assertEqual(records[3]['error_type'], 'ValueError')
            self.assertGreaterEqual(records[1]['wall_seconds'], 0)
            from tools.lta_trace_summary import summarize
            summary = summarize(Path(temporary))
            stream = summary['streams'][0]
            self.assertEqual(stream['phases']['reduce']['calls'], 1)
            self.assertEqual(stream['phases']['prepare']['failures'], 1)
            self.assertEqual(stream['unfinished_phases'], [])

    def test_start_is_readable_while_a_long_phase_is_still_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = LtaExecutionTrace(temporary, 'worker')
            with trace.phase('waiting_for_seed'):
                from tools.lta_trace_summary import summarize
                pending = summarize(Path(temporary))['streams'][0]['unfinished_phases']
                self.assertEqual([item['phase'] for item in pending], ['waiting_for_seed'])
            trace.close()

    def test_disabled_trace_has_no_output_and_source_fingerprint_is_stable(self):
        trace = LtaExecutionTrace(None, 'disabled')
        with trace.phase('work'):
            trace.event('done')
        trace.close()
        self.assertIsNone(trace.path)
        first = lta_source_fingerprint()
        self.assertEqual(first, lta_source_fingerprint())
        self.assertIn('lta_worker_adapter.py', first['files'])
        self.assertIn('media.py', first['files'])
        self.assertIn('sam_hole_fill.py', first['files'])
        self.assertIn('unification/tta_manifest.py', first['files'])
        self.assertEqual(first['scope'], 'xta.package_python_sources/1')
        self.assertEqual(len(first['sha256']), 64)

    def test_source_fingerprint_covers_shared_sam_and_new_nested_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'lta_worker_adapter.py').write_text('worker = 1\n', encoding='utf-8')
            (root / 'sam_hole_fill.py').write_text('fill = 1\n', encoding='utf-8')
            shared = root / 'unification' / 'sampling.py'
            shared.parent.mkdir()
            shared.write_text('sample = 1\n', encoding='utf-8')
            original = lta_source_fingerprint(package_root=root)
            self.assertEqual(set(original['files']), {
                'lta_worker_adapter.py', 'sam_hole_fill.py', 'unification/sampling.py',
            })
            for path in (root / 'sam_hole_fill.py', shared):
                with self.subTest(path=path.relative_to(root)):
                    previous = lta_source_fingerprint(package_root=root)
                    path.write_text(path.read_text(encoding='utf-8') + '# changed\n', encoding='utf-8')
                    current = lta_source_fingerprint(package_root=root)
                    self.assertNotEqual(previous['sha256'], current['sha256'])
                    self.assertEqual(current, lta_source_fingerprint(package_root=root))

            # Adding a future dependency requires no update to an inventory list.
            (root / 'unification' / 'new_helper.py').write_text('new = 1\n', encoding='utf-8')
            added = lta_source_fingerprint(package_root=root)
            self.assertNotEqual(current['sha256'], added['sha256'])
            self.assertIn('unification/new_helper.py', added['files'])

            # Non-source runtime artifacts do not identify a different source.
            (root / 'preview.png').write_bytes(b'preview')
            (root / '__pycache__').mkdir()
            (root / '__pycache__' / 'cached.pyc').write_bytes(b'bytecode')
            self.assertEqual(added, lta_source_fingerprint(package_root=root))

    def test_source_fingerprint_distinguishes_nested_paths_with_same_basename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ('first', 'second'):
                (root / folder).mkdir()
                (root / folder / 'helper.py').write_text('value = 1\n', encoding='utf-8')
            original = lta_source_fingerprint(package_root=root)
            self.assertEqual(set(original['files']), {'first/helper.py', 'second/helper.py'})
            (root / 'first' / 'helper.py').rename(root / 'first' / 'renamed.py')
            renamed = lta_source_fingerprint(package_root=root)
            self.assertNotEqual(original['sha256'], renamed['sha256'])
            self.assertEqual(original['files']['first/helper.py'], renamed['files']['first/renamed.py'])


if __name__ == '__main__':
    unittest.main()
