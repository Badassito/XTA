"""Standard-library-only preparation tests: no array/GPU imports or probes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT.parent/'Scratch/Experiments/sam-image-batching-20261016/production-qualification/cpu-tests'
spec=importlib.util.spec_from_file_location('sam_batch_benchmark',ROOT/'tools/benchmark_sam_image_batch_integration.py')
benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)


class FakeRenderer:
    def __init__(self,size=4,receipt_change=None,fail_after_sink=False):
        self.size=size;self.receipt_change=receipt_change or {};self.fail_after_sink=fail_after_sink
        self.scalar=[];self.batches=[]
    def render(self,view,frame,inverse,**crop):
        self.scalar.append(frame);return ('pixel',frame)
    def render_compatible_prefix(self,view,frames,inverse,*,consume,**crop):
        selected=tuple(frames[:self.size]);self.batches.append(selected)
        for position,frame in enumerate(selected):
            consume(position,('pixel',frame))
            if self.fail_after_sink:raise RuntimeError('post-sink failure')
        result=dict(completed_images=len(selected),effective_batch_size=len(selected),
            batch_path_used=True,host_result_mode='sink')
        result.update(self.receipt_change);return result


class PreparationTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.root=Path(self.temp.name)
        for name in benchmark.SOURCE_FILES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# source\n')
        functions='\n'.join(f'def {name}():\n    return 1\n' for name in benchmark.NUMERICAL_FUNCTIONS+benchmark.SUPPORT_FUNCTIONS)
        (self.root/'XTA/sam_azimuthal_batch.py').write_text(functions)
        reference=self.root/'qualified.py';reference.write_text(functions)
        renderer_reference=self.root/'renderer-qualified.py';renderer_reference.write_text('def crop_inverse():\n    return 1\n')
        (self.root/'XTA/sam_gpu_rendering.py').write_bytes(renderer_reference.read_bytes())
        fixture=self.root/'fixture.json';fixture.write_text('{}')
        prior=self.root/'prior.json'
        prior.write_text(json.dumps(dict(status='passed',pytest_exit_code=0,changed_during_run=[],oracle_execution_device='cuda:0',
            test_cases=[dict(name=f'{name}[{i}]',skipped=False,failure=False)
                for name,count in benchmark.NUMERICAL_CASE_COUNTS.items() for i in range(count)],source_sha256={
                **benchmark.source_pins(self.root),'prototype':benchmark.sha(reference)})))
        self.config=dict(qualified_operator_source=str(reference),prior_cuda_receipt=str(prior),
            qualified_renderer_source=str(renderer_reference),
            stress_fixture=str(fixture),scratch_bytes=256*1024**2,heatsoak_seconds=45,cycles=3,
            **{name:False for name in benchmark.GATES})
        self.manifest=benchmark.preparation_manifest(self.config,self.root)
        self.config.update({key:self.manifest[key] for key in ('source_sha256','qualified_operator_source_sha256','qualified_renderer_source_sha256',
            'prior_cuda_receipt_sha256','stress_fixture_sha256')})
    def tearDown(self):self.temp.cleanup()

    def test_manifest_fingerprints_itself_and_reuses_exact_numerical_asts(self):
        self.assertEqual(self.manifest['self_sha256'],benchmark.sha(self.root/benchmark.SELF))
        self.assertTrue(self.manifest['reuse_unchanged_48_numerical_cuda_cases'])
        self.assertTrue(all(self.manifest['operator_ast_unchanged'].values()))

    def test_every_fresh_user_and_coordinator_gate_is_required(self):
        for missing in benchmark.GATES:
            config=dict(self.config,**{name:name!=missing for name in benchmark.GATES})
            with self.subTest(missing=missing),self.assertRaisesRegex(benchmark.QualificationFailure,missing):
                benchmark.validate_run(config,self.manifest)

    def test_closed_main_never_enters_runtime(self):
        config_path=self.root/'config.json';config_path.write_text(json.dumps(self.config))
        with mock.patch.object(sys,'argv',['benchmark','--config',str(config_path),'--run']), \
                mock.patch.object(benchmark,'run_qualification',side_effect=AssertionError('GPU runtime entered')) as run, \
                self.assertRaisesRegex(benchmark.QualificationFailure,'gate is closed'):
            benchmark.main()
        run.assert_not_called()

    def test_numerical_operator_change_invalidates_reuse(self):
        path=self.root/'XTA/sam_azimuthal_batch.py'
        path.write_text(path.read_text().replace('return 1','return 2',1))
        current=benchmark.preparation_manifest(self.config,self.root)
        self.assertFalse(current['reuse_unchanged_48_numerical_cuda_cases'])

    def test_lifecycle_change_is_reported_separately_from_numerical_operator_reuse(self):
        path=self.root/'XTA/sam_azimuthal_batch.py'
        path.write_text(path.read_text().replace('def render_compatible_batch():\n    return 1',
            'def render_compatible_batch():\n    return 2'))
        current=benchmark.preparation_manifest(self.config,self.root)
        self.assertFalse(current['operator_ast_unchanged']['render_compatible_batch'])
        self.assertTrue(current['reuse_unchanged_48_numerical_cuda_cases'])

    def test_numerical_dependency_change_invalidates_reuse(self):
        (self.root/'XTA/inference.py').write_text('# changed numerical dependency\n')
        self.assertFalse(benchmark.preparation_manifest(self.config,self.root)['reuse_unchanged_48_numerical_cuda_cases'])

    def test_skipped_prior_cuda_case_cannot_authorize_reuse(self):
        path=Path(self.config['prior_cuda_receipt']);prior=json.loads(path.read_text())
        prior['test_cases'][0]['skipped']=True;path.write_text(json.dumps(prior))
        self.assertFalse(benchmark.preparation_manifest(self.config,self.root)['reuse_unchanged_48_numerical_cuda_cases'])

    def test_cpu_only_prior_receipt_cannot_authorize_numerical_cuda_reuse(self):
        path=Path(self.config['prior_cuda_receipt']);prior=json.loads(path.read_text())
        prior['oracle_execution_device']='cpu';path.write_text(json.dumps(prior))
        self.assertFalse(benchmark.preparation_manifest(self.config,self.root)['reuse_unchanged_48_numerical_cuda_cases'])

    def test_changed_crop_phase_invalidates_numerical_reuse(self):
        (self.root/'XTA/sam_gpu_rendering.py').write_text('def crop_inverse():\n    return 2\n')
        self.assertFalse(benchmark.preparation_manifest(self.config,self.root)['reuse_unchanged_48_numerical_cuda_cases'])

    def test_direct_runtime_call_cannot_bypass_authorization(self):
        with self.assertRaisesRegex(benchmark.QualificationFailure,'gate is closed'):
            benchmark.run_qualification(self.config,self.manifest,self.root/'must-not-create','test')
        self.assertFalse((self.root/'must-not-create').exists())

    def test_changed_benchmark_self_pin_refuses_execution(self):
        config=dict(self.config,**{name:True for name in benchmark.GATES})
        (self.root/benchmark.SELF).write_text('# changed driver\n')
        current=benchmark.preparation_manifest(self.config,self.root)
        with self.assertRaisesRegex(benchmark.QualificationFailure,'source_sha256'):
            benchmark.validate_run(config,current)

    def test_four_frame_production_selection_and_identical_scalar_sink(self):
        frames=(2,3,6,7)
        for size,enabled,expected in ((4,True,[4]),(2,True,[2,2]),(3,True,[3,1]),(4,False,[1,1,1,1])):
            renderer=FakeRenderer(size);writes=[]
            result=benchmark.render_resident_unit(renderer,None,frames,None,{},
                lambda position,image:writes.append((position,image)),enabled=enabled)
            self.assertEqual(result['chunks'],expected);self.assertEqual(result['completed_crops'],4)
            self.assertEqual(writes,[(i,('pixel',frame)) for i,frame in enumerate(frames)])
            benchmark.validate_selected_chunks(result['chunks'],enabled,expected)

    def test_guarded_stress_cannot_silently_measure_largest_fit_b4(self):
        with self.assertRaisesRegex(benchmark.QualificationFailure,'differs'):
            benchmark.validate_selected_chunks([4],True,[2,2])

    def test_small_wide_crop_uses_valid_square_processing_canvas(self):
        view=SimpleNamespace(num_slices=8,src_h=1024,src_w=512)
        shape=benchmark.case_canvas_shape(view,'small')
        self.assertEqual(shape,(8,1024,1024))
        self.assertLessEqual(32+537,shape[2])
        self.assertGreater(32+537,view.src_w)
        self.assertEqual(benchmark.case_canvas_shape(view,'stress'),(8,1024,512))
        for y0,x0,y1,x1 in benchmark.small_case_boxes(view):
            self.assertTrue(0<=y0<y1<=shape[1] and 0<=x0<x1<=shape[2])
            self.assertLessEqual(abs((x0+x1)-shape[2]),1)

    def test_post_sink_failure_does_not_retry_scalar(self):
        renderer=FakeRenderer(2,fail_after_sink=True)
        with self.assertRaisesRegex(RuntimeError,'post-sink'):
            benchmark.render_resident_unit(renderer,None,(2,3,6,7),None,{},lambda *_:None,enabled=True)
        self.assertEqual(renderer.scalar,[]);self.assertEqual(renderer.batches,[(2,3)])

    def test_collector_or_false_batch_receipt_is_rejected(self):
        for change in (dict(host_result_mode='collected'),dict(batch_path_used=False),dict(completed_images=3)):
            with self.subTest(change=change),self.assertRaises(benchmark.QualificationFailure):
                benchmark.render_resident_unit(FakeRenderer(2,change),None,(2,3,6,7),None,{},lambda *_:None,enabled=True)

    def test_no_array_or_gpu_modules_imported_by_preparation(self):
        # A full pytest session may legitimately have these packages loaded.
        # Inspect only this benchmark's preparation in an isolated interpreter,
        # and block an accidental GPU import before its package can initialize.
        script=r'''
import importlib.abc,importlib.util,json,sys
from pathlib import Path
forbidden={'numpy','torch','cupy','XTA'}
attempts=[]
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in forbidden:
            attempts.append(fullname)
            raise ImportError('Forbidden benchmark preparation import: '+fullname)
        return None
sys.meta_path.insert(0,BlockImports())
spec=importlib.util.spec_from_file_location('isolated_batch_benchmark',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.preparation_manifest(json.loads(sys.argv[3]),Path(sys.argv[2]))
loaded=sorted(forbidden.intersection(sys.modules))
assert not attempts and not loaded,(attempts,loaded)
print(json.dumps({'forbidden_import_attempts':attempts,'loaded_forbidden_modules':loaded}))
'''
        completed=subprocess.run([sys.executable,'-B','-c',script,
            str(ROOT/'tools/benchmark_sam_image_batch_integration.py'),str(self.root),json.dumps(self.config)],
            cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        self.assertEqual(json.loads(completed.stdout),dict(forbidden_import_attempts=[],loaded_forbidden_modules=[]))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests))
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (ARTIFACTS/'results.json').write_text(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        imported_array_or_gpu_modules=[name for name in ('numpy','torch','cupy','XTA') if name in sys.modules],
        no_gpu_imports_or_probes=True,benchmark_sha256=benchmark.sha(ROOT/benchmark.SELF)),indent=2)+'\n',encoding='utf-8')
    sys.exit(not result.wasSuccessful())
