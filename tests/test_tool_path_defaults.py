"""CLI tools take workspace paths explicitly instead of personal or dataset defaults."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'


class Stop(Exception):
    """Raised by fakes once the behaviour under test has been observed."""


def run_main(monkeypatch, module, argv):
    monkeypatch.setattr(sys, 'argv', [module.__name__, *map(str, argv)])
    return module.main()


def usage_error(capsys, call):
    with pytest.raises(SystemExit) as raised:
        call()
    assert raised.value.code == 2
    return capsys.readouterr().err


def test_tool_sources_name_no_personal_workspace_or_retired_dataset_paths():
    for path in sorted(TOOLS.rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        assert 'ChatGPT' not in text, path
    retired = {
        'benchmark_confidence_capture.py': ('F3_10_4_30_2026', 'Job150790_150798'),
        'qualify_nvcomp_nrrd.py': ('gpu-utilization-148334',),
        'qualify_semantic_trt.py': ('v24-semantic',),
        'prepare_pta_input.py': ('/tmp/pta-smoke',),
    }
    for name, literals in retired.items():
        text = (TOOLS / name).read_text(encoding='utf-8')
        for literal in literals:
            assert literal not in text, (name, literal)


@pytest.mark.parametrize('name', (
    'compare_sam_crop_strategies.py', 'heatsoak_sam_benchmark.py', 'prepare_sam_holdout.py',
    'qualify_sam_feature_cache.py', 'qualify_sam_tiled_integration.py',
    'run_sam_outer_crop_experiment.py'))
def test_sam_research_tools_reserve_the_gpu_through_the_explicit_lock_argument(name):
    tree = ast.parse((TOOLS / name).read_text(encoding='utf-8'))
    locks = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'gpu_lock']
    assert locks
    for call in locks:
        assert ast.unparse(call.args[0]) == 'args.gpu_lock', (name, ast.unparse(call))


def test_diagnostic_gpu_stages_require_lock_and_decoders_default_to_path(monkeypatch, capsys, tmp_path):
    from tools import diagnose_sam_interpolation as tool
    seen = []
    for stage in ('prepare', 'figures'):
        monkeypatch.setattr(tool, stage, lambda args, stage=stage: seen.append((stage, args)))
    for stage in sorted(tool.GPU_LOCK_STAGES):
        error = usage_error(capsys, lambda: run_main(monkeypatch, tool, [stage, '--output', tmp_path]))
        assert f'--gpu-lock is required for the {stage} stage' in error
    run_main(monkeypatch, tool, ['figures', '--output', tmp_path])
    stage, args = seen.pop()
    assert stage == 'figures' and args.gpu_lock is None
    assert (args.ffmpeg, args.ffprobe) == (Path('ffmpeg'), Path('ffprobe'))
    lock, ffmpeg, ffprobe = tmp_path / 'GPU_LOCK', tmp_path / 'bin/ffmpeg.exe', tmp_path / 'bin/ffprobe.exe'
    run_main(monkeypatch, tool, ['prepare', '--output', tmp_path, '--gpu-lock', lock,
                                 '--ffmpeg', ffmpeg, '--ffprobe', ffprobe])
    stage, args = seen.pop()
    assert stage == 'prepare' and (args.gpu_lock, args.ffmpeg, args.ffprobe) == (lock, ffmpeg, ffprobe)


@pytest.mark.parametrize('explicit', (False, True))
def test_outer_crop_preparation_probes_with_path_or_explicit_ffprobe(monkeypatch, tmp_path, explicit):
    from tools import prepare_sam_outer_crop_experiment as tool
    commands = []

    def probe(command, *args, **kwargs):
        commands.append(command)
        raise Stop
    monkeypatch.setattr(tool.subprocess, 'check_output', probe)
    ffprobe = tmp_path / 'bin' / 'ffprobe.exe'
    argv = ['--input', tmp_path / 'source.mkv', '--retained-crop', tmp_path / 'crop.mkv',
            '--output', tmp_path / 'out', *(['--ffprobe', ffprobe] if explicit else [])]
    with pytest.raises(Stop):
        run_main(monkeypatch, tool, argv)
    assert commands[0][0] == (str(ffprobe) if explicit else 'ffprobe')


def test_holdout_requires_lock_and_probes_with_path_ffprobe(monkeypatch, capsys, tmp_path):
    from tools import prepare_sam_holdout as tool
    base = ['--input', tmp_path / 'source.mkv', '--detector', tmp_path / 'detector.pt',
            '--output', tmp_path / 'out']
    assert '--gpu-lock' in usage_error(capsys, lambda: run_main(monkeypatch, tool, base))
    commands = []

    def probe(command, *args, **kwargs):
        commands.append(command)
        raise Stop
    monkeypatch.setattr(tool.subprocess, 'run', probe)
    with pytest.raises(Stop):
        run_main(monkeypatch, tool, [*base, '--gpu-lock', tmp_path / 'GPU_LOCK'])
    assert commands[0][0] == 'ffprobe'


@pytest.mark.parametrize(('module_name', 'argv'), (
    ('heatsoak_sam_benchmark', ['--output', 'out']),
    ('qualify_sam_feature_cache', ['--source-root', 'src', '--inventory', 'inv', '--model', 'm',
                                   '--output', 'out']),
    ('compare_sam_crop_strategies', ['endpoints', '--output', 'out']),
    ('compare_sam_crop_strategies', ['run', '--output', 'out']),
))
def test_single_purpose_gpu_tools_have_no_lock_default(monkeypatch, capsys, module_name, argv):
    tool = importlib.import_module(f'tools.{module_name}')
    assert '--gpu-lock' in usage_error(capsys, lambda: run_main(monkeypatch, tool, argv))


@pytest.mark.parametrize(('module_name', 'locked', 'unlocked'), (
    ('qualify_sam_tiled_integration', 'seam', 'prepare'),
    ('run_sam_outer_crop_experiment', 'infer', 'prepare'),
))
def test_staged_tools_require_lock_only_for_gpu_stage(monkeypatch, capsys, tmp_path, module_name, locked, unlocked):
    tool = importlib.import_module(f'tools.{module_name}')
    seen = []
    for stage in (locked, unlocked):
        monkeypatch.setattr(tool, stage, lambda args, stage=stage: seen.append((stage, args.gpu_lock)))
    error = usage_error(capsys, lambda: run_main(monkeypatch, tool, [locked, '--output', tmp_path]))
    assert f'--gpu-lock is required for the {locked} stage' in error
    run_main(monkeypatch, tool, [unlocked, '--output', tmp_path])
    run_main(monkeypatch, tool, [locked, '--output', tmp_path, '--gpu-lock', tmp_path / 'GPU_LOCK'])
    assert seen == [(unlocked, None), (locked, tmp_path / 'GPU_LOCK')]


def test_nvcomp_benchmarks_require_explicit_dependencies_and_lock(monkeypatch, capsys, tmp_path):
    from tools import qualify_nvcomp_nrrd as tool
    calls = []
    monkeypatch.setattr(tool, 'prepare', lambda *args: calls.append(('prepare', args)) or {})
    monkeypatch.setattr(tool, 'benchmark', lambda *args, **kwargs: calls.append(('benchmark', args, kwargs)) or {})
    with pytest.raises(TypeError):
        tool.GpuLock()
    error = usage_error(capsys, lambda: tool.main(['--benchmark', str(tmp_path / 'samples.json'),
                                                   '--output', str(tmp_path / 'out')]))
    assert 'Benchmark modes require --deps, --runtime, --deflate-deps, --gpu-lock' in error
    tool.main(['--prepare', str(tmp_path / 'source.nrrd'), '--output', str(tmp_path / 'samples')])
    explicit = {flag: tmp_path / flag.strip('-') for flag in ('--deps', '--runtime', '--deflate-deps', '--gpu-lock')}
    tool.main(['--benchmark', str(tmp_path / 'samples.json'), '--output', str(tmp_path / 'out'),
               *[str(item) for pair in explicit.items() for item in pair]])
    assert calls[0][0] == 'prepare'
    kind, args, kwargs = calls[1]
    assert kind == 'benchmark' and args[2:5] == (explicit['--deps'], explicit['--runtime'], explicit['--deflate-deps'])
    assert kwargs == {'gpu_lock': explicit['--gpu-lock']}


@pytest.fixture
def capture_probe(tmp_path, monkeypatch):
    from tools import benchmark_confidence_capture as probe
    monkeypatch.setattr(probe, 'SCRATCH', tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setattr(probe, 'verify_archive', lambda path, sha: Path(path))
    monkeypatch.setattr(probe, 'digest_file', lambda path: probe.ARCHIVE_SHA256)
    return probe


def test_confidence_capture_requires_output_and_runs_real_case_only_when_named(capture_probe, capsys, tmp_path):
    probe = capture_probe
    assert '--output-dir is required' in usage_error(capsys, lambda: probe.parse_args(['--no-real']))
    assert probe.parse_args(['--child-spec', str(tmp_path / 'spec.json')]).output_dir is None
    assert probe.main(['--output-dir', str(tmp_path / 'synthetic')]) == 0
    receipt = json.loads((tmp_path / 'synthetic/benchmark.json').read_text())
    assert receipt['status'] == 'planned' and receipt['real_unavailable'] == 'no --real-manifest supplied'
    assert all(case['kind'] == 'synthetic' for case in receipt['cases'])
    missing = tmp_path / 'retained' / 'manifest.json'
    assert probe.main(['--output-dir', str(tmp_path / 'named'), '--real-manifest', str(missing)]) == 0
    assert json.loads((tmp_path / 'named/benchmark.json').read_text())['real_unavailable'] == str(missing)


def test_semantic_qualification_requires_explicit_output_and_input(monkeypatch, capsys, tmp_path):
    from tools import qualify_semantic_trt as tool
    engine = tmp_path / 'semantic.engine'
    engine.write_bytes(b'engine')
    base = ['--engine', f'512:1:{engine}', '--frames', '2']
    error = usage_error(capsys, lambda: tool.main([*base, '--skip-pipeline']))
    assert '--output is required' in error
    monkeypatch.setattr(tool, 'SCRATCH', tmp_path / 'scratch')
    error = usage_error(capsys, lambda: tool.main([*base, '--skip-pipeline', '--output', str(tmp_path / 'elsewhere')]))
    assert 'Use a new --output beneath' in error
    error = usage_error(capsys, lambda: tool.main([*base, '--output', str(tmp_path / 'scratch/run')]))
    assert 'requires --input' in error
    assert not (tmp_path / 'scratch').exists()


def test_nrrd_fixture_output_stays_in_sibling_scratch(monkeypatch, tmp_path):
    from tools import nrrd_output_fixture as fixture
    assert fixture.SCRATCH == ROOT.parent / 'Scratch'
    source = tmp_path / 'case.seg.nrrd'
    source.write_bytes(b'NRRD0004\n\n')
    (tmp_path / 'scratch').mkdir()
    monkeypatch.setattr(fixture, 'SCRATCH', tmp_path / 'scratch')
    with pytest.raises(ValueError, match='under Scratch'):
        fixture.convert(source, tmp_path / 'elsewhere')
    assert not (tmp_path / 'elsewhere').exists()


FAKE_PTA = '''import json, sys
from pathlib import Path
def option(flag):return Path(sys.argv[sys.argv.index(flag)+1])
source=option('--input')
(Path(sys.argv[0]).parent/'seen.json').write_text(json.dumps({'input':str(source),'files':sorted(p.name for p in source.iterdir())}))
if '--fail' in Path(sys.argv[0]).read_text().splitlines()[-1]:raise SystemExit(3)
output=option('--output');output.mkdir(parents=True);(output/'manifest.json').write_text('{}')
'''


@pytest.mark.parametrize('fails', (False, True))
def test_pta_smoke_copies_inputs_into_a_fresh_scratch_directory(tmp_path, fails):
    from types import SimpleNamespace
    from tools import prepare_pta_input as prep
    video = tmp_path / 'source.mkv'
    video.write_bytes(b'video')
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'sample.nrrd').write_bytes(b'labels')
    prep.save_json(prepared / 'preparation.json', {'identity': {'stem': 'sample', 'intensity': {'path': str(video)}}})
    script = tmp_path / 'tools' / 'pta.py'
    script.parent.mkdir()
    script.write_text(FAKE_PTA + ('# --fail\n' if fails else '# pass\n'), encoding='utf-8')
    scratch_root = tmp_path / 'local-scratch'
    log_dir = tmp_path / 'log'
    log_dir.mkdir()
    args = SimpleNamespace(smoke_script=str(script), scratch_root=str(scratch_root), python=sys.executable,
                           smoke_imgsz=64, smoke_fraction=.5)
    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            prep.smoke(args, [str(prepared)], log_dir)
    else:
        prep.smoke(args, [str(prepared)], log_dir)
    seen = json.loads((script.parent / 'seen.json').read_text())
    used = Path(seen['input'])
    assert used.parent == scratch_root.resolve() and used.name.startswith('pta-smoke-')
    assert seen['files'] == ['sample.mkv', 'sample.nrrd']
    command = prep.read_json(log_dir / 'smoke-command.json')['command']
    assert command[command.index('--input') + 1] == str(used)
    assert used.exists() == fails
