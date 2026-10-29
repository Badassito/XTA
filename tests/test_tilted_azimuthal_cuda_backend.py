"""Retired scatter interfaces refuse work before input conversion or GPU access."""
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from XTA import tilted_azimuthal_projection as plans
from XTA import tilted_azimuthal_projection_cuda as backend


class ForbiddenInput:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Retired backend converted its input')

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise AssertionError(f'Retired backend inspected {name}')


@pytest.mark.parametrize('source,plan', [(None, None), (object(), object()),
                                        (ForbiddenInput(), ForbiddenInput())])
def test_retired_interfaces_refuse_without_inspecting_inputs_or_importing_cuda(source, plan):
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'cupy'):
            raise AssertionError('Retired backend imported CUDA runtime')
        return original_import(name, *args, **kwargs)

    with mock.patch('builtins.__import__', side_effect=guarded_import):
        available, reason = plans.tilted_azimuthal_cuda_capability()
        assert not available and 'native destination' in reason
        with pytest.raises(plans.TiltedAzimuthalPlanUnavailable, match='native destination'):
            plans.build_tilted_azimuthal_plan(source, plan, ForbiddenInput(),
                known_row_occupancy=ForbiddenInput(), known_slice_bboxes=ForbiddenInput(),
                max_plan_bytes=0)
        with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnavailable, match='native destination'):
            backend.TiltedAzimuthalCudaProjector(source, plan, device_index=-1,
                initial_packed=ForbiddenInput(), first_frame=-1,
                block_bytes=0, upload_bytes=0, reserve_bytes=0)


def test_retired_benchmark_exits_with_guidance_without_creating_output(tmp_path):
    script = Path(__file__).resolve().parents[1]/'tools'/'benchmark_tilted_azimuthal_projection.py'
    output = tmp_path/'unused'
    result = subprocess.run([sys.executable, '-B', str(script), '--output', str(output),
                             '--size', '256'], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'retired' in result.stderr
    assert 'benchmark_native_pull_projection.py --help' in result.stderr
    assert 'CUDA performance requires separate qualification' in result.stderr
    assert not output.exists()
