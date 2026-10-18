"""Independent budget and gather-iteration admission invariants; CPU only."""
import random
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_sam_azimuthal_batch_oracle import batch, metadata_case, lifecycle, call, torch
from tests.test_sam_azimuthal_native_roi import views_for
from dataclasses import replace
from XTA import geometry


def iterations(rows, block):
    whole, remainder = divmod(rows, block)
    return whole + bool(remainder)



def memory_reference(arguments):
    """Charge independent stage buffers; find the maximal admitted row lane by bisection."""
    size = arguments['batch_size']
    height, width = arguments['native_shape_yx']
    roi_height, roi_width = arguments['roi_shape_yx']
    output_height, output_width = arguments['output_shape_yx']
    logical_t = arguments['logical_t']
    scratch, retained = arguments['scratch_bytes'], arguments['retained_cache_bytes']
    area, pixels = height*width, output_height*output_width
    allowance = 256*area + 96*pixels
    cap = min(scratch, max(0, allowance-retained))
    axes_and_maps = sum((4*width, 20*logical_t, 32*height))
    projection_fixed = sum((axes_and_maps, 256*size*width, size*area, 32*size*roi_height*roi_width))
    warp = sum((axes_and_maps, 6*size*area, 16*size*pixels, 32*pixels))
    host = sum((2*size*pixels, 128*(height+logical_t+width+roi_width), 64*size))
    low, high = 0, min(arguments['scalar_row_block'], arguments['selected_rows'])
    while low < high:
        middle = (low+high+1)//2
        if projection_fixed + 256*size*roi_width*middle <= cap:
            low = middle
        else:
            high = middle-1
    projection = projection_fixed + 256*size*roi_width*low
    fits = warp <= cap and host <= scratch and low > 0
    return dict(eligible=fits, reason='compatible' if fits else 'scratch_or_cache_bound',
        row_block=low, projection_bytes=projection, warp_bytes=warp, host_bytes=host,
        device_cap_bytes=cap, host_cap_bytes=scratch, retained_cache_bytes=retained,
        existing_gpu_allowance_bytes=allowance, total_device_bound_bytes=retained+max(projection,warp),
        cold_cache_growth_bound_bytes=20*logical_t+20*height+8*pixels,
        cold_cache_work_bound_bytes=20*logical_t+32*height+32*pixels)


def check_ledger(arguments):
    reference = memory_reference(arguments)
    actual = batch.batch_workspace_ledger(**arguments)
    for key, value in reference.items():
        if key not in ('eligible', 'reason'):
            assert actual[key] == value, key
    scalar_count = arguments['batch_size'] * iterations(arguments['selected_rows'], arguments['scalar_row_block'])
    batch_count = iterations(arguments['selected_rows'], reference['row_block']) if reference['row_block'] else None
    safe = batch_count is not None and batch_count <= scalar_count
    assert actual['memory_eligible'] == reference['eligible']
    assert actual['batch_gather_iterations'] == batch_count
    assert actual['scalar_gather_iterations'] == scalar_count
    assert actual['gather_iteration_guard_passed'] == safe
    assert actual['eligible'] == (reference['eligible'] and safe)
    if not reference['eligible']:
        assert actual['reason'] == reference['reason'] == 'scratch_or_cache_bound'
    elif not safe:
        assert actual['reason'] == 'gather_iteration_regression'
    else:
        assert actual['reason'] == 'compatible'
    return actual


@pytest.mark.parametrize('size', [1, 2, 3, 4])
def test_loop_guard_never_enlarges_independent_memory_admission(size):
    generator = random.Random(157546 + size)
    for _ in range(350):
        h, w = (generator.randint(1, 8192) for _ in range(2))
        rh, rw = generator.randint(1, h), generator.randint(1, w)
        height, width = generator.randint(1, 2048), generator.randint(1, 2048)
        allowance = 256*h*w + 96*height*width
        check_ledger(dict(batch_size=size, native_shape_yx=(h, w), logical_t=generator.randint(1, 8192),
            roi_shape_yx=(rh, rw), output_shape_yx=(height, width), selected_rows=rh*generator.randint(1, 2),
            scalar_row_block=generator.choice((1, 3, 64, 256, 512, 1024)),
            scratch_bytes=generator.randint(1, 512*1024**2), retained_cache_bytes=generator.randint(0, allowance+1)))


@pytest.mark.parametrize('size', [1, 2, 3, 4])
@pytest.mark.parametrize('rows', [1, 513, 2**60+1])
def test_integer_ceiling_never_uses_float_and_handles_zero_row_budget(size, rows):
    for scratch in (1, 256*1024**2, 2**80):
        check_ledger(dict(batch_size=size, native_shape_yx=(128, 256), logical_t=128,
            roi_shape_yx=(32, 64), output_shape_yx=(16, 16), selected_rows=rows,
            scalar_row_block=512, scratch_bytes=scratch, retained_cache_bytes=0))


def test_guard_exact_equality_then_one_more_iteration():
    # The same q=1 workspace admits two rows but declines three: B2 scalar q3
    # takes two iterations in either case. Other memory gates remain satisfied.
    common = dict(batch_size=2, native_shape_yx=(4, 4), logical_t=4,
        roi_shape_yx=(1, 1), output_shape_yx=(1, 1), scalar_row_block=3,
        scratch_bytes=2920, retained_cache_bytes=0)
    equality = check_ledger(dict(common, selected_rows=2))
    regression = check_ledger(dict(common, selected_rows=3))
    assert equality['row_block'] == regression['row_block'] == 1
    assert equality['eligible'] and equality['batch_gather_iterations'] == 2
    assert regression['memory_eligible'] and not regression['eligible']
    assert regression['batch_gather_iterations'] == 3


def test_measured_stress_prefix_covers_four_frames_exactly_with_eighteen_loops(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_GPU_AZIMUTHAL_STACK_BLOCK', raising=False)
    view = next(view for view in views_for(np.zeros((7, 11, 17), np.uint8), 7)
        if geometry.azimuthal_base_view_name(view) == 'coronal')
    view = replace(view, full_t=2911, full_h=3064, full_w=3022, src_h=2048, src_w=2048,
        center_x=1531.5, center_y=1455., roi_radius=1455.)
    renderer = SimpleNamespace(view=view, source=SimpleNamespace(shape=(64, 3064, 3022)),
        logical_t=2911, scratch_bytes=256*1024**2)
    crop = dict(output_origin_yx=(384, 328), output_height=1100, output_width=1276, retained_cache_bytes=0)
    inverse = np.eye(2, 3, dtype=np.float32)
    frames = (7, 2, 4, 0)
    plans = {n: batch.plan_batch(renderer, view, frames[:n], inverse, **crop) for n in (2, 3, 4)}
    assert [(plans[n].row_block, plans[n].batch_gather_iterations) for n in (2, 3, 4)] == [(256, 9), (120, 19), (52, 43)]
    assert [plans[n].eligible for n in (2, 3, 4)] == [True, False, False]
    assert all(plan.memory_eligible for plan in plans.values())
    offset, output, loops = 0, [], 0
    while offset < len(frames):
        plan = batch.choose_batch_prefix(renderer, view, frames[offset:], inverse, **crop)
        assert plan.eligible and plan.batch_size == 2
        output.extend(frames[offset:offset+plan.batch_size])
        offset += plan.batch_size
        loops += plan.batch_gather_iterations
    assert tuple(output) == frames and loops == 18
    assert 4*plans[2].scalar_gather_iterations_per_image == 20
    assert plans[3].batch_gather_iterations + plans[3].scalar_gather_iterations_per_image == 24


def test_actual_large_canvas_still_selects_memory_eligible_b3():
    renderer, view, matrix, crop = metadata_case()
    after = batch.choose_batch_prefix(renderer, view, (0, 1, 2, 3), matrix, retained_cache_bytes=0, **crop)
    assert after.batch_size == 3
    assert after.eligible and after.gather_iteration_guard_passed


@pytest.mark.parametrize('sink', [False, True])
def test_post_start_iteration_decline_is_prebatch_and_keeps_owner(lifecycle, sink):
    case = lifecycle
    frames = (0, 2, 4, 7)
    cold = batch.plan_batch(case.renderer, case.view, frames, case.inverse, retained_cache_bytes=0, **case.crop)
    assert cold.eligible
    minimum_q = iterations(cold.selected_rows, cold.scalar_gather_iterations)
    target_q = minimum_q - 1
    assert target_q > 0
    denominator = 256*len(frames)*(cold.bbox_yx[3]-cold.bbox_yx[1])
    fixed = cold.projection_bytes-denominator*cold.row_block
    retained = cold.existing_gpu_allowance_bytes-(fixed+denominator*target_q)
    warm = batch.plan_batch(case.renderer, case.view, frames, case.inverse,
        retained_cache_bytes=retained, **case.crop)
    assert warm.memory_eligible and not warm.eligible
    assert warm.reason == 'gather_iteration_regression'
    start = case.renderer._start
    def grow():
        engine = start()
        owner = torch.empty(retained, dtype=torch.uint8)
        engine._fold_cache['post-start-guard-growth'] = (owner[:1], owner[1:2])
        return engine
    case.renderer._start = grow
    seen = []
    with pytest.raises(batch.BatchDeclined) as caught:
        call(case, consume=(lambda position, image: seen.append(position)) if sink else None)
    assert caught.value.reason == 'gather_iteration_regression'
    assert caught.value.receipt['memory_eligible']
    assert caught.value.receipt['retained_cache_bytes'] == retained
    assert not caught.value.receipt['batch_path_used'] and not seen
    assert not any(event in case.state.events for event in ('project', 'warp', 'copy'))
    assert not any(isinstance(event, tuple) and event[0] == 'scalar' for event in case.state.events)
    assert not case.renderer._render_failed
    assert case.renderer.lease is case.owner and case.renderer.engine is case.engine


def test_unmeasured_cache_cannot_admit_even_when_other_guards_fit():
    renderer, view, matrix, crop = metadata_case()
    cold = batch.plan_batch(renderer, view, (0, 1), matrix, retained_cache_bytes=0, **crop)
    assert cold.eligible
    unknown = batch.plan_batch(renderer, view, (0, 1), matrix, **crop)
    assert unknown.reason == 'cache_measurement_required' and not unknown.eligible
    assert unknown.retained_cache_bytes is None
