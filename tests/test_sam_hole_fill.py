from types import SimpleNamespace

import numpy as np
import pytest

from XTA.sam_hole_fill import configure_sam_hole_fill, fill_sam_mask_logits_inplace


def _donut():
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:8] = True
    mask[3:6, 3:6] = False
    return mask


def test_logits_fill_is_per_object_and_retains_open_background_and_scores():
    torch = pytest.importorskip('torch')
    masks = np.stack((_donut(), _donut()))
    masks[1, :5, 4] = False  # This object's cavity connects to the image edge.
    logits = torch.tensor(np.where(masks[:, None], 2.5, -3.5))
    original = logits.clone()
    fill_sam_mask_logits_inplace(logits)
    assert bool((logits[0, 0, 3:6, 3:6] > 0).all())
    assert bool((logits[1] == original[1]).all())
    assert bool((logits[original > 0] == original[original > 0]).all())
    assert logits[0, 0, 0, 0] == original[0, 0, 0, 0]


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_prediction_is_filled_before_memory_encoding_and_next_prediction(wrapped, enabled):
    torch = pytest.importorskip('torch')

    class Tracker:
        fill_hole_area = 0

        def _encode_new_memory(self, *, pred_masks_high_res):
            self.memory = pred_masks_high_res.clone()
            return self.memory

        def _get_orig_video_res_output(self, any_res_masks):
            return any_res_masks, any_res_masks

    model = Tracker()
    tracker = SimpleNamespace(model=model, fill_hole_area=0) if wrapped else model
    predictor = SimpleNamespace(model=SimpleNamespace(tracker=tracker))
    receipt = configure_sam_hole_fill(predictor, enabled=enabled)
    logits = torch.tensor(np.where(_donut()[None, None], 2.0, -2.0))
    model._encode_new_memory(pred_masks_high_res=logits)
    # The next prediction's recurrent memory already sees the filled cavity.
    assert bool(model.memory[0, 0, 4, 4] > 0) is enabled
    output = torch.tensor(np.where(_donut()[None, None], 2.0, -2.0))
    _, output = model._get_orig_video_res_output(output)
    assert bool(output[0, 0, 4, 4] > 0) is enabled
    assert receipt['memory_hook'] is enabled


@pytest.mark.parametrize('enabled', [False, True])
def test_lta_seeds_predictions_and_dogfood_preserve_holes_unless_enabled(enabled):
    from XTA.lta_propagation import LtaMaskSeed, LtaPropagationRequest, run_mask_injected_session
    from XTA.lta_sam import SamFramePrediction, SamSessionPlan
    from XTA.lta_tile_tracking import LtaLineageId

    mask = _donut()
    seed = LtaMaskSeed(lineage=LtaLineageId('volume', 'physical', 'runtime', 'tiles', 'lineage'),
                       frame_index=0, object_id=0, mask=mask)
    request = LtaPropagationRequest(work_id='holes', session=SamSessionPlan('sequence', 0, 0, 2),
        prompt_frame=0, direction='forward', seeds=(seed,), conf=0.15)

    def adapter(_measured, _raw, **kwargs):
        assert bool(kwargs['object_masks'][0][4, 4]) is enabled
        return dict(seed_roundtrip_policy='overlap-aware', seed_roundtrip_passed=True,
            propagation=(SamFramePrediction('sequence', 0, 1, 0, 1.0, mask, 0.9),))

    result = run_mask_injected_session(object(), object(), resource=[object(), object()],
        request=request, adapter=adapter, enable_2D_hole_fill=enabled)
    assert all(bool(item.prediction.binary_mask[4, 4]) is enabled for item in result.predictions)
    assert bool(result.dogfood_seeds[0].mask[4, 4]) is enabled
    assert bool(mask[4, 4]) is False


@pytest.mark.parametrize('enabled', [False, True])
def test_lta_completed_view_and_merged_relays_apply_the_same_opt_in(enabled):
    from XTA.lta_postprocessing import (fill_completed_view_holes_2d_inplace,
                                       fill_merged_seed_mask_holes_2d)
    mask = _donut()
    merged = fill_merged_seed_mask_holes_2d(mask, enabled=enabled)
    assert bool(merged.mask[4, 4]) is enabled
    volume = mask[None].astype(np.uint8)
    receipt = fill_completed_view_holes_2d_inplace(volume, runtime_view_id='view', enabled=enabled)
    assert bool(volume[0, 4, 4]) is enabled
    assert receipt.added_pixels == (9 if enabled else 0)
    assert merged.receipt.algorithm == receipt.algorithm
