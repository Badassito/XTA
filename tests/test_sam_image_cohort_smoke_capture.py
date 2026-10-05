"""CPU real-writer smoke capture preserves nullable parent and child scores."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.geometry import get_view_infos
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_extrapolation import extrapolate_sam_view_volume_pass
from XTA.sam_tracker_runtime import SamTrackerRunResult
from tools.smoke_sam_image_cohorts import capture, eligible


class Tracker:
    device_ids = (0,)

    def run(self, **request):
        masks = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            mask = np.zeros(request['seed_mask'].shape, bool)
            mask[0, 0] = True
            if frame == request['seed_frame']:
                mask = request['seed_mask'].copy()
            masks[frame] = mask
        model = dict(checkpoint_sha256='synthetic', model_version='synthetic', sdk_output_hole_fill_area=0)
        return SamTrackerRunResult(masks, {frame: .25 + frame / 100 for frame in masks},
            {frame: 'observed' for frame in masks}, dict(run_id=request['run_id'],
                coverage_complete=True, sam_model=model, sam_runtime={'sdk': 'synthetic'},
                adapter_receipt=dict(seed_roundtrip_exact=True, seed_roundtrip_passed=True,
                                     raw_observation_complete=True),
                image_cache_lifetime=dict(gray_mapping_retired_after_render=True)))


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_real_writer_capture_keeps_parent_score_semantics_and_exact_child_scores(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    baseline = np.zeros((23, 48, 64), np.uint8)
    baseline[4:19, 15:24, 20:31] = 1
    view = get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    _, stats, components = extrapolate_sam_view_volume_pass(baseline, view=view,
        work_dir=tmp_path / 'core', runtime=Tracker(), distance=4, walk_back=0,
        min_radius=3., crop_mode=mode, eligible_terminals=eligible)
    bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
    protocol = dict(checkpoint_sha256='synthetic', source_shape_tyx=list(baseline.shape))
    records, exports = capture(bundle, baseline, protocol, tmp_path,
        SimpleNamespace(bundle_identity='synthetic', source_volume=baseline), view, components)
    assert set(records) == {'4_backward', '18_forward'}
    assert set(exports) == {'forward', 'backward'}
    for record in records.values():
        if mode == 'tiled':
            assert record['parent_scores'] is None
            assert len(record['tile_scores']) == 1
            scores = next(iter(record['tile_scores'].values()))
            assert set(record['tile_raw_hashes']) == set(record['tile_scores'])
        else:
            assert record['tile_scores'] == {}
            scores = record['parent_scores']
        assert scores == {str(frame): .25 + frame / 100 for frame in record['expected_frames']}
