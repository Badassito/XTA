"""Frozen extrapolation roles reuse bounded immutable rasters with exact evidence."""
from collections import Counter
import gc
import weakref

import numpy as np
import pytest

from XTA import sam_extrapolation as extrapolation, sam_evidence as evidence
from XTA.sam_extrapolation_planning import plan_sam_extrapolation


class Capture:
    def add_group(self, metadata, masks):
        self.metadata, self.masks = metadata, masks


def planned(*, cyclic=False, unrelated=False, shape=(8, 96, 128)):
    source = np.zeros(shape, np.uint8)
    if cyclic:
        source = np.zeros((10, 96, 128), np.uint8)
        source[8:10, 40:47, 20:27] = 1
        source[0:2, 40:47, 101:108] = 1
        eligible = lambda terminal, direction: terminal.native_frame_index == 1 and direction == 1
    else:
        source[2:4, 40:47, 50:57] = 1
        if unrelated:
            source[5, 40:43, 65:68] = 1
        eligible = lambda terminal, direction: terminal.frame_index == 3 and direction == 1
    plan = plan_sam_extrapolation(source, extrapolation_distance=3,
        extrapolation_walk_back=1, extrapolation_min_radius=0., wrap_axis=cyclic,
        eligible_terminals=eligible)
    group = next(group for group in plan.groups if group.status == 'planned')
    return source, plan, group


def original_plane(source, group, frame):
    if group.frame_addressing:
        address = group.frame_addresses[frame]
        plane = source[address['native_index']]
        if address['mirror_u']:
            plane = plane[:, ::-1]
    else:
        plane = source[frame]
    y0, x0, y1, x1 = group.context_bbox_yx
    return plane[y0:y1, x0:x1] != 0


def expected_roles(source, plan, group):
    shape = (group.context_bbox_yx[2]-group.context_bbox_yx[0], group.context_bbox_yx[3]-group.context_bbox_yx[1])
    masks = {}
    for oid in group.observation_ids:
        observation = plan.by_id[oid]
        endpoint = observation.mask_in_crop(group.context_bbox_yx)
        masks['endpoint:'+oid] = endpoint
        masks['evaluation:'+oid] = np.ones(shape, bool)
        masks['permitted:'+oid] = original_plane(source, group, observation.frame_index) & ~endpoint
    for frame in group.frame_indices:
        baseline = original_plane(source, group, frame)
        masks[f'acceptance:{frame}'] = np.ones(shape, bool)
        masks[f'known_foreground:{frame}'] = baseline
        masks[f'unrelated:{frame}'] = baseline
        masks[f'write:{frame}'] = ~baseline if frame in group.output_frames else np.zeros(shape, bool)
    return masks


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('unrelated', [False, True])
def test_cached_roles_match_frozen_originals_and_keep_serialized_bytes(tmp_path, cyclic, unrelated):
    source, plan, group = planned(cyclic=cyclic, unrelated=unrelated)
    expected = expected_roles(source, plan, group)
    capture = Capture()
    extrapolation.write_extrapolation_group(capture, group, plan)
    assert set(capture.masks) == set(expected)
    for key in list(expected) + list(reversed(expected)):
        np.testing.assert_array_equal(capture.masks[key], expected[key], err_msg=key)
    scope = dict(shape_tyx=list(plan.native_shape_tyx))
    if cyclic:
        scope.update(frame_addressing=plan.frame_addressing, evidence_purpose='sam_extrapolation')
    with evidence.SamEvidenceWriter(tmp_path/'expected', scope) as writer:
        writer.add_group(capture.metadata, expected)
        reference = writer.commit()
    with evidence.SamEvidenceWriter(tmp_path/'actual', scope) as writer:
        extrapolation.write_extrapolation_group(writer, group, plan)
        actual = writer.commit()
    for name in ('index.json', 'masks.bin', 'manifest.json'):
        assert (actual.directory/name).read_bytes() == (reference.directory/name).read_bytes()


def test_empty_frames_are_reconstructed_once_across_role_passes_and_aliases_are_safe(monkeypatch):
    source, plan, group = planned()
    calls = Counter()
    frozen = extrapolation._frozen_plane
    def counted(plan, frame, crop):
        calls[frame] += 1
        return frozen(plan, frame, crop)
    monkeypatch.setattr(extrapolation, '_frozen_plane', counted)
    capture = Capture()
    extrapolation.write_extrapolation_group(capture, group, plan)
    empty = [frame for frame in group.output_frames if not original_plane(source, group, frame).any()]
    assert empty
    retained = capture.masks[f'known_foreground:{empty[0]}']
    for frame in empty + list(reversed(empty)):
        assert capture.masks[f'known_foreground:{frame}'] is retained
        assert capture.masks[f'unrelated:{frame}'] is retained
        assert capture.masks[f'write:{frame}'].all()
    assert all(calls[frame] == 1 for frame in empty)
    nonempty = [frame for frame in group.frame_indices if original_plane(source, group, frame).any()]
    assert len(nonempty) >= 2
    previous = capture.masks[f'known_foreground:{nonempty[0]}']
    owner = weakref.ref(previous)
    np.testing.assert_array_equal(previous, original_plane(source, group, nonempty[0]))
    del previous
    capture.masks[f'known_foreground:{nonempty[1]}']
    gc.collect()
    assert owner() is None
    assert not retained.any()
    with pytest.raises(ValueError):
        retained.flags.writeable = True


def test_real_writer_role_order_reuses_constant_packets_without_changing_outputs(tmp_path, monkeypatch):
    source, plan, group = planned(shape=(8, 512, 768))
    expected = expected_roles(source, plan, group)
    original_encode = evidence._encode_mask
    counts = {}
    capture = Capture()
    extrapolation.write_extrapolation_group(capture, group, plan)
    for name, masks in (('fresh-roles', expected), ('shared-roles', capture.masks)):
        calls = []
        def encode(value):
            calls.append(int(np.count_nonzero(value)))
            return original_encode(value)
        monkeypatch.setattr(evidence, '_encode_mask', encode)
        with evidence.SamEvidenceWriter(tmp_path/name, {'shape_tyx': list(source.shape)}) as writer:
            writer.add_group(capture.metadata, masks)
            writer.commit()
        counts[name] = len(calls)
    assert counts['shared-roles'] < counts['fresh-roles'] // 2
    for name in ('index.json', 'masks.bin', 'manifest.json'):
        assert (tmp_path/'fresh-roles'/name).read_bytes() == (tmp_path/'shared-roles'/name).read_bytes()
