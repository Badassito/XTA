"""Archive-backed SAM layers preserve source masks and portable evidence."""
from dataclasses import replace
from contextlib import closing
import gc
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry, interpolation, outputs
from XTA.artifact_archive import ArchiveError, artifact_exists, artifact_size, iter_artifacts, publish_directory, reference, split_reference
from XTA.config import GIB, resolve_tilted_view_groups
from XTA.projection_queue import ComponentProjectionQueue
from XTA.reconciliation_io import read_layer_manifest
from XTA.reconciliation_runtime import RuntimeLayer
from XTA.runtime import wait_for_retired_memmap_directory_cleanup, wait_for_retired_memmap_unlinks
from XTA.tta_outputs import measure_bridge_output_survival
from XTA.view_prepare import SamLayerProjectionSubmitter
from tests.test_sam_output_provenance import final_fixture


def views():
    all_views = geometry.get_view_infos(7, 9, 11,
        cartesian_views=('transverse', 'sagittal', 'coronal'),
        tilt_groups=resolve_tilted_view_groups(['transverse:20:vertical']),
        azimuthal_views=('transverse',), azimuthal_azimuth_angles=(60.,),
        radial_views=('transverse',), radial_min_radius=.5, radial_patch_size=8,
        spherical_views=('transverse',), spherical_min_radius=.5, spherical_patch_size=8)
    selected = {}
    for view in all_views:
        selected.setdefault(view.name if view.family == 'orthogonal' else view.family, view)
    return tuple(selected.values())


@pytest.mark.parametrize('view', views(), ids=lambda view: view.name)
@pytest.mark.parametrize('empty', (False, True))
def test_archived_projection_matches_legacy_and_admission_reads_virtual_store(tmp_path, monkeypatch, view, empty):
    for flag in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT',
                 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', 'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT'):
        monkeypatch.setenv(flag, '0')
    native = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    if not empty:
        if view.family in ('azimuthal', 'radial', 'spherical'):
            native[:] = 1
        else:
            native[len(native)//2, native.shape[1]//2, native.shape[2]//2] = 1
    original = tmp_path / 'native.cvol'
    interpolation.write_raw_bbox_mask_store(native, original,
        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    source_shape = (7, 9, 11)
    context = SimpleNamespace(detector_identity='detector', bundle_identity='bundle',
                             source_volume=np.zeros(source_shape, np.uint8))
    kwargs = dict(model_name='detector', view=view, source='fullframe', pass_index=1,
                  sam_context=context, workers=1)
    entry = dict(direction='forward', path=str(original), voxel_count=int(native.sum()), policy_hash='policy')
    virtual = Path(reference(tmp_path / 'sam-artifacts.tar',
        '/'.join(('sam_interpolation', *('long-' + 'a'*170 for _ in range(3)), 'native.cvol'))))
    with mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None), \
         mock.patch.object(assembly, 'final_source_output_shape', return_value=source_shape):
        legacy = assembly.materialize_sam_directional_view_layer(entry, **kwargs)
        with closing(RuntimeLayer(legacy, source_shape)) as reader:
            expected = reader.read_slab(0, source_shape[0]).copy()
        assert bool(expected.any()) is not empty
        publish_directory(original, virtual)
        shutil.rmtree(original)
        queue = ComponentProjectionQueue(workers=1, max_pending=2,
                                         max_source_bytes=GIB, max_working_bytes=16*GIB)
        retained_workspaces = []
        materialize_workspace = interpolation.materialize_raw_bbox_mask_store_workspace

        def retain_workspace(*args, **kwargs):
            workspace = materialize_workspace(*args, **kwargs)
            retained_workspaces.append(workspace)
            return workspace

        try:
            submit = SamLayerProjectionSubmitter(queue, source_shape,
                assembly.materialize_sam_directional_view_layer,
                assembly.materialize_sam_extrapolation_view_layer)
            with mock.patch.object(interpolation, 'materialize_raw_bbox_mask_store_workspace', side_effect=retain_workspace):
                actual = submit(dict(entry, path=str(virtual)), layer_kind='interpolation', **kwargs).result(30)
        finally:
            queue.shutdown()
    workspace_roots = [Path(workspace.filename).parent for workspace in retained_workspaces]
    assert split_reference(actual.path) is not None
    with closing(RuntimeLayer(actual, source_shape)) as reader:
        np.testing.assert_array_equal(reader.read_slab(0, source_shape[0]), expected)
    assert all(artifact_size(member) >= 0 for member in iter_artifacts(actual.path))
    archive, _ = split_reference(actual.path)
    assert not any(member.name.endswith('.dat') for member in iter_artifacts(Path(reference(archive))))
    retained_workspaces.clear()
    gc.collect()
    wait_for_retired_memmap_unlinks(timeout_s=5)
    for root in workspace_roots:
        wait_for_retired_memmap_directory_cleanup(root, timeout_s=5)
        assert not root.exists()
    assert not any(path.name.endswith('.tar#') for path in tmp_path.iterdir())


@pytest.mark.parametrize('nested', (False, True))
def test_archived_evidence_survives_export_relocation_and_final_audit(tmp_path, nested):
    original = tmp_path / 'original'
    original.mkdir()
    layer, final = final_fixture(original)
    archive = original / 'sam-artifacts.tar'
    evidence = Path(reference(archive, 'sam_interpolation/test/proposal_evidence'))
    publish_directory(original / 'evidence', evidence)
    layer = replace(layer, proposal_evidence_path=str(evidence))
    receipt = measure_bridge_output_survival([layer], final)[0]
    assert receipt['connection_survival'] == 'survived'
    export_dir = original / ('low_quality/2x/nrrd' if nested else 'nrrd')
    sink = outputs.NrrdLayerSink(nrrd_dir=export_dir, stem='case',
                                output_shape_tyx=final.shape, max_workers=1)
    try:
        sink.submit_layer(layer, 'selected')
        sink.wait()
        manifest = sink.write_manifest()
    finally:
        sink.shutdown()
    serialized = json.loads(manifest.read_text())['layers'][0]['proposal_evidence_path']
    assert serialized == ('../../../' if nested else '../') + 'sam-artifacts.tar#/sam_interpolation/test/proposal_evidence'
    portable = tmp_path / 'portable'
    shutil.copytree(original, portable)
    shutil.rmtree(original)
    with read_layer_manifest(portable / manifest.relative_to(original), workspace=tmp_path/'read') as layers:
        bundle = layers[0].proposal_bundle()
        assert list(bundle.runs) == ['run']
        assert bundle.candidate_mask('run', 2)[2, 3]


def test_failed_archive_projection_retains_working_files_and_publishes_nothing(tmp_path):
    view = geometry.get_view_infos(7, 9, 11, cartesian_views=('sagittal',))[0]
    original = tmp_path / 'native.cvol'
    interpolation.write_raw_bbox_mask_store(np.ones((9, 7, 11), np.uint8), original,
        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    virtual = Path(reference(tmp_path / 'sam-artifacts.tar', 'native.cvol'))
    publish_directory(original, virtual)
    recovery = []

    def fail_projection(_store, destination, *_args):
        recovery.append(destination.parent)
        (destination.parent / 'diagnostic.bin').write_bytes(b'recoverable')
        raise RuntimeError('injected source projection failure')

    context = SimpleNamespace(detector_identity='detector', bundle_identity='bundle', source_volume=None)
    with mock.patch.object(assembly, '_transpose_sparse_component_store', side_effect=fail_projection):
        with pytest.raises(ArchiveError, match='injected source projection failure'):
            assembly.materialize_sam_directional_view_layer(
                dict(direction='forward', path=str(virtual), voxel_count=693, policy_hash='policy'),
                model_name='detector', view=view, source='fullframe', pass_index=1, sam_context=context)
    assert not artifact_exists(virtual.parent / 'source_layers' / virtual.stem)
    assert (recovery[0] / 'diagnostic.bin').read_bytes() == b'recoverable'
    staging = recovery[0].with_name(recovery[0].name.removesuffix('.work'))
    shutil.rmtree(recovery[0])
    shutil.rmtree(staging)
