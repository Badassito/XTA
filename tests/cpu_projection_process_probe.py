"""Real CPU projection entry points in a fresh interpreter with no TTA owner."""
from pathlib import Path
import sys

import numpy as np

from XTA import assembly, backprojection, geometry, native_scheduler, outputs, runtime
from tests.reference_backends.native_pull import iter_destination_samples


def main():
    root, mode, route = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    root.mkdir(parents=True, exist_ok=True)
    if mode == 'isolated':
        runtime._interpolation_process_initializer()
        assert runtime.interpolation_process_worker_active()
    else:
        assert not runtime.interpolation_process_worker_active()
    coordinator = backprojection._MAIN_PROCESS_GPU_STAGE_COORDINATOR
    assert coordinator._owner is None and not coordinator._identities
    assert native_scheduler._native_module is None

    target = (5, 7, 9)
    view = geometry._build_tilted_view_infos(*target, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]
    source = np.zeros((view.num_slices, 7, 9), np.uint8)
    source[:, 1:6, 2:7] = 1
    source[2, 3, 4] = 0
    expected = np.zeros(target, np.uint8)
    for dest, frame, row, column in iter_destination_samples(view, source.shape, target):
        expected.reshape(-1)[dest[source[frame, row, column] != 0]] = 1
    assert expected.any()
    if route == 'dense':
        actual = backprojection.backproject_tilted_volume_to_volume(
            source, view, root / 'projected.u8', 'fresh CPU tilted projection',
            prefer_memory=False, reserve_bytes=0, workers=1, out_shape_tyx=target)
        try:
            np.testing.assert_array_equal(actual, expected)
        finally:
            runtime.close_memmap_array(actual)
    else:
        assembly.set_final_source_output_shape(target)
        ref = assembly.materialize_nrrd_view_layer(
            source, model_name='cpu-model', view=view, source='fullframe', mask_kind='bridge',
            temp_dir=root, workers=1, known_has_foreground=True,
            submit_to_sink=False, force_path_backed_store=True)
        assert ref is not None
        opened = outputs._open_nrrd_layer_ref(ref)
        try:
            actual = np.stack([outputs._read_layer_slice_in_output_shape(opened, target, z)
                               for z in range(target[0])])
            np.testing.assert_array_equal(actual, expected)
        finally:
            outputs._close_nrrd_layer_source(opened)
            outputs._drop_nrrd_raw_store_chunks_ram_cache(opened)
            assembly.set_final_source_output_shape(None)
    assert coordinator._owner is None and not coordinator._identities
    assert native_scheduler._native_module is None
    print(f'PASS {mode} {route}: exact CPU output; no native scheduler/request created')


if __name__ == '__main__':
    main()
