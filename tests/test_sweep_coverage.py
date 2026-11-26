from concurrent.futures import ThreadPoolExecutor
import gzip
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.reconciliation_io import ReferenceGeometry
from XTA.sweep_coverage import DirectCoverage, plan_direct_coverage


@pytest.mark.parametrize("per_plane", [130, 260])
def test_concurrent_coverage_bands_match_spool(tmp_path, per_plane):
    geometry = ReferenceGeometry((6, 12, 21))
    layers = 260
    extents = [(0, 20, 1, 9, 1, 1)] * per_plane + [(0, 20, 2, 11, 3, 3)] * (layers - per_plane)
    plan = plan_direct_coverage(geometry.shape_tyx, extents)
    assert plan.dtype.itemsize == (1 if per_plane <= 255 else 2)

    crop = np.ones((3, 9), dtype=np.uint8)
    crop[1, 2] = 0
    row_bits = np.empty((crop.shape[0], 3), np.uint8)[:, :2]
    row_bits[:] = np.packbits(crop, axis=1, bitorder="little")
    expected = np.zeros(geometry.shape_tyx, dtype="<u2")
    expected[1, 3:6, 4:13] = crop.astype(expected.dtype) * per_plane
    expected[3, 5:8, 4:13] = crop.astype(expected.dtype) * (layers - per_plane)
    emitted = []
    for mode, band_plan in (("spool", None), ("resident", plan)):
        counter = DirectCoverage(tmp_path, geometry, layers, dense_plan=band_plan)
        try:
            def contribute(index):
                z, y = (1, 3) if index < per_plane else (3, 5)
                if index % 2:
                    counter.observe_packed_plane(z, row_bits, y, 4, crop.shape[1])
                else:
                    counter.observe_plane(z, crop, y, 4)

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(contribute, range(layers)))
            path = tmp_path / f"{mode}.nrrd"
            stats = counter.write(path, sweep_id="test", grid_id="native", cutoff_cent=80)
            emitted.append((stats, path.read_bytes()))
        finally:
            counter.close()
    assert emitted[0] == emitted[1]
    stats, contents = emitted[1]
    assert stats["maximum_count"] == per_plane
    assert stats["voxel_votes"] == int(expected.sum())
    assert gzip.decompress(contents.split(b"\n\n", 1)[1]) == expected.tobytes()


def test_native_coverage_admits_cutoffs_within_one_workspace_budget(monkeypatch):
    from XTA import runtime, sam_resources
    from XTA.sweep_publication import _native_coverage_plans

    geometry = ReferenceGeometry((4, 16, 21))
    role = {"role": "direct"}
    member = SimpleNamespace(info=role, extent=(0, 20, 0, 15, 0, 3))
    cutoff_layers = {80: [member], 90: [member], 100: []}
    bands_bytes = 4 * 16 * 21
    monkeypatch.setattr(sam_resources, "physical_sam_headroom", lambda: 1 << 30)
    writer_bytes = 2 * 128 * 1024**2
    monkeypatch.setattr(runtime, "workspace_anon_cap_bytes", lambda: writer_bytes + bands_bytes)
    plans = _native_coverage_plans(geometry, cutoff_layers, 128, 2)
    assert set(plans) == {80}
    assert plans[80].size_bytes == bands_bytes

    monkeypatch.setattr(runtime, "workspace_anon_cap_bytes", lambda: writer_bytes)
    assert _native_coverage_plans(geometry, cutoff_layers, 128, 2) == {}

    monkeypatch.setattr(runtime, "workspace_anon_cap_bytes", lambda: 0)
    monkeypatch.setattr(sam_resources, "physical_sam_headroom", lambda: 256 << 20)
    assert _native_coverage_plans(geometry, cutoff_layers, 128, 2) == {}
