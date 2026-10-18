"""Optional image credit is unused by a borrowed native source, whose owner stays live."""
import numpy as np

from XTA import sam_resources as resources
from tests.test_sam_image_prefetch import account, setup
from tests.test_sam_view_image_cache import crop_from


def test_prefetched_borrowed_source_has_no_owned_image_hold_and_survives_shutdown(tmp_path, monkeypatch):
    pixels, context, view, bbox, plans, retirements = setup(tmp_path, monkeypatch)
    path = tmp_path/'native-source.gray8.dat'
    source = np.memmap(path, mode='w+', dtype=np.uint8, shape=pixels.shape)
    source[:] = pixels
    source.flush()
    context.source_volume = source
    pool = account()
    holder = None
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'borrowed-source',
                base_allowance_bytes=4*resources.GIB, headroom_probe=lambda:16*resources.GIB) as parent:
            with context.resource_scope(parent):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                assert holder is not None
                with holder as reference:
                    assert reference.path == path.resolve()
                    assert not context._cache_owners[str(reference.path)]['owned']
                    assert not context._retained_parent_image_credits
                    assert pool._sam_image_staging_pool.in_use == holder.phase_bytes
                    np.testing.assert_array_equal(crop_from(reference, 0, bbox), pixels[0, 2:8, 3:9])
                assert pool._sam_image_staging_pool.in_use == 0
                assert not retirements and path.exists()
        assert resources.sam_parent_promised_bytes(pool) == 0
        context.close()
        assert path.exists() and not source._mmap.closed
    finally:
        if holder is not None:
            holder.close()
        context.close()
        source._mmap.close()
