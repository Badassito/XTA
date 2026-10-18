"""A private failed image remains funded until its actual file is retired."""
from pathlib import Path
from unittest import mock
import gc

import pytest

from XTA import runtime, sam_resources as resources
from tests.test_sam_mandatory_image_lifetime import case
from tests.test_sam_view_image_cache import demand


@pytest.mark.parametrize('funding', ('mandatory_parent', 'current_image_extra', 'optional_prefetch'))
def test_private_build_failure_with_failed_unlink_keeps_its_funding_credit(tmp_path, monkeypatch, funding):
    context, view, pool = case(tmp_path, monkeypatch)
    planned = demand(context.source_volume.shape, {1:(1, 1, 5, 6)})
    original_unlink = Path.unlink
    original_close = runtime.close_memmap_array_without_flush
    def close_without_background_unlink(array, *, unlink_path=None):
        # Retire the actual mmap normally, but keep deletion under the tested
        # transaction's control; the generic daemon would otherwise race the
        # explicit failure injection after its temporary patch exits.
        return original_close(array, unlink_path=None)
    monkeypatch.setattr(runtime, 'close_memmap_array_without_flush', close_without_background_unlink)
    render_failure = ValueError('controlled rendering failure before publication')
    def deny_owned_image(path, *args, **kwargs):
        if path.parent == context.temp_dir/'sam_image_cache' and path.name.endswith('.gray8.dat'):
            raise PermissionError('controlled private cache unlink failure')
        return original_unlink(path, *args, **kwargs)
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0 if funding == 'mandatory_parent' else 128,
                image_phase_bytes=context.mandatory_image_phase_bytes() if funding == 'mandatory_parent' else 0,
                headroom_probe=lambda:128*resources.GIB) as profile:
            with context.resource_scope(profile), mock.patch.object(context, '_render_demand_crop',
                    side_effect=render_failure), mock.patch.object(Path, 'unlink', deny_owned_image):
                with pytest.raises(ValueError) as caught:
                    provider = (context.prefetch_image_cohort(view, context.source_volume.shape, planned)
                        if funding == 'optional_prefetch' else
                        context.image_cohort_provider(view, context.source_volume.shape, planned))
                    assert provider is not None
                    with provider:
                        pytest.fail('failed private image reached its consumer')
            assert caught.value is render_failure
            paths = tuple((context.temp_dir/'sam_image_cache').glob('*.gray8.dat'))
            assert paths and all(path.exists() for path in paths)
        expected_credit = profile._lease.charged_bytes if funding == 'mandatory_parent' else 4096+20
        assert resources.sam_parent_promised_bytes(pool) == expected_credit
        assert not profile._lease.active
        assert profile._lease.credit_returned is (funding != 'mandatory_parent')
        with mock.patch.object(Path, 'unlink', deny_owned_image):
            with pytest.raises(PermissionError, match='private cache unlink failure'):
                context.close()
        assert resources.sam_parent_promised_bytes(pool) == expected_credit
        assert all(path.exists() for path in paths)
        context.close()
        assert resources.sam_parent_promised_bytes(pool) == 0 and profile._lease.credit_returned
        assert all(not path.exists() for path in paths)
    finally:
        context.close()


@pytest.mark.parametrize('funding', ('mandatory_parent', 'current_image_extra', 'optional_prefetch'))
def test_private_build_failure_keeps_funding_until_last_mmap_alias_retires(tmp_path, monkeypatch, funding):
    context, view, pool = case(tmp_path, monkeypatch)
    planned = demand(context.source_volume.shape, {1:(1, 1, 5, 6)})
    original_close = runtime.close_memmap_array_without_flush
    monkeypatch.setattr(runtime, 'close_memmap_array_without_flush',
        lambda array, *, unlink_path=None:original_close(array, unlink_path=None))
    aliases = []
    failure = ValueError('controlled donor-copy failure with retained mapping')
    def retain_then_fail(target, *args, **kwargs):
        aliases.append(memoryview(target))
        raise failure
    monkeypatch.setattr(context, '_copy_cached_pixels', retain_then_fail)
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0 if funding == 'mandatory_parent' else 128,
                image_phase_bytes=context.mandatory_image_phase_bytes() if funding == 'mandatory_parent' else 0,
                headroom_probe=lambda:128*resources.GIB) as profile:
            with context.resource_scope(profile):
                with pytest.raises(ValueError) as caught:
                    provider = (context.prefetch_image_cohort(view, context.source_volume.shape, planned)
                        if funding == 'optional_prefetch' else
                        context.image_cohort_provider(view, context.source_volume.shape, planned))
                    assert provider is not None
                    with provider:
                        pytest.fail('failed private image reached its consumer')
            assert caught.value is failure and len(aliases) == 1
            paths = tuple((context.temp_dir/'sam_image_cache').glob('*.gray8.dat'))
            assert paths and all(path.exists() for path in paths)
        expected_credit = profile._lease.charged_bytes if funding == 'mandatory_parent' else 4096+20
        assert resources.sam_parent_promised_bytes(pool) == expected_credit
        with pytest.raises(RuntimeError, match='private image mappings remain live'):
            context.close()
        assert resources.sam_parent_promised_bytes(pool) == expected_credit
        assert all(path.exists() for path in paths)
        aliases.pop().release()
        gc.collect()
        context.close()
        assert resources.sam_parent_promised_bytes(pool) == 0 and profile._lease.credit_returned
        assert all(not path.exists() for path in paths)
    finally:
        for alias in aliases:
            alias.release()
        aliases.clear()
        gc.collect()
        context.close()
