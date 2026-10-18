"""Exact cache funding survives failed retirement without unrelated extra owners."""
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest

from XTA import sam_resources as resources
from tests.test_sam_mandatory_image_lifetime import case
from tests.test_sam_view_image_cache import demand


@pytest.mark.parametrize('funding', ['mandatory_parent', 'current_image_extra', 'optional_prefetch'])
def test_published_funded_close_requires_actual_file_retirement(tmp_path, monkeypatch, funding):
    context, view, pool = case(tmp_path, monkeypatch)
    planned = demand(context.source_volume.shape, {1: (1, 1, 5, 6)})
    context._runtime.release_source_cache = mock.Mock(
        side_effect=RuntimeError('unproven published worker retirement'))
    original_unlink = Path.unlink

    def deny_owned(path, *args, **kwargs):
        if path.parent == context.temp_dir / 'sam_image_cache' and path.name.endswith('.gray8.dat'):
            raise PermissionError('published image unlink denied')
        return original_unlink(path, *args, **kwargs)

    try:
        with resources.admit_sam_parent_resources(pool, 4 * resources.GIB, funding,
                base_allowance_bytes=0 if funding == 'mandatory_parent' else 128,
                image_phase_bytes=context.mandatory_image_phase_bytes() if funding == 'mandatory_parent' else 0,
                headroom_probe=lambda: 128 * resources.GIB) as profile:
            with context.resource_scope(profile):
                provider = (context.prefetch_image_cohort(view, context.source_volume.shape, planned)
                    if funding == 'optional_prefetch' else
                    context.image_cohort_provider(view, context.source_volume.shape, planned))
                assert provider is not None
                with pytest.raises(RuntimeError, match='unproven published worker retirement'):
                    with provider as reference:
                        assert reference.path.exists()
        charged = resources.sam_parent_promised_bytes(pool)
        assert charged > 0 and reference.path.exists()
        monkeypatch.setattr(Path, 'unlink', deny_owned)
        with pytest.raises(PermissionError, match='published image unlink denied'):
            context.close()
        assert reference.path.exists() and resources.sam_parent_promised_bytes(pool) == charged
        monkeypatch.setattr(Path, 'unlink', original_unlink)
        context.close()
        assert not reference.path.exists() and resources.sam_parent_promised_bytes(pool) == 0
    finally:
        monkeypatch.setattr(Path, 'unlink', original_unlink)
        context.close()


def test_unproven_cache_does_not_retain_successfully_retired_other_grant(tmp_path, monkeypatch):
    context, view, pool = case(tmp_path, monkeypatch)
    shape = context.source_volume.shape
    plans = [demand(shape, {frame: (1, 1, 5, 6)}) for frame in (0, 1)]
    first_scope, second_scope = ExitStack(), ExitStack()
    first_path = None
    original_retire = context._runtime.release_source_cache

    def retire(reference):
        if reference.path == first_path:
            raise RuntimeError('first cache remains unproven')
        return original_retire(reference)

    monkeypatch.setattr(context._runtime, 'release_source_cache', retire)
    try:
        with resources.admit_sam_parent_resources(pool, 4 * resources.GIB, 'two-image-grants',
                base_allowance_bytes=4 * resources.GIB, headroom_probe=lambda: 128 * resources.GIB) as profile:
            with context.resource_scope(profile):
                first = context.prefetch_image_cohort(view, shape, plans[0])
                second = context.prefetch_image_cohort(view, shape, plans[1])
                assert first is not None and second is not None
                first_reference = first_scope.enter_context(first)
                first_path = first_reference.path
                second_reference = second_scope.enter_context(second)
                assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == \
                    first.phase_bytes + second.phase_bytes
                with pytest.raises(RuntimeError, match='first cache remains unproven'):
                    first_scope.close()
                assert first_path.exists()
                second_scope.close()
                assert not second_reference.path.exists()
                # Only the actual unsettled cache remains funded. An unrelated
                # failed owner cannot capture a completed producer's whole grant.
                assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == first.phase_bytes
                context.close()
                assert not first_path.exists()
                assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == 0
        assert resources.sam_parent_promised_bytes(pool) == 0
    finally:
        try:
            first_scope.close()
        finally:
            try:
                second_scope.close()
            finally:
                context.close()
