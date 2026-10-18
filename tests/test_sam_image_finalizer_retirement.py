"""Proven image cleanup must release callbacks as well as resource credit."""
import gc
import weakref

import pytest

from XTA import runtime, sam_resources as resources
from tests.test_sam_mandatory_image_lifetime import case
from tests.test_sam_view_image_cache import demand


def test_successful_close_detaches_finalizer_while_closed_mapping_object_is_retained(tmp_path, monkeypatch):
    context, view, pool = case(tmp_path, monkeypatch)
    planned = demand(context.source_volume.shape, {1:(1, 1, 5, 6)})
    original_close = runtime.close_memmap_array_without_flush
    observed = {}
    failure = ValueError('retain private mapping object after image failure')
    def retain_then_fail(target, *args, **kwargs):
        observed['mapping'] = target._mmap
        observed['alias'] = memoryview(target)
        raise failure
    with monkeypatch.context() as construction:
        construction.setattr(runtime, 'close_memmap_array_without_flush',
            lambda array, *, unlink_path=None:original_close(array, unlink_path=None))
        construction.setattr(context, '_copy_cached_pixels', retain_then_fail)
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0, image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as profile:
            with context.resource_scope(profile):
                with pytest.raises(ValueError, match='retain private mapping'):
                    with context.image_cohort_provider(view, context.source_volume.shape, planned):
                        pytest.fail('failed image reached its consumer')
    failure.__traceback__ = None
    mapping = observed.pop('mapping')
    observed.pop('alias').release()
    gc.collect()
    mapping.close()
    assert mapping.closed and resources.sam_parent_promised_bytes(pool) > 0
    finalizers = [private['mapping_finalizer']
                  for private in context._parent_image_private_paths.values()]
    assert finalizers and all(finalizer.alive for finalizer in finalizers)
    context.close()
    assert resources.sam_parent_promised_bytes(pool) == 0
    assert all(not finalizer.alive for finalizer in finalizers)
    reference = weakref.ref(context)
    del context
    gc.collect()
    assert reference() is None
    assert mapping.closed  # Keeping this Python object no longer pins the context.
