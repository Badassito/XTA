"""A failed SAM shutdown preserves the scheduler failure and retained owners."""
import ast
from pathlib import Path

import pytest


def close_after_scheduler():
    # Exercise the production helper without importing the GPU pipeline stack.
    source = Path(__file__).parents[1] / 'XTA' / 'pipeline.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_close_sam_context_after_scheduler')
    namespace = {'__name__': 'XTA.pipeline', '__package__': 'XTA'}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[helper.name]


class RetainedContext:
    def __init__(self, failure=None):
        self.failure = failure
        self.closed = self._closed = False
        self.source = object()
        self.source_volume = self.source
        self.retained_owners = [object()]
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure
        self.closed = self._closed = True
        self.source_volume = None


@pytest.fixture(autouse=True)
def retained_contexts(monkeypatch):
    from XTA import sam_integration
    retained = []
    original = sam_integration._retain_unsettled_context

    def retain(context):
        original(context)
        retained.append(context)

    monkeypatch.setattr(sam_integration, '_retain_unsettled_context', retain)
    yield retained
    for context in retained:
        sam_integration._forget_settled_context(context)


@pytest.mark.parametrize('cleanup_type', [RuntimeError, KeyboardInterrupt])
def test_cleanup_failure_preserves_primary_and_allows_independent_teardown(cleanup_type, retained_contexts):
    close = close_after_scheduler()
    primary = RuntimeError('image-only admission failed')
    secondary = cleanup_type('retry source ownership remains unsettled')
    context = RetainedContext(secondary)
    source, owners = context.source, list(context.retained_owners)
    independently_closed = []

    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        finally:
            close(context, primary)
            independently_closed.append('finalization and reduction executors')

    assert caught.value is primary
    assert primary.__notes__ == [
        f'SAM context cleanup also failed: {cleanup_type.__name__}: {secondary}']
    assert independently_closed == ['finalization and reduction executors']
    assert context.close_calls == 1 and not context.closed
    assert context.source is source and context.retained_owners == owners
    assert retained_contexts == [context]


def test_cleanup_failure_without_primary_still_propagates(retained_contexts):
    failure = RuntimeError('unsettled SAM ownership')
    context = RetainedContext(failure)
    with pytest.raises(RuntimeError) as caught:
        close_after_scheduler()(context, None)
    assert caught.value is failure
    assert context.close_calls == 1 and not context.closed
    assert retained_contexts == [context]


@pytest.mark.parametrize('primary', [None, RuntimeError('scheduler failed')])
def test_successful_cleanup_is_unchanged(primary, retained_contexts):
    context = RetainedContext()
    close_after_scheduler()(context, primary)
    assert context.close_calls == 1 and context.closed
    assert not retained_contexts
    if primary is not None:
        assert not hasattr(primary, '__notes__')


def test_fully_closed_context_with_reported_shutdown_error_is_not_retained(retained_contexts):
    class ClosedContext(RetainedContext):
        def close(self):
            super().close()
            raise RuntimeError('settled worker reported a shutdown error')

    context = ClosedContext()
    primary = RuntimeError('scheduler failed')
    close_after_scheduler()(context, primary)
    assert context.closed and not retained_contexts
    assert primary.__notes__ == [
        'SAM context cleanup also failed: RuntimeError: settled worker reported a shutdown error']


def test_closed_flag_before_final_source_retirement_still_retains_context(retained_contexts):
    context = RetainedContext(RuntimeError('lease retirement failed'))
    context._closed = True
    primary = RuntimeError('scheduler failed')
    close_after_scheduler()(context, primary)
    assert context.source_volume is context.source
    assert retained_contexts == [context]
