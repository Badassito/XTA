"""Do not add terminal score pre-masking to synchronous-only capture modes."""
import numpy as np
import pytest

from XTA import confidence_evidence as evidence
from XTA.confidence_publication import ConfidencePublicationQueue
from tests.test_terminal_score_mask_admission import _owned_prepare


@pytest.mark.parametrize('mode', ['none', 'custom', 'configured', 'refused', 'disabled'])
def test_assembly_score_preparation_requires_funded_async_capture(tmp_path, monkeypatch, mode):
    monkeypatch.delenv('YOLO_TTA_TERMINAL_SCORE_ONLY', raising=False)
    if mode == 'disabled':
        monkeypatch.setenv('YOLO_TTA_TERMINAL_SCORE_ONLY', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '0')
    evidence.configure_confidence_evidence(tmp_path/'output', enabled=True,
        defer_projection=True, background_publication=True)
    task, plan, _roots, expected_mask, _expected_scores = _owned_prepare(
        tmp_path, lambda *_args: None, lambda *_args: None, with_metadata=False)
    scores = task.confmap_mm
    scores.fill(143)  # Positive background makes an unwanted prepass observable.
    original = scores.copy()
    expected_masked = np.where(expected_mask != 0, original, np.uint8(0))
    calls, captures, refusals = [], [], []
    actual_prepare = evidence.prepare_terminal_confidence_scores

    def prepare(mask, values, **kwargs):
        assert mode in ('configured', 'refused'), 'synchronous capture must not pre-mask'
        calls.append(id(values))
        return actual_prepare(mask, values, **kwargs)

    def capture(mask, values, **_kwargs):
        assert values is scores
        if mode in ('none', 'custom', 'disabled'):
            assert mask is not values and values.flags.writeable
            np.testing.assert_array_equal(values, original)
        else:
            assert mask is values and not values.flags.writeable
            np.testing.assert_array_equal(values, expected_masked)
        captures.append(id(values))

    monkeypatch.setattr(evidence, 'prepare_terminal_confidence_scores', prepare)
    monkeypatch.setattr(evidence, 'capture_prediction_confidence', capture)
    try:
        if mode == 'custom':
            # Custom numeric publication limits intentionally decline the startup
            # terminal queue; they still support the original synchronous route.
            evidence._PUBLICATION.close()
            evidence._PUBLICATION = ConfidencePublicationQueue(
                max_pending_bytes=128*1024**2, stage_bytes=32*1024**2)
            assert not evidence.configure_terminal_confidence_capture(slots=1,
                workspace_bytes=plan.workspace_bytes, source_bytes=2*scores.nbytes)
        elif mode in ('configured', 'refused', 'disabled'):
            owner = evidence.configure_terminal_confidence_capture(slots=1,
                workspace_bytes=plan.workspace_bytes, source_bytes=2*scores.nbytes)
            assert owner
            if mode == 'refused':
                def refuse(request):
                    # Availability is not an admission grant. A later refusal
                    # must reuse the same prepared source synchronously.
                    assert request.mask is request.scores is scores
                    assert request.source_bytes == scores.nbytes
                    refusals.append(id(request.scores))
                    return None
                monkeypatch.setattr(owner, 'try_submit', refuse)
        assert evidence.terminal_confidence_capture_available() == (mode in ('configured', 'refused'))
        result = task()
        evidence.drain_confidence_publication()
        assert result._packed_native_source_retired
        assert captures == [id(scores)]
        if mode in ('none', 'custom', 'disabled'):
            assert calls == [] and scores.flags.writeable
            np.testing.assert_array_equal(scores, original)
        else:
            assert calls == [id(scores)] and not scores.flags.writeable
            np.testing.assert_array_equal(scores, expected_masked)
        assert refusals == ([id(scores)] if mode == 'refused' else [])
    finally:
        evidence.shutdown_confidence_publication(raise_errors=False)
        evidence.configure_confidence_evidence(None, enabled=False)


@pytest.mark.parametrize('value,expected', [(None, True), ('0', False), ('1', True)])
def test_score_ownership_comparison_mode_is_in_source_provenance(monkeypatch, value, expected):
    from XTA.pipeline import _execution_runtime_provenance
    monkeypatch.delenv('YOLO_TTA_TERMINAL_SCORE_ONLY', raising=False)
    if value is not None:
        monkeypatch.setenv('YOLO_TTA_TERMINAL_SCORE_ONLY', value)
    assert _execution_runtime_provenance()['terminal_confidence_score_only_requested'] is expected
