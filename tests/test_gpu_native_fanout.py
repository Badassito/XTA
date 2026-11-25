"""One-forward direct/ring native cutoff fanout, exercised without CUDA."""
import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import inference, backprojection, forward_replay


class Sink:
    floor_cent = 20
    cutoffs = (50, 90, 95, 100)
    def __init__(self, fail_cutoff=None):
        self.results = {}
        self.disabled = []
        self.fail_cutoff = fail_cutoff
        self.counters = {key: 0 for key in ('rendered_frames', 'render_batches', 'forward_calls', 'mask_decodes')}
    def record_detector_work(self, **values):
        for key, value in values.items():
            self.counters[key] += value
    def disable(self, reason):
        self.disabled.append(reason)
    def consume_result(self, index, cutoff, payload):
        if cutoff == self.fail_cutoff:
            raise RuntimeError('writer sentinel')
        self.results[index, cutoff] = payload
    def consume_native(self, index, cutoff, mask, score, count, *, proto_treated=False):
        if cutoff == self.fail_cutoff:
            raise RuntimeError('writer sentinel')
        self.results[index, cutoff] = (mask.clone(), score.clone(), int(count.item()), proto_treated)


class AliasingSink(Sink):
    def __init__(self, *, admitted=True):
        super().__init__()
        self.admitted = admitted
        self.aliases = []
    def consume_same_result(self, index, cutoff, source_cutoff):
        if not self.admitted:
            return False
        self.results[index, cutoff] = self.results[index, source_cutoff]
        self.aliases.append((index, cutoff, source_cutoff))
        return True

class Source:
    _tta_announce_prediction = False
    def __init__(self, torch, sink, count=1):
        self._xta_native_fanout = sink
        self.torch = torch
        self.count = count
        self.iterations = 0
    def __iter__(self):
        self.iterations += 1
        for index in range(self.count):
            yield ['frame'], [self.torch.full((1, 1, 4, 4), float(index))], ['frame']


def direct_fixture(monkeypatch, *, sink=None, frames=1, unsupported=False):
    torch = pytest.importorskip('torch')
    sink = sink or Sink()
    source = Source(torch, sink, frames)
    class Backend:
        names = {0: 'foreground'}
        calls = 0
        def __call__(self, image):
            self.calls += 1
            head = torch.zeros((1, 7, 3))
            head[:, :4] = 2.
            head[:, 4] = torch.tensor([.1, .5, .9]) if image.flatten()[0] == 0 else .1
            return head, torch.ones((1, 2, 2, 2))
    backend = Backend()
    predictor = SimpleNamespace(model=backend, preprocess=lambda images: images[0])
    cache = SimpleNamespace(stores=0, lookups=0)
    def key(*args, **kwargs):
        cache.lookups += 1
        return 'raw'
    cache.input_key = key
    cache.load_tensors = lambda *args, **kwargs: None
    def store(*args, **kwargs):
        cache.stores += 1
    cache.store_instance = store
    monkeypatch.setattr(forward_replay, 'torch_replay_for_predictor', lambda *_: cache)
    monkeypatch.setattr(inference, 'direct_device_compaction_enabled', lambda: not unsupported)
    monkeypatch.setattr(inference, 'angle_variant_gpu_fastpath', lambda: None)
    decoded = []
    def decode(head, proto, image, threshold, **kwargs):
        decoded.append(threshold)
        count = int((head[4] >= np.float32(threshold)).sum())
        return inference.GpuFlattenedRetinaPayload(
            union_gpu=(torch.ones((4, 4)) if count else None),
            conf_gpu=(torch.full((4, 4), .9) if count else None),
            instance_count=count, compaction_kernel='scalar')
    monkeypatch.setattr(inference, '_build_direct_device_compacted_payload', decode)
    return torch, source, predictor, backend, sink, cache, decoded


def test_direct_fanout_captures_floor_once_and_streams_siblings_without_raw_journal(monkeypatch):
    torch, source, predictor, backend, sink, cache, decoded = direct_fixture(monkeypatch)
    cfg = inference.PredictConfig(4, .2, 'cuda:0', 'fp32')
    floor = list(inference._direct_predict_stream(predictor, source, cfg, 'fanout'))
    assert backend.calls == source.iterations == 1
    assert cache.stores == cache.lookups == 0
    assert sink.counters == dict(rendered_frames=1, render_batches=1, forward_calls=1, mask_decodes=3)
    assert floor[0]._tta_gpu_flattened_payload.sweep_fanout_capture == (sink, 0)
    assert decoded == [.2, .5, .9]  # .95 and 1.0 are provably empty.
    assert [sink.results[0, cutoff].instance_count for cutoff in sink.cutoffs] == [2, 1, 0, 0]
    assert source._native_trt_data_consumed
    assert not torch.cuda.is_initialized()


def test_direct_empty_floor_is_tagged_and_all_sibling_empty_counts_are_captured(monkeypatch):
    _torch, source, predictor, backend, sink, cache, decoded = direct_fixture(monkeypatch, frames=2)
    cfg = inference.PredictConfig(4, .2, 'cuda:0', 'fp32')
    floors = list(inference._direct_predict_stream(predictor, source, cfg, 'fanout'))
    assert backend.calls == 2 and source.iterations == 1
    assert floors[1]._tta_gpu_flattened_payload.union_gpu is None
    assert floors[1]._tta_gpu_flattened_payload.sweep_fanout_capture == (sink, 1)
    assert all(sink.results[1, cutoff].instance_count == 0 for cutoff in sink.cutoffs)
    assert decoded.count(.2) == 2 and len(decoded) == 4
    assert cache.stores == 0


def test_direct_unsupported_compaction_declares_fallback_and_preserves_floor(monkeypatch):
    _torch, source, predictor, backend, sink, cache, _decoded = direct_fixture(monkeypatch, unsupported=True)
    cfg = inference.PredictConfig(4, .2, 'cuda:0', 'fp32')
    floor = list(inference._direct_predict_stream(predictor, source, cfg, 'unsupported'))
    assert len(floor) == backend.calls == 1
    assert len(sink.disabled) == 1 and not sink.results
    assert cache.stores == 1
    assert getattr(floor[0]._tta_gpu_flattened_payload, 'sweep_fanout_capture', None) is None


def test_direct_writer_failure_propagates_after_floor_without_retry(monkeypatch):
    _torch, source, predictor, backend, sink, cache, _decoded = direct_fixture(
        monkeypatch, sink=Sink(fail_cutoff=50))
    cfg = inference.PredictConfig(4, .2, 'cuda:0', 'fp32')
    generator = inference._direct_predict_stream(predictor, source, cfg, 'failed-writer')
    next(generator)
    with pytest.raises(RuntimeError, match='writer sentinel'):
        next(generator)
    assert backend.calls == 1 and source._native_trt_data_consumed
    assert cache.stores == 0


def test_source_fanout_is_stable_across_wrappers_and_cycles():
    sink = Sink()
    base = SimpleNamespace(_xta_native_fanout=sink)
    outer = SimpleNamespace(base_source=SimpleNamespace(base_source=base))
    assert inference._native_fanout_for_source(outer) is sink
    inference._mark_native_fanout_consumed(outer)
    assert base._native_trt_data_consumed and outer._native_trt_data_consumed
    cycle = SimpleNamespace()
    cycle.base_source = cycle
    assert inference._native_fanout_for_source(cycle) is None


def ring_fixture(*, graph=False, sink=None):
    torch = pytest.importorskip('torch')
    events, launches = [], []
    stream = SimpleNamespace(wait_event=lambda event: events.append('wait:' + event))
    slot = SimpleNamespace(
        post_stream=stream, infer_done='infer',
        post_done=SimpleNamespace(record=lambda stream: events.append('post_done')),
        native_union=torch.zeros((4, 4), dtype=torch.uint8),
        native_conf=torch.zeros((4, 4), dtype=torch.uint8),
        compact_count=torch.zeros((1,), dtype=torch.int32), post_valid=False,
    )
    executor = object.__new__(backprojection._ResidentTensorRTRingExecutor)
    executor.torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext()))
    executor.collect_slice_bboxes = False
    executor.confidence_threshold = .2
    executor.proto_hole_treatment_active = False
    executor._set_slot_unit_descriptor = lambda slot, desc: None
    scores = torch.tensor([.1, .5, .9])
    def launch(slot):
        launches.append(executor.confidence_threshold)
        accepted = scores >= np.float32(executor.confidence_threshold)
        slot.compact_count[0] = int(accepted.sum())
        slot.native_union.zero_(); slot.native_conf.zero_()
        slot.native_union.view(-1)[:3] = accepted.to(torch.uint8)
        slot.native_conf.view(-1)[:3] = torch.where(accepted, (scores * 255).round(), 0).to(torch.uint8)
    executor._launch_post = launch
    slot.post_graph = SimpleNamespace(replay=lambda: launch(slot)) if graph else None
    union = SimpleNamespace(
        union_dev=torch.zeros((1, 4, 4), dtype=torch.uint8),
        conf_dev=torch.zeros((1, 4, 4), dtype=torch.uint8), written=[False],
        register_producer_stream=lambda stream: events.append('producer'),
    )
    frame_counts = torch.zeros(1, dtype=torch.int32)
    return executor, slot, union, frame_counts, sink or Sink(), events, launches


@pytest.mark.parametrize('graph', [False, True])
def test_ring_fanout_matches_independent_scalar_post_and_preserves_floor(graph):
    torch = pytest.importorskip('torch')
    executor, slot, union, counts, sink, events, launches = ring_fixture(graph=graph)
    executor.enqueue_postprocess(slot, descriptor=SimpleNamespace(unit_index=0, destination_index=0),
        device_union=union, frame_counts_dev=counts, native_fanout=sink)
    assert launches == [.2, .5, .9, .95]  # Last 1.0 shares the proven empty .95 output.
    assert executor.confidence_threshold == .2
    assert counts.item() == 2 and union.written == [True]
    assert slot.post_valid and events[-1] == 'post_done'
    assert sink.counters['mask_decodes'] == 4
    assert torch.equal(union.union_dev[0], sink.results[0, 20][0])
    assert torch.equal(union.conf_dev[0], sink.results[0, 20][1])
    for cutoff in (20, *sink.cutoffs):
        expected = (torch.tensor([.1, .5, .9]) >= np.float32(cutoff / 100.))
        mask, score, count, treated = sink.results[0, cutoff]
        assert torch.equal(mask.flatten()[:3], expected.to(torch.uint8))
        assert count == int(expected.sum()) and not treated
        assert not score[~mask.bool()].any()
    assert not torch.cuda.is_initialized()


def test_ring_writer_failure_restores_floor_gate_and_does_not_publish_post_done():
    executor, slot, union, counts, sink, events, launches = ring_fixture(sink=Sink(fail_cutoff=50))
    with pytest.raises(RuntimeError, match='writer sentinel'):
        executor.enqueue_postprocess(slot, descriptor=SimpleNamespace(unit_index=0, destination_index=0),
            device_union=union, frame_counts_dev=counts, native_fanout=sink)
    assert executor.confidence_threshold == .2
    assert 'post_done' not in events and not slot.post_valid
    assert not union.written[0]

def test_direct_sibling_decode_unavailable_invalidates_fanout_but_keeps_floor(monkeypatch):
    _torch, source, predictor, backend, sink, cache, _decoded = direct_fixture(monkeypatch)
    original = inference._build_direct_device_compacted_payload
    monkeypatch.setattr(inference, '_build_direct_device_compacted_payload',
        lambda head, proto, image, threshold, **kwargs:
            original(head, proto, image, threshold, **kwargs) if threshold == .2 else None)
    cfg = inference.PredictConfig(4, .2, 'cuda:0', 'fp32')
    floor = list(inference._direct_predict_stream(predictor, source, cfg, 'unsupported-sibling'))
    assert len(floor) == backend.calls == 1
    assert floor[0]._tta_gpu_flattened_payload.instance_count == 2
    assert len(sink.disabled) == 1 and not sink.results
    assert cache.stores == cache.lookups == 0

def test_direct_fanout_prefers_cumulative_retina_consumer(monkeypatch):
    from XTA import sweep_proto
    sink = Sink()
    calls = []
    def cumulative(*args, **kwargs):
        calls.append((args, kwargs))
        return True
    monkeypatch.setattr(sweep_proto, 'consume_sibling_bins', cumulative)
    # A successful cumulative consumer must return before reading the raw head.
    assert inference._direct_native_fanout_siblings(sink, 4, None, None, None)
    assert len(calls) == 1
    assert calls[0][0][:2] == (sink, 4)
    assert calls[0][1] == {}

@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_FANOUT_CUDA') != '1',
    reason='explicit coordinated native-fanout CUDA qualification only')
@pytest.mark.parametrize('dtype_name', ['float16', 'float32'])
@pytest.mark.parametrize('policy', ['legacy_proto', 'native_mask'])
@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('affine', [False, True])
def test_real_cuda_ring_native_fanout_matches_independent_scalar_post(dtype_name, policy, graph, affine):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    kernels = inference._resident_mask_kernels()
    assert kernels is not None
    dtype = getattr(torch, dtype_name)
    head = torch.zeros((37, 3), dtype=dtype, device='cuda')
    head[:4] = torch.tensor([[32.], [32.], [64.], [64.]], dtype=dtype, device='cuda')
    head[4] = torch.tensor([.1, .5, .9], dtype=dtype, device='cuda')
    head[5, 1] = head[6, 2] = 1.
    proto = torch.full((32, 16, 16), -1., dtype=dtype, device='cuda')
    proto[0, 2:14, 2:14] = 1.
    proto[0, 7:9, 7:9] = -1.
    proto[1, 4:12, 9:14] = 1.
    matrix = (np.asarray(((.93, .08, 1.25), (-.07, .9, 2.1)), np.float32)
              if affine else np.eye(2, 3, dtype=np.float32))
    descriptor = inference.ResidentRingUnitDescriptor(0, 0, 64, 64, matrix)
    def executor():
        ex = object.__new__(backprojection._ResidentTensorRTRingExecutor)
        ex.torch, ex.kernels = torch, kernels
        ex.post_policy = policy
        ex.native_h = ex.native_w = ex.out_size = 64
        ex.track_conf, ex.retain_confidence = False, True
        ex.collect_slice_bboxes = True
        ex.confidence_threshold = .2
        ex.proto_hole_radius = 1
        ex.proto_hole_treatment_active = False
        ex.dynamic_unit_descriptors = False
        ex.default_descriptor = descriptor
        ex.identity_native_warp, ex.native_to_out = ex._descriptor_warp(descriptor)
        slot = SimpleNamespace(head=head[None], proto=proto[None], post_stream=torch.cuda.Stream(),
            slot_id=0, post_valid=False, infer_valid=False, post_graph=None,
            infer_done=torch.cuda.Event(blocking=False), post_done=torch.cuda.Event(blocking=False))
        ex._allocate_post_buffers(slot)
        ex._set_slot_unit_descriptor(slot, descriptor)
        slot.infer_done.record(torch.cuda.current_stream())
        return ex, slot
    baseline, plain = executor()
    expected = {}
    for cutoff in (20, 50, 90, 95, 100):
        baseline.confidence_threshold = cutoff / 100.
        with torch.cuda.stream(plain.post_stream):
            plain.post_stream.wait_event(plain.infer_done)
            baseline._launch_post(plain)
        plain.post_stream.synchronize()
        expected[cutoff] = (plain.native_union.cpu(), plain.native_conf.cpu(),
                            int(plain.compact_count.item()), plain.native_bbox.cpu())
    ex, slot = executor()
    if graph:
        with torch.cuda.stream(slot.post_stream):
            slot.post_stream.wait_event(slot.infer_done)
            ex._launch_post(slot)
        slot.post_stream.synchronize()
        captured = torch.cuda.CUDAGraph()
        with inference._cuda_graph_capture_context(torch, captured, slot.post_stream):
            ex._launch_post(slot)
        slot.post_stream.synchronize()
        slot.post_graph = captured
    sink = AliasingSink()
    union = SimpleNamespace(
        union_dev=torch.zeros((1, 64, 64), dtype=torch.uint8, device='cuda'),
        conf_dev=torch.zeros((1, 64, 64), dtype=torch.uint8, device='cuda'),
        slice_bboxes_dev=torch.empty((1, 4), dtype=torch.int32, device='cuda'), written=[False],
        register_producer_stream=lambda stream: None)
    union.write_slice_bbox = lambda index, bbox: union.slice_bboxes_dev[index].copy_(bbox)
    counts = torch.zeros(1, dtype=torch.int32, device='cuda')
    slot.infer_done.record(torch.cuda.current_stream())
    ex.enqueue_postprocess(slot, descriptor=descriptor, device_union=union,
        frame_counts_dev=counts, native_fanout=sink)
    slot.post_done.synchronize()
    for cutoff in (20, *sink.cutoffs):
        mask, score, count, treated = sink.results[0, cutoff]
        assert torch.equal(mask.cpu(), expected[cutoff][0])
        assert torch.equal(score.cpu(), expected[cutoff][1])
        assert count == expected[cutoff][2] and not treated
    assert torch.equal(union.union_dev[0].cpu(), expected[20][0])
    assert torch.equal(union.conf_dev[0].cpu(), expected[20][1])
    assert torch.equal(union.slice_bboxes_dev[0].cpu(), expected[20][3])
    assert int(counts.item()) == expected[20][2]
    assert ex.confidence_threshold == .2 and slot.post_valid and union.written[0]

@pytest.mark.parametrize('dtype_name', ['float16', 'float32'])
@pytest.mark.parametrize('cutoffs', [(90, 100), (50, 70, 90, 100), (50, 60, 70, 90, 100)])
def test_ring_cutoff_counts_match_exact_fp32_scalar_gates(dtype_name, cutoffs):
    torch = pytest.importorskip('torch')
    head = torch.zeros((1, 37, 12), dtype=getattr(torch, dtype_name))
    head[0, 4] = torch.tensor([.2, .5, .6, .7, .9, 1., .4999, .7001,
                             float('nan'), -float('inf'), float('inf'), 0.])
    actual = backprojection._ring_native_cutoff_counts(head, .2, cutoffs)
    expected = tuple(int((head[0, 4].float() >= np.float32(gate)).sum())
                     for gate in (.2, *[cutoff / 100. for cutoff in cutoffs]))
    assert actual == expected


def test_ring_two_cut_schedule_does_not_build_histogram(monkeypatch):
    torch = pytest.importorskip('torch')
    head = torch.zeros((1, 37, 12))
    monkeypatch.setattr(torch, 'bucketize', lambda *a, **kw:
        (_ for _ in ()).throw(AssertionError('two cuts must not add histogram work')))
    assert backprojection._ring_native_cutoff_counts(head, .2, (50,)) is None


@pytest.mark.parametrize('admitted', [False, True])
def test_ring_equal_candidate_sets_alias_only_when_writer_accepts(admitted):
    torch = pytest.importorskip('torch')
    sink = AliasingSink(admitted=admitted)
    executor, slot, union, counts, sink, events, launches = ring_fixture(sink=sink)
    slot.head = torch.zeros((1, 7, 3))
    slot.head[0, 4] = torch.tensor([.1, .5, .9])
    executor.enqueue_postprocess(slot, descriptor=SimpleNamespace(unit_index=0, destination_index=0),
        device_union=union, frame_counts_dev=counts, native_fanout=sink)
    assert launches == ([.2, .9, .95] if admitted else [.2, .5, .9, .95])
    assert sink.counters['mask_decodes'] == (3 if admitted else 4)
    assert len(sink.results) == 5 and counts.item() == 2
    if admitted:
        assert sink.aliases == [(0, 50, 20), (0, 100, 95)]
    assert executor.confidence_threshold == .2 and events[-1] == 'post_done'


def test_ring_scalar_owned_floor_materializes_first_equal_sibling_from_live_slot():
    torch = pytest.importorskip('torch')
    class ScalarOwnedSink(AliasingSink):
        scalar_owned_floor = True

        def consume_same_result(self, index, centroid, source):
            if source == self.floor_cent:
                return False
            return super().consume_same_result(index, centroid, source)

        def consume_native(self,index,cutoff,mask,score,count,*,proto_treated=False):
            if cutoff == self.floor_cent:
                self.results[index,cutoff] = ('owned',int(count))
            else:
                super().consume_native(index,cutoff,mask,score,count,proto_treated=proto_treated)

        def consume_empty(self,index,cutoff,count):
            self.results[index,cutoff] = ('empty',int(count))

    sink=ScalarOwnedSink(admitted=True)
    executor, slot, union, counts, _, events, launches = ring_fixture(sink=sink)
    slot.head=torch.zeros((1,7,3))
    slot.head[0,4]=torch.tensor([.1,.5,.9])
    executor.enqueue_postprocess(slot, descriptor=SimpleNamespace(unit_index=0,destination_index=0),
                                 device_union=union,frame_counts_dev=counts,native_fanout=sink)
    assert sink.results[0,20] == ('owned',2)
    assert torch.equal(sink.results[0,50][0],union.union_dev[0])
    assert torch.equal(sink.results[0,50][1],union.conf_dev[0])
    assert sink.results[0,95] == sink.results[0,100] == ('empty',0)
    assert launches == [.2,.9]
    assert counts.item()==2 and events[-1]=='post_done'
