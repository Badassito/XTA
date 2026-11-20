"""CPU-only qualification of sparse raw packets; no CUDA context required."""
import json
import numpy as np
import pytest

from XTA.forward_replay_sparse import (
    is_sparse_instance_packet, pack_instance_packet, unpack_instance_packet,
)


def fixture(dtype='float32', proto_dtype=None, *, batch=2, anchors=512):
    rng = np.random.default_rng(117)
    head = np.zeros((batch, 37, anchors), dtype=dtype)
    head[:, 4] = -1
    proto = rng.normal(size=(batch, 32, 32, 48)).astype(proto_dtype or dtype)
    for b in range(batch):
        indices = np.array([1, 7, 29, 103])
        head[b, :4, indices] = np.array([[8, 10, 7, 9], [17, 15, 5, 3], [26, 23, 6, 8], [33, 11, 4, 3]], dtype=dtype)
        head[b, 4, indices] = np.array([.2, .5, .8, .99], dtype=dtype)
        head[b, 5:, indices] = rng.normal(size=(4, 32)).astype(dtype)
    return head, proto


def decoder_crop_cells(xx, yy, x1, y1, x2, y2):
    """Proto cells the mask decoders read for one box (XTA.inference crop rule).

    A cell is read when its centre c+.5 lies in the half-open box, plus the cell holding a
    non-empty box's centre. Both stay inside the packer's [floor(lo)-1, ceil(hi)+1) halo,
    so persisted candidate-box packets remain decodable under this rule.
    """
    half = np.float32(.5)
    in_x = ((xx + half >= x1) & (xx + half < x2)) | ((x2 > x1) & (xx == np.floor(half * (x1 + x2))))
    in_y = ((yy + half >= y1) & (yy + half < y2)) | ((y2 > y1) & (yy == np.floor(half * (y1 + y2))))
    return in_x & in_y


def nbytes(packet):
    return sum(int(a.numel()) * a.element_size() if hasattr(a, 'numel') else a.nbytes for a in packet)


def check_packet(head, proto, floor=20, support='candidate_boxes'):
    packet = pack_instance_packet(head, proto, floor_hundredths=floor, input_shape_hw=(32, 48), proto_support=support)
    assert packet is not None and is_sparse_instance_packet(packet)
    restored = unpack_instance_packet(packet)
    assert [v.shape for v in restored] == [v.shape for v in (head, proto)]
    if hasattr(head, 'detach'):
        assert all(a.stride() == b.stride() for a, b in zip((head, proto), restored))
        arrays = [a.detach().numpy() for a in (head, proto, *restored)]
        ids = packet[1].numpy(); pixels = packet[3].numpy()
    else:
        assert all(a.strides == b.strides for a, b in zip((head, proto), restored))
        arrays = [head, proto, *restored]; ids = packet[1]; pixels = packet[3]
    h, p, rh, rp = arrays
    b, a = np.divmod(ids, h.shape[2])
    np.testing.assert_array_equal(rh[b, :, a], h[b, :, a])
    keep = np.zeros((h.shape[0], h.shape[2]), bool); keep[b, a] = 1
    assert np.isneginf(rh[:, 4][~keep]).all()
    if support == 'full':
        np.testing.assert_array_equal(rp, p)
    else:
        pb, rem = np.divmod(pixels, p.shape[2] * p.shape[3]); py, px = np.divmod(rem, p.shape[3])
        np.testing.assert_array_equal(rp[pb, :, py, px], p[pb, :, py, px])
        # Independent full global-grid crop oracle: every potential decoder read
        # for each retained anchor must retain every original prototype byte.
        yy, xx = np.indices(p.shape[-2:], dtype=np.float32)
        for bi, ai in zip(b, a):
            cx, cy, w, height = h[bi, :4, ai].astype(np.float32)
            inside = decoder_crop_cells(xx, yy, cx - .5*w, cy - .5*height, cx + .5*w, cy + .5*height)
            np.testing.assert_array_equal(rp[bi, :, inside], p[bi, :, inside])
    return packet


@pytest.mark.parametrize('head_dtype,proto_dtype', [('float16','float16'),('float16','float32'),('float32','float16'),('float32','float32')])
@pytest.mark.parametrize('container', ['numpy', 'torch'])
@pytest.mark.parametrize('floor', [0,20,50,100])
def test_sparse_values_order_global_crop_dtype_and_strides(head_dtype, proto_dtype, container, floor):
    head, proto = fixture(head_dtype, proto_dtype)
    if container == 'torch':
        torch = pytest.importorskip('torch')
        head, proto = torch.from_numpy(head), torch.from_numpy(proto)
    check_packet(head, proto, floor)


@pytest.mark.parametrize('container', ['numpy', 'torch'])
def test_sub_cell_boxes_retain_the_decoder_centre_cell(container):
    head, proto = fixture()
    head[:, :4, 7] = np.array([17.8, 15.8, .5, .4], dtype=np.float32)  # Neither box holds a cell centre.
    head[:, :4, 29] = np.array([26.0, 23.0, .9, .9], dtype=np.float32)
    yy, xx = np.indices(proto.shape[-2:], dtype=np.float32)
    for cx, cy, w, h in head[0, :4, [7, 29]]:
        x1, y1, x2, y2 = cx - np.float32(.5) * w, cy - np.float32(.5) * h, cx + np.float32(.5) * w, cy + np.float32(.5) * h
        assert np.count_nonzero(decoder_crop_cells(xx, yy, x1, y1, x2, y2)) == 1
        assert not np.any((xx + np.float32(.5) >= x1) & (xx + np.float32(.5) < x2)
                          & (yy + np.float32(.5) >= y1) & (yy + np.float32(.5) < y2))
    if container == 'torch':
        torch = pytest.importorskip('torch')
        head, proto = torch.from_numpy(head), torch.from_numpy(proto)
    check_packet(head, proto)


def test_empty_packet_omits_head_and_prototype_payload():
    head, proto = fixture(); head[:, 4] = -1
    packet = check_packet(head, proto)
    assert all(v.size == 0 for v in packet[1:])
    assert nbytes(packet) < 600


@pytest.mark.parametrize('container', ['numpy','torch'])
def test_floor_zero_retains_zero_scores_and_strict_reject_sentinel(container):
    head, proto = fixture(); head[:, 4] = -1; head[:, 4, 7] = 0
    if container == 'torch':
        torch = pytest.importorskip('torch'); head, proto = torch.from_numpy(head), torch.from_numpy(proto)
    packet = check_packet(head, proto, 0)
    ids = packet[1].numpy() if container == 'torch' else packet[1]
    np.testing.assert_array_equal(ids, [7, 519])


def test_half_rounded_gate_conservatively_retained():
    head, proto = fixture('float16'); head[:,4] = -1; head[:,4,1] = np.float16(.2)
    assert float(head[0,4,1]) < float(np.float32(.2))
    packet = check_packet(head, proto, 20)
    np.testing.assert_array_equal(packet[1], [1, 513])


@pytest.mark.parametrize('container', ['numpy','torch'])
def test_dense_permutation_layout_roundtrips(container):
    head, proto = fixture()
    head = np.ascontiguousarray(head.transpose(0,2,1)).transpose(0,2,1)
    proto = np.ascontiguousarray(proto.transpose(0,2,3,1)).transpose(0,3,1,2)
    if container == 'torch':
        torch = pytest.importorskip('torch'); head, proto = torch.from_numpy(head), torch.from_numpy(proto)
    check_packet(head, proto)


@pytest.mark.parametrize('support', ['candidate_boxes','full'])
def test_full_support_and_sparse_payload_smaller(support):
    head, proto = fixture(anchors=16000)
    packet = check_packet(head, proto, support=support)
    assert nbytes(packet) < head.nbytes + proto.nbytes


@pytest.mark.parametrize('mutation', ['float64','multiclass','batch_mismatch','hole_stride','nonfinite_box','negative_width','bad_geometry','bad_floor'])
def test_unsupported_packets_fall_back_without_precision_conversion(mutation):
    head, proto = fixture()
    kwargs = dict(floor_hundredths=20,input_shape_hw=(32,48))
    if mutation == 'float64': head=head.astype(np.float64)
    elif mutation == 'multiclass':head=np.concatenate([head,head[:,:1]],axis=1)
    elif mutation == 'batch_mismatch':proto=proto[:1]
    elif mutation == 'hole_stride':head=head[:,:,::2]
    elif mutation == 'nonfinite_box':head[0,0,1]=np.nan
    elif mutation == 'negative_width':head[0,2,1]=-1
    elif mutation == 'bad_geometry':kwargs['input_shape_hw']=(0,48)
    else:kwargs['floor_hundredths']=-1
    assert pack_instance_packet(head,proto,**kwargs) is None


def test_dense_packet_falls_back_when_no_payload_saving():
    head, proto = fixture(batch=1,anchors=128)
    head[:,4]=1;head[:,0]=24;head[:,1]=16;head[:,2]=48;head[:,3]=32
    assert pack_instance_packet(head,proto,floor_hundredths=0,input_shape_hw=(32,48)) is None


@pytest.mark.parametrize('bad', ['indices','dtype','shape','header','mode','stride'])
def test_corrupt_envelopes_rejected(bad):
    packet=list(check_packet(*fixture()))
    if bad == 'indices':packet[1]=np.array([-1],np.int64)
    elif bad == 'dtype':packet[2]=packet[2].astype(np.float64)
    elif bad == 'shape':packet[4]=packet[4][:,:1]
    else:
        data=json.loads(packet[0].tobytes())
        if bad=='header':data['schema']='wrong'
        elif bad=='mode':data['proto_support']='unknown'
        else:data['head_strides']=[0,0,0]
        packet[0]=np.frombuffer(json.dumps(data).encode(),np.uint8).copy()
    with pytest.raises(ValueError):unpack_instance_packet(packet)


def test_torch_pack_owns_values_and_does_not_initialize_cuda():
    torch=pytest.importorskip('torch')
    head,proto=[torch.from_numpy(v) for v in fixture()]
    packet=check_packet(head,proto)
    saved=[v.clone() for v in packet]
    head.fill_(100);proto.fill_(100)
    assert all(torch.equal(a,b) for a,b in zip(packet,saved))
    assert not torch.cuda.is_initialized()


def test_host_packet_validated_then_sparse_upload_to_requested_device():
    torch=pytest.importorskip('torch')
    packet=check_packet(*fixture('float16','float32'))
    numpy_values=unpack_instance_packet(packet)
    torch_values=unpack_instance_packet(packet,device='cpu')
    assert all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in torch_values)
    for a,b in zip(numpy_values,torch_values):
        np.testing.assert_array_equal(a,b.numpy())
        assert tuple(s//a.itemsize for s in a.strides)==b.stride()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('container',['numpy','torch'])
def test_huge_finite_box_coordinates_do_not_wrap_to_origin(container):
    head,proto=fixture();head[:,4]=-1;head[:,4,1]=.9
    head[:,0,1]=1e20;head[:,1,1]=1e20;head[:,2:4,1]=2.
    if container=='torch':
        torch=pytest.importorskip('torch');head,proto=torch.from_numpy(head),torch.from_numpy(proto)
    packet=pack_instance_packet(head,proto,floor_hundredths=20,input_shape_hw=(32,48))
    assert packet is not None
    assert packet[3].shape[0]==0


def test_bfloat16_returns_dense_fallback_without_conversion():
    torch=pytest.importorskip('torch');head,proto=[torch.from_numpy(v).to(torch.bfloat16) for v in fixture()]
    assert pack_instance_packet(head,proto,floor_hundredths=20,input_shape_hw=(32,48)) is None


@pytest.mark.parametrize('container',['numpy','torch'])
@pytest.mark.parametrize('image_hw',[(97,151),(19,71),(133,29)])
def test_noninteger_global_prototype_scaling_preserves_all_crop_reads(container,image_hw):
    head,proto=fixture()
    head[:,0:2]*=np.float32(3.125)
    head[:,2:4]*=np.float32(1.375)
    if container=='torch':
        torch=pytest.importorskip('torch');head,proto=torch.from_numpy(head),torch.from_numpy(proto)
    packet=pack_instance_packet(head,proto,floor_hundredths=20,input_shape_hw=image_hw)
    assert packet is not None
    rh,rp=unpack_instance_packet(packet)
    if container=='torch':head,proto,rh,rp=[v.numpy() for v in (head,proto,rh,rp)]
    yy,xx=np.indices(proto.shape[-2:],dtype=np.float32)
    for b in range(head.shape[0]):
        for a in np.flatnonzero(head[b,4]>=.2):
            cx,cy,w,h=head[b,:4,a]
            sx=np.float32(proto.shape[3]/image_hw[1]);sy=np.float32(proto.shape[2]/image_hw[0])
            inside=decoder_crop_cells(xx,yy,(cx-np.float32(.5)*w)*sx,(cy-np.float32(.5)*h)*sy,(cx+np.float32(.5)*w)*sx,(cy+np.float32(.5)*h)*sy)
            np.testing.assert_array_equal(rp[b,:,inside],proto[b,:,inside])


@pytest.mark.parametrize('container',['numpy','torch'])
def test_floor_zero_all_anchor_dense_fallback_precedes_candidate_gather(monkeypatch,container):
    head,proto=fixture();head[:,4]=0
    if container=='torch':
        torch=pytest.importorskip('torch');head,proto=torch.from_numpy(head),torch.from_numpy(proto)
        monkeypatch.setattr(torch,'nonzero',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must skip candidate gather')))
    else:
        monkeypatch.setattr(np,'argwhere',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must skip candidate gather')))
    assert pack_instance_packet(head,proto,floor_hundredths=0,input_shape_hw=(32,48)) is None
