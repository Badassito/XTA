"""YOLO polygon eligibility must never masquerade as proof of a zero source mask."""
from types import SimpleNamespace
from unittest import mock
import os

import numpy as np
import pytest

from XTA import pta_workers as workers
from XTA.pta_dataset import OutputCandidate
from tests.test_pta_gpu_publication import FakeTorch, FakeTensor


def candidate(*, foreground=False, augmentation_index=0):
    return OutputCandidate(order=0, volume_name='sample', parent_view_tag='Transverse',
        output_tag='Transverse', item_key='full', frame_idx=0, is_tile=False,
        label_enabled=True, foreground=foreground, augmentation_index=augmentation_index,
        augmentation_seed=7 if augmentation_index else None,
        augmentation_tag='AbCd1234EfGh5678' if augmentation_index else None)


def publish(tmp_path, mask, cand, *, source_nonempty, save_binary=False):
    torch = FakeTorch()
    with mock.patch.dict(workers._WORKER_STATIC, dict(
        out_dir=tmp_path, split_active=False, image_format='png', save_images=False,
        save_labels=True, save_binary=save_binary, save_semantic=False,
    ), clear=True):
        return workers._publish_gpu_policy_batch(
            runtime={'torch':torch,'device_id':0},
            batch_images=FakeTensor(torch,np.zeros((1,1,*mask.shape),np.uint8)),
            batch_masks=FakeTensor(torch,mask[None]), candidates=[cand],
            output_size=mask.shape, channel_kind='gray', local_warnings=workers.WarningLog(),
            source_mask_nonempty=(source_nonempty,),
        )


def test_original_tiny_nonexportable_mask_is_not_synthesis_and_binary_is_preserved(tmp_path):
    mask = np.zeros((8,8),np.uint8)
    mask[3,3] = 1
    assert publish(tmp_path,mask,candidate(),source_nonempty=True,save_binary=True)==(1,{})
    assert next((tmp_path/'labels').glob('*.txt')).read_text()==''
    from XTA.pta_binary import candidate_binary_output_path
    from XTA import pta
    output=candidate_binary_output_path(tmp_path,candidate(),split_active=False)
    np.testing.assert_array_equal(pta.cv2.imread(str(output),pta.cv2.IMREAD_UNCHANGED)>0,mask>0)


def test_tiny_source_can_legitimately_grow_into_an_exportable_augmented_polygon(tmp_path):
    grown=np.zeros((8,8),np.uint8)
    grown[1:7,1:7]=1
    cand=candidate(augmentation_index=1)
    assert not cand.foreground
    assert publish(tmp_path,grown,cand,source_nonempty=True)==(1,{})
    assert next((tmp_path/'labels').glob('*.txt')).read_text().strip()


@pytest.mark.parametrize('planned_foreground',[False,True])
def test_truly_empty_source_still_rejects_synthesized_foreground(tmp_path,planned_foreground):
    mask=np.ones((8,8),np.uint8)
    with pytest.raises(RuntimeError,match='synthesized a mask'):
        publish(tmp_path,mask,candidate(foreground=planned_foreground,augmentation_index=1),source_nonempty=False)


def test_foreground_copy_that_becomes_empty_still_drops(tmp_path):
    assert publish(tmp_path,np.zeros((8,8),np.uint8),
        candidate(foreground=True,augmentation_index=1),source_nonempty=True)==(0,{'all':1})


def test_true_background_copy_is_kept(tmp_path):
    assert publish(tmp_path,np.zeros((8,8),np.uint8),
        candidate(augmentation_index=1),source_nonempty=False)==(1,{})


def test_source_occupancy_snapshot_precedes_mutation_and_respects_semantic_unknowns():
    tiny=np.zeros((8,8),np.uint8)
    tiny[3,3]=1
    empty=np.zeros_like(tiny)
    batch=[SimpleNamespace(mask=tiny,semantic_coverage=None,candidates=(0,1)),
           SimpleNamespace(mask=tiny,semantic_coverage=empty,candidates=(2,)),
           SimpleNamespace(mask=empty,semantic_coverage=None,candidates=(3,))]
    flags=workers._gpu_source_mask_nonempty({'torch':None},batch)
    raw_flags=workers._gpu_source_mask_nonempty({'torch':None},batch,include_coverage=False)
    tiny[:]=0
    empty[:]=1
    assert flags==(True,True,False,False)
    assert raw_flags==(True,True,True,False)


def test_publisher_requires_correctly_sized_source_truth(tmp_path):
    torch=FakeTorch()
    with pytest.raises(ValueError,match='source-mask occupancy'):
        workers._publish_gpu_policy_batch(runtime={'torch':torch,'device_id':0},
            batch_images=FakeTensor(torch,np.zeros((1,1,8,8),np.uint8)),
            batch_masks=FakeTensor(torch,np.zeros((1,8,8),np.uint8)),
            candidates=[candidate()], output_size=(8,8),channel_kind='gray',
            local_warnings=workers.WarningLog(),source_mask_nonempty=())


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_GPU_PUBLICATION')!='1',reason='requires owned GPU qualification slot')
def test_cuda_source_flags_are_immutable_before_inplace_policy_and_cross_publication_stream():
    import torch
    from XTA.pta_gpu_publication import GpuPublicationResources
    runtime={'torch':torch,'device_id':0}
    resources=GpuPublicationResources(runtime,cpu_threads=1)
    results=[]
    producer=torch.cuda.Stream(device=0)
    def consume(*,source_mask_nonempty,publication,**kwargs):
        return source_mask_nonempty.to('cpu').tolist()
    task=resources.task(consume,results.append)
    try:
        with torch.cuda.stream(producer):
            tiny=torch.zeros((7,9),dtype=torch.uint8,device='cuda:0')
            tiny[3,4]=1
            empty=torch.zeros_like(tiny)
            flags=workers._gpu_source_mask_nonempty(runtime,[
                SimpleNamespace(mask=tiny,semantic_coverage=None,candidates=(0,1)),
                SimpleNamespace(mask=tiny,semantic_coverage=empty,candidates=(2,)),
                SimpleNamespace(mask=empty,semantic_coverage=None,candidates=(3,)),
            ])
            images=torch.zeros((4,1,7,9),dtype=torch.uint8,device='cuda:0')
            masks=torch.zeros((4,7,9),dtype=torch.uint8,device='cuda:0')
            with task.reserve(2*(images.numel()+masks.numel())) as reservation:
                task.submit(reservation,dict(batch_images=images,batch_masks=masks,source_mask_nonempty=flags))
            tiny.zero_()
            empty.fill_(1)
        task.close()
        assert results==[[True,True,False,False]]
    finally:
        task.close()
        resources.finalizer.cancel()
        resources.close()
