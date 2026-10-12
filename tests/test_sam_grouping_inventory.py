from tools.analyze_sam_grouping_inventory import inspect_index


def test_grouping_interval_bounds_keep_each_objects_full_coverage():
    crop=[0,0,100,100]
    data={'groups':{'g':{'context_bbox_yx':crop}},'runs':{
        'a':{'group_id':'g','direction':'forward','expected_frames':[3,4,5]},
        'b':{'group_id':'g','direction':'forward','expected_frames':[3,4,5]},
        'c':{'group_id':'g','direction':'forward','expected_frames':[3,4,5,6]},
    }}
    row=inspect_index(data,'fixture')
    assert row['tracker_frame_work']==10
    assert row['ideal_unique_crop_frames']==4
    assert row['strict_grouped_frame_step_reduction']==3
    assert row['same_crop_direction_start_varying_end_step_reduction']==6


def test_different_direction_and_seed_frame_do_not_share_recurrent_inventory():
    data={'groups':{'g':{'context_bbox_yx':[0,0,100,100]}},'runs':{
        'a':{'group_id':'g','direction':'forward','expected_frames':[3,4,5]},
        'b':{'group_id':'g','direction':'backward','expected_frames':[5,4,3]},
        'c':{'group_id':'g','direction':'forward','expected_frames':[4,5]},
    }}
    row=inspect_index(data,'fixture')
    assert row['strict_grouped_frame_step_reduction']==0
    assert row['same_crop_direction_start_varying_end_step_reduction']==0
    assert row['ideal_unique_crop_frames']==3


def test_shared_key_is_global_but_worker_copy_identity_stays_separate():
    identities=(set(),set())
    def data(worker):
        return {'groups':{'g':{'context_bbox_yx':[0,0,100,100]}},'runs':{
            'a':{'group_id':'g','direction':'forward','expected_frames':[3],
                 'runtime_receipt':{'image_cache':{'identity_sha256':'pixels','physical_view_id':'view'},
                    'worker_startup_receipt':{'cuda_allocator_quota':{
                        'execution_device_id':0,'worker_index':worker}}}}
        }}
    inspect_index(data(1),'first',identity_sets=identities)
    inspect_index(data(2),'second',identity_sets=identities)
    assert len(identities[0])==1
    assert len(identities[1])==2
