"""SAM/component publication resolves nonlinear geometry on the final grid once."""
import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.config import TiltedViewGroup
from XTA.interpolation import RawBBoxMaskStore


@pytest.mark.parametrize('base',('transverse','sagittal','coronal'))
def test_reduced_tilted_component_projects_directly_to_final_source_grid(tmp_path,monkeypatch,base):
    view=geometry.get_view_infos(5,7,9,cartesian_views=(),
        tilt_groups=(TiltedViewGroup((base,),(23.,),('vertical',)),))[0]
    source=np.ones((view.num_slices,3,3),np.uint8)
    final_shape=(8,11,13)
    calls=[]
    def native_project(array,actual_view,path,desc,**kwargs):
        assert actual_view is view
        calls.append(kwargs['out_shape_tyx'])
        return np.ones(final_shape,np.uint8)
    monkeypatch.setattr(assembly,'project_view_volume_to_orthogonal_volume',native_project)
    monkeypatch.setattr(assembly,'nrrd_layer_sink',lambda:None)
    old=assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(final_shape)
    try:
        monkeypatch.setattr(assembly,'delayed_native_expansion_enabled',lambda:True)
        ref=assembly.materialize_nrrd_view_layer(source,model_name='sam-source',view=view,
            source='fullframe',mask_kind='bridge',stage='sam_selected_forward',temp_dir=tmp_path,
            known_has_foreground=True,submit_to_sink=False,force_path_backed_store=True)
        assert calls==[final_shape]
        assert ref.shape==final_shape
        store=RawBBoxMaskStore.open(ref.path)
        try:
            assert store.shape==final_shape
            assert store.meta['projection_geometry_contract']=='xta.native_destination_pull/1'
            assert store.meta['projection_output_shape_tyx']==list(final_shape)
            assert store.meta['projection_payload_fusion']=='tilted_native_destination_pull'
            assert all(np.all(store.decode_slice(z)==1) for z in range(final_shape[0]))
        finally:store.close()
    finally:assembly.set_final_source_output_shape(old)
