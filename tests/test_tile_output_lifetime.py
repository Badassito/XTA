"""Retained tile component stores belong to the run through global output retirement."""
import ast
from contextlib import closing
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import pipeline
from XTA.interpolation import RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.tile_pipeline import TileStaging
from XTA.tta_lifecycle import PipelineRunResources


def test_science_spool_closes_only_after_global_readers_release_their_refs():
    # These two closures live inside the orchestration loop. Pin their ownership
    # boundary without creating a detector engine or depending on scheduler timing.
    tree = ast.parse(inspect.getsource(pipeline._main_impl))
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    submit = functions['_submit_tile_config']
    settings, = (node.value for node in submit.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'settings')
    selected = next(kw.value for kw in settings.keywords if kw.arg == 'temp_dir')
    assert ast.unparse(selected) == 'tile_staging.path'
    finalizer = functions['_finalize_selected_run_after_output_close']
    calls = {ast.unparse(node.func): node for node in ast.walk(finalizer) if isinstance(node, ast.Call)}
    close = calls['tile_staging.close']
    assert calls['nrrd_layer_refs.clear'].lineno < close.lineno
    assert [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == 'tile_staging.close'] == [close]
    assert any(kw.arg == 'after_close' and ast.unparse(kw.value) == finalizer.name
               for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == '_close_tta_output_artifacts' for kw in node.keywords)


@pytest.mark.parametrize('fail_output', (False, True))
def test_global_component_reader_settles_before_private_spool_retirement(tmp_path, monkeypatch, fail_output):
    temp, output = tmp_path / 'tmpfs', tmp_path / 'output'
    temp.mkdir()
    output.mkdir()
    sibling = output / 'existing'
    sibling.write_bytes(b'unchanged')
    monkeypatch.setattr('XTA.runtime.path_is_memory_backed', lambda path: Path(path) == temp)
    stage = TileStaging(temp, output)
    assert stage.path.parent == output
    source = np.array([[[0, 1, 0], [1, 1, 0]]], np.uint8)
    path = stage.path / 'component.cvol'
    write_raw_bbox_mask_store(source, path, workers=1, desc='independent source-grid tile')
    events = []

    def read_component():
        with closing(RawBBoxMaskStore.open(path, mmap_payload=True)) as store:
            np.testing.assert_array_equal(store.decode_slice(0), source[0])
        events.append('read')

    original = RuntimeError('reconciliation failed after component read')
    real_close = stage.close

    def close():
        if stage.path.exists():
            events.append('retire')
        real_close()
        if fail_output:
            raise OSError('secondary spool retirement failure')

    monkeypatch.setattr(stage, 'close', close)
    resources = PipelineRunResources()
    resources.track_closeable(stage)
    if fail_output:
        resources.track_output_manager(SimpleNamespace(wait=read_component))
        resources.track_sink(SimpleNamespace(shutdown=read_component))
        with pytest.raises(RuntimeError) as raised:
            try:
                read_component()
                raise original
            finally:
                resources.close(failed=True)
        assert raised.value is original
        assert events == ['read', 'read', 'read', 'retire']
    else:
        read_component()
        artifacts = SimpleNamespace(close=lambda **kwargs: read_component())
        pipeline._close_tta_output_artifacts(artifacts, after_close=stage.close)
        resources.close(failed=False)
        assert events == ['read', 'read', 'retire']
    assert not stage.path.exists()
    assert sibling.read_bytes() == b'unchanged'
