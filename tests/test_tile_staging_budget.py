"""Independent tile output cannot spend the same physical RAM as another native lane."""
from pathlib import Path

import pytest

from XTA.tile_pipeline import TileStaging, tile_result_headroom


@pytest.mark.parametrize('temp_is_ram,output_is_ram', [(False, False), (False, True), (True, False), (True, True)])
def test_staging_uses_only_private_disk_directory(tmp_path, monkeypatch, temp_is_ram, output_is_ram):
    temp, output = tmp_path / 'temp', tmp_path / 'output'
    for parent in (temp, output):
        parent.mkdir()
        (parent / 'keep').write_bytes(b'pre-existing')
    monkeypatch.setattr('XTA.runtime.path_is_memory_backed',
                        lambda path: {temp: temp_is_ram, output: output_is_ram}[Path(path)])
    if temp_is_ram and output_is_ram:
        with pytest.raises(RuntimeError, match='both are memory-backed'):
            TileStaging(temp, output)
    else:
        stage = TileStaging(temp, output)
        assert stage.path.parent == (output if temp_is_ram else temp)
        (stage.path / 'private').write_bytes(b'x')
        stage.close()
        stage.close()
    assert [p.name for p in temp.iterdir()] == ['keep']
    assert [p.name for p in output.iterdir()] == ['keep']


@pytest.mark.parametrize('physical,policy,cap,expected', [
    (160, 0, 0, 20), (160, 135, 0, 15), (160, 135, 10, 10),
    (140, 0, 0, 0), (130, 0, 0, 0), (160, 190, 0, 0),
])
def test_raw_tile_admission_protects_overlapping_non_tile_commitments(physical, policy, cap, expected):
    assert tile_result_headroom(physical, shared_reserve=90, transient_reserve=40,
                                policy_non_tile_reserve=policy, retained_publication=10, anon_cap=cap) == expected
