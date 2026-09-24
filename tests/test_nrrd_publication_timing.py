"""Durability telemetry separates NRRD publication without weakening atomicity."""
from contextlib import contextmanager
from unittest import mock

import pytest

from XTA import outputs


class Recorder:
    def __init__(self): self.phases=[]
    @contextmanager
    def span(self,name):
        self.phases.append(name)
        yield


def test_nrrd_publication_reports_durability_and_rename_in_order(tmp_path):
    stage,target=tmp_path/'stage',tmp_path/'target.seg.nrrd'
    stage.write_bytes(b'complete payload')
    record=Recorder()
    with mock.patch.object(outputs,'runtime_telemetry',return_value=record):
        outputs._publish_staged_file_atomically(stage,target)
    assert target.read_bytes()==b'complete payload' and not stage.exists()
    assert record.phases==['nrrd.publication.file_durability','nrrd.publication.rename',
                           'nrrd.publication.directory_durability']


def test_file_sync_failure_preserves_existing_output_and_original_error(tmp_path):
    target=tmp_path/'target.seg.nrrd'; target.write_bytes(b'previous output')
    failure=OSError('injected durability failure')
    record=Recorder()
    with mock.patch.object(outputs,'runtime_telemetry',return_value=record), \
         mock.patch.object(outputs.os,'fsync',side_effect=failure):
        with pytest.raises(OSError) as caught:
            with outputs._same_directory_atomic_output(target) as stage:
                stage.write_bytes(b'new output')
    assert caught.value is failure
    assert target.read_bytes()==b'previous output' and not stage.exists()
    assert record.phases==['nrrd.publication.file_durability']


def test_other_outputs_do_not_initialize_nrrd_telemetry(tmp_path):
    stage,target=tmp_path/'stage',tmp_path/'target.bin'
    stage.write_bytes(b'complete payload')
    with mock.patch.object(outputs,'runtime_telemetry',side_effect=AssertionError('NRRD telemetry')):
        outputs._publish_staged_file_atomically(stage,target)
    assert target.read_bytes()==b'complete payload'
