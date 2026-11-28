"""CPU-only failure ownership checks for worker padding backings."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import workers
from XTA.geometry import ViewInfo
from XTA.runtime import wait_for_retired_memmap_unlinks


def test_later_seam_parity_failure_retires_both_unpublished_groups(tmp_path, monkeypatch):
    from XTA import assembly, confidence_tiles, pipeline
    from XTA.confidence_evidence import configure_confidence_evidence
    from XTA.geometry import azimuthal_batch_padding_frame_specs
    from XTA.tile_pipeline import consume_tile_files

    predicate = '_submit_azimuthal_padding_tile_cleanups'
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
    function = next(node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == predicate)
    actual_write = assembly._write_raw_bbox_payload_store
    paths = []
    injected = OSError('second seam encode failed')

    def write(**kwargs):
        path = kwargs['store_dir']
        paths.append(path)
        if len(paths) == 2:
            assert paths[0].is_dir()
            path.mkdir()
            (path / 'chunks.bin').write_bytes(b'partial')
            raise injected
        return actual_write(**kwargs)

    monkeypatch.setattr(assembly, '_write_raw_bbox_payload_store', write)
    mask_path, conf_path = tmp_path / 'mask.dat', tmp_path / 'conf.dat'
    view = ViewInfo('azimuthal', 7, 3, 10, 'clamp', family='azimuthal')
    specs = azimuthal_batch_padding_frame_specs(view, view.num_slices, 16)
    assert {spec.mirror_azimuthal_u for spec in specs} == {False, True}
    mask = np.ones((len(specs), 3, 4), np.uint8)
    mask_path.write_bytes(mask.tobytes())
    conf_path.write_bytes((mask * np.uint8(154)).tobytes())
    job = SimpleNamespace(tile_id='seam', config_id='c', parent_crop=(0, 3, 1, 5))
    configure_confidence_evidence(tmp_path / 'out', enabled=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            namespace = {**vars(pipeline), 'tile_staging': SimpleNamespace(path=tmp_path),
                'consume_tile_files': consume_tile_files,
                'args': SimpleNamespace(min_conf=0, min_radius=0),
                'angle_variant_streaming_cleanup_active': False,
                'tile_result_retirement_executor': executor, 'tile_result_retirement_slice_workers': 1,
                'tile_cleanup_futures': {}}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(pipeline.__file__), 'exec'),
                namespace)
            namespace[predicate](model_name='m', view=view, tile_job=job,
                padding_shape=mask.shape, expected_frames=specs,
                padding_mask_path=mask_path, padding_conf_path=conf_path,
                padding_frames=[dict(ordinal=spec.azimuthal_padding_ordinal,
                                     destination=spec.global_destination_index,
                                     mirror_azimuthal_u=spec.mirror_azimuthal_u) for spec in specs],
                threshold_plane_shape=(3, 10))
            future, = namespace['tile_cleanup_futures']
            with pytest.raises(OSError, match='second seam encode failed') as caught:
                future.result(timeout=5)
            assert caught.value is injected
        assert len(paths) == 2 and not any(path.exists() for path in paths)
        assert not mask_path.exists() and not conf_path.exists()
        assert not confidence_tiles._RAW and not any((tmp_path / 'tile_confidence').iterdir())
    finally:
        configure_confidence_evidence(None, enabled=False)


class WorkerPaddingRetirementTests(unittest.TestCase):
    @staticmethod
    def _task(root: Path) -> dict[str, object]:
        return {
            "view": ViewInfo("azimuthal", 1, 2, 2, "clamp"),
            "job": object(), "job_id": "a0", "task_id": 1,
            "kind": "fullframe", "out_size": 2, "processing_shape": (1, 2, 2),
            "source_volume_path": str(root / "input.dat"),
            "result_conf_path": str(root / "confidence.dat"),
            "azimuthal_padding_dir": str(root / "padding"),
        }

    @staticmethod
    def _patch_worker_memmap(memmap_fn):
        return (
            mock.patch.object(workers, "AugJob", object),
            mock.patch.object(workers, "_canonical_raster_plan_for_task", return_value=None),
            mock.patch.object(workers, "_canonical_image_sink_for_task", return_value=None),
            mock.patch.object(workers, "view_processing_volume_shape", return_value=(1, 2, 2)),
            mock.patch.object(workers, "azimuthal_batch_padding_count", return_value=1),
            mock.patch.object(workers, "np", SimpleNamespace(memmap=memmap_fn, uint8=np.uint8)),
        )

    def test_second_padding_allocation_failure_defers_first_unlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            opened = []

            def open_padding(*args, **kwargs):
                if opened:
                    raise OSError("intentional confidence padding failure")
                mapping = np.memmap(*args, **kwargs)
                opened.append(mapping)
                return mapping

            patches = self._patch_worker_memmap(open_padding)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaisesRegex(OSError, "intentional confidence padding failure"):
                    workers.run_prediction_volume_in_worker(
                        None, SimpleNamespace(input_channels=1, batch=1), self._task(root)
                    )

            path = next((root / "padding").glob("*.mask.u8.dat"))
            retained_view = opened[0][0]
            self.assertEqual(int(retained_view[0, 0]), 0)
            self.assertTrue(path.exists())
            opened.clear()
            self.assertTrue(path.exists())
            del retained_view
            gc.collect()
            wait_for_retired_memmap_unlinks(path=path)
            self.assertFalse(path.exists())

    def test_preexisting_padding_file_is_neither_overwritten_nor_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            padding_dir = root / "padding"
            padding_dir.mkdir()
            path = padding_dir / "fullframe-azimuthal-a0-task1.mask.u8.dat"
            path.write_bytes(b"original")
            patches = self._patch_worker_memmap(mock.Mock(side_effect=AssertionError("unexpected memmap")))
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaises(FileExistsError):
                    workers.run_prediction_volume_in_worker(
                        None, SimpleNamespace(input_channels=1, batch=1), self._task(root)
                    )
            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
