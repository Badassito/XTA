from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from XTA import geometry as g, pta, workers
from XTA.config import resolve_azimuthal_view_requests, resolve_tilted_view_groups
from XTA.lta_config import parse_lta_args
from XTA.lta_inputs import LtaInputDiscovery, LtaVolumeSpec, SourceRole, VolumeClass
from XTA.lta_rendering import build_lta_rendered_view
from XTA.lta_runtime import build_lta_run_plan
from XTA.unification.geometry_identity import physical_view_recipe
from XTA.unification.runtime import compile_physical_views
from XTA.unification.sampling import raster_plan_from_spawn_spec, raster_plan_spawn_spec


def _physical(*, shape=(16, 16, 16), sweep=None):
    return compile_physical_views(
        t_dim=shape[0], height=shape[1], width=shape[2],
        cartesian_views=("transverse",) if sweep is None else (), tilted_groups=(),
        azimuthal_requests=() if sweep is None else resolve_azimuthal_view_requests(f"transverse:{sweep}"),
    ).views[0]


def _item(physical, root):
    view = g.expand_views_into_tta_variants((physical,), (0,))[0]
    job = g.build_aug_job_for_variant(view, 16, root)
    return view, job, g.build_fullframe_raster_plan(view, job)


def _task(plan, view, job, *, kind="fullframe"):
    return dict(kind=kind, view=view, job=job, channel_format="gray", raster_plan_spec=raster_plan_spawn_spec(plan))


class GeometryIdentityProvenanceTests(unittest.TestCase):
    def test_public_sweep_changes_pixels_digest_and_worker_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [_item(_physical(sweep=step), root) for step in (30, 31)]
            volume = np.random.default_rng(777).integers(0, 256, (16, 16, 16), dtype=np.uint8)
            frames = [g.render_fullframe_frame_for_job(volume_rgb=volume, view=v, job=j, frame_idx=1)
                      for v, j, plan in items]
            self.assertGreater(np.count_nonzero(frames[0] != frames[1]), 0)
            self.assertNotEqual(items[0][2].digest, items[1][2].digest)
            for view, job, plan in items:
                rebuilt = g.build_fullframe_raster_plan(view, job)
                accepted = workers._canonical_raster_plan_for_task(_task(rebuilt, view, job))
                self.assertEqual(accepted.digest, plan.digest)
            with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                workers._canonical_raster_plan_for_task(_task(items[0][2], items[1][0], items[1][1]))

    def test_public_source_shape_and_actual_affine_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = _item(_physical(), root)
            b = _item(_physical(shape=(16, 20, 18)), root)
            self.assertNotEqual(a[2].digest, b[2].digest)
            with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                workers._canonical_raster_plan_for_task(_task(a[2], b[0], b[1]))
            forward = a[1].aff.M_src_to_out.copy()
            forward[0, 2] += np.float32(1)
            transform = np.eye(3, dtype=np.float64)
            transform[:2] = forward
            affine = dataclasses.replace(a[1].aff, M_src_to_out=forward,
                                         M_out_to_src=np.linalg.inv(transform)[:2].astype(np.float32))
            job = dataclasses.replace(a[1], aff=affine)
            changed = g.build_fullframe_raster_plan(a[0], job)
            self.assertNotEqual(a[2].digest, changed.digest)
            volume = np.random.default_rng(8).integers(0, 256, (16, 16, 16), dtype=np.uint8)
            before = g.render_fullframe_frame_for_job(volume_rgb=volume, view=a[0], job=a[1], frame_idx=1)
            after = g.render_fullframe_frame_for_job(volume_rgb=volume, view=a[0], job=job, frame_idx=1)
            self.assertGreater(np.count_nonzero(before != after), 0)
            with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                workers._canonical_raster_plan_for_task(_task(a[2], a[0], job))

    def test_dense_tile_binds_physical_sweep_and_actual_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiles = []
            for step in (30, 31):
                view, job, plan = _item(_physical(sweep=step), root)
                tile = g.build_dense_tile_jobs_for_aug(view, job, g.TileConfig(8, 8, "tiles8"), 8, root)[0]
                plan = g.build_dense_tile_raster_plan(view, tile)
                tiles.append((view, tile, plan))
                accepted = workers._canonical_raster_plan_for_task(_task(plan, view, tile, kind="tile"))
                self.assertEqual(accepted.digest, plan.digest)
            self.assertNotEqual(tiles[0][2].digest, tiles[1][2].digest)
            with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                workers._canonical_raster_plan_for_task(_task(tiles[0][2], tiles[1][0], tiles[1][1], kind="tile"))
            matrix = tiles[0][1].M_out_to_src.copy()
            matrix[0, 2] += 1
            changed = dataclasses.replace(tiles[0][1], M_out_to_src=matrix)
            self.assertNotEqual(g.build_dense_tile_raster_plan(tiles[0][0], changed).digest, tiles[0][2].digest)

    def test_plan_snapshots_mutable_matrices_and_ignores_presentation_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            view, job, plan = _item(_physical(), Path(temporary))
            old_record = plan.canonical_record()
            job.aff.M_src_to_out[0, 2] += 1
            self.assertEqual(plan.canonical_record(), old_record)
            self.assertNotEqual(g.build_fullframe_raster_plan(view, job).digest, plan.digest)
            cosmetic = dataclasses.replace(view, display_name="changed display", summary_family="changed summary",
                                             azimuthal_request_token="different request spelling")
            self.assertEqual(physical_view_recipe(view), physical_view_recipe(cosmetic))
            self.assertEqual(g.build_fullframe_raster_plan(view, job).digest,
                             g.build_fullframe_raster_plan(cosmetic, job).digest)

    def test_historical_incomplete_plan_keeps_hash_but_fails_new_task_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            view, job, plan = _item(_physical(sweep=30), Path(temporary))
            historical = dataclasses.replace(plan, metadata={key: value for key, value in plan.metadata.items()
                                                            if key not in {"physical_geometry", "affine_geometry"}})
            restored = raster_plan_from_spawn_spec(raster_plan_spawn_spec(historical))
            self.assertEqual(restored.digest, historical.digest)
            self.assertEqual(restored.canonical_record(), historical.canonical_record())
            self.assertNotEqual(historical.digest, plan.digest)
            with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                workers._canonical_raster_plan_for_task(_task(historical, view, job))

    def test_historical_high_precision_record_retains_recorded_hash(self):
        fixture = Path(__file__).parent / "fixtures" / "legacy_high_precision_raster_plan.json"
        spec = json.loads(fixture.read_text(encoding="utf-8"))
        restored = raster_plan_from_spawn_spec(spec)
        self.assertEqual(restored.digest, spec["plan_digest"])
        self.assertEqual(restored.digest, "69bbc9538d720c46a63f29e1f8760cfccaae76cd47b9ffdaf263f3864a8e3fbd")
        self.assertEqual(restored.canonical_record(), spec["plan"])
        self.assertEqual(restored.in_plane_variant.variant_id, "a12p3457")
        with tempfile.TemporaryDirectory() as temporary:
            physical = _physical()
            view = g.expand_views_into_tta_variants((physical,), (12.345671,))[0]
            job = g.build_aug_job_for_variant(view, 16, Path(temporary))
            current = g.build_fullframe_raster_plan(view, job)
            self.assertNotEqual(current.digest, restored.digest)
            with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                workers._canonical_raster_plan_for_task(_task(restored, view, job))
        spec["plan"]["in_plane_variant"]["variant_id"] = "a12p3456"
        spec["plan_digest"] = hashlib.sha256(json.dumps(spec["plan"], sort_keys=True, separators=(",", ":"),
                                                        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "historical raster-plan angle identity"):
            raster_plan_from_spawn_spec(spec)

    def test_recipe_binds_tilt_and_compact_exact_trajectory(self):
        view = compile_physical_views(
            t_dim=8, height=8, width=8, cartesian_views=(), azimuthal_requests=(),
            tilted_groups=resolve_tilted_view_groups("transverse:12.345671:horizontal"),
        ).views[0]
        self.assertNotEqual(physical_view_recipe(view), physical_view_recipe(dataclasses.replace(view, tilt_angle_deg=12.345672)))
        base = _physical(sweep=30)
        angles = np.arange(10000, dtype=np.float64)
        extended = dataclasses.replace(base, azimuths_deg=angles)
        record = physical_view_recipe(extended)
        self.assertLess(len(json.dumps(record)), 1500)
        angles[-1] = np.nextafter(angles[-1], np.inf)
        self.assertNotEqual(record, physical_view_recipe(extended))
        self.assertEqual(record["azimuthal"]["angles_deg"]["count"], 10000)

    def test_recipe_resolves_legacy_orientation_fallbacks(self):
        transverse = g.ViewInfo("transverse", 8, 8, 8, "clamp", full_t=8, full_h=8, full_w=8)
        sagittal = dataclasses.replace(transverse, name="sagittal")
        self.assertNotEqual(physical_view_recipe(transverse), physical_view_recipe(sagittal))
        base = _physical(sweep=30)
        implicit = dataclasses.replace(base, azimuthal_base_view="", tilt_base_view="")
        explicit = dataclasses.replace(base, azimuthal_base_view=" Transverse ", tilt_base_view="")
        self.assertEqual(physical_view_recipe(implicit), physical_view_recipe(explicit))

    def test_pta_fullframe_and_tiles_bind_physical_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            plans = []
            for step in (30, 31):
                view = pta.adapt_shared_view(_physical(sweep=step))
                affine = pta.build_affine(view.src_w, view.src_h, 0, view.pad_mode, 16, shared_view=view.shared_view)
                plan = pta.build_render_plan(
                    view=view, aff=affine, tag="a0", out_dir=Path(temporary), stem="probe",
                    tile_configs=(pta.TileConfig(8, 8, "tiles8"),), save_overlay=False,
                    imgsz=8, label_enabled=False, publish_images=False, publish_labels=False,
                )
                plans.append(plan)
            self.assertNotEqual(plans[0].canonical_plan.digest, plans[1].canonical_plan.digest)
            self.assertNotEqual(plans[0].tile_layout[0].canonical_plan.digest, plans[1].tile_layout[0].canonical_plan.digest)

    def test_lta_reference_builder_binds_recipe_and_actual_affine(self):
        with tempfile.TemporaryDirectory() as temporary:
            volume = np.random.default_rng(7).integers(0, 256, (16, 16, 16), dtype=np.uint8)
            rendered = []
            for step in (30, 31):
                view, job, plan = _item(_physical(sweep=step), Path(temporary))
                rendered.append(build_lta_rendered_view(volume, view, temp_dir=Path(temporary), output_size=16))
            self.assertNotEqual(rendered[0].raster_plan.digest, rendered[1].raster_plan.digest)
            self.assertGreater(np.count_nonzero(rendered[0].render_frame_rgb(1) != rendered[1].render_frame_rgb(1)), 0)
            self.assertIn("affine_geometry", rendered[0].raster_plan.metadata)

    def test_lta_production_planner_binds_stock_compiled_sweep(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"planning-only-placeholder")
            volume = LtaVolumeSpec(
                source_role=SourceRole.TARGET, source_root=root, volume_id="probe", stem="probe",
                kind="sequence", media=(), video_path=None, video_sha256=None, video_identity_sha256=None,
                annotations=(), volume_class=VolumeClass.PARTIALLY_LABELED, encoded_indices=tuple(range(16)),
                index_origin=0, frame_count=16, width=16, height=16,
            )
            discovery = LtaInputDiscovery(root, (volume,), (), (), (), ())
            views = []
            for step in (30, 31):
                config = parse_lta_args(["--input", str(root), "--output", str(root / "out"),
                                         "--model", str(checkpoint), "--device", "0",
                                         "--enable_azimuthal", f"transverse:{step}", "--angle", "0"])
                plan = build_lta_run_plan(config, discovery_fn=lambda *a, **kw: discovery, run_id="probe")
                views.append(plan.volumes[0].runtime_views[0])
            self.assertNotEqual(views[0].runtime_view.azimuths_deg, views[1].runtime_view.azimuths_deg)
            self.assertNotEqual(views[0].raster_plan_digest, views[1].raster_plan_digest)


if __name__ == "__main__":
    unittest.main()
