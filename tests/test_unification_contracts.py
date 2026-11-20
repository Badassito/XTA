from __future__ import annotations

import os
import pickle
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from XTA.config import ChannelFormat, resolve_channel_format
from XTA.unification import (
    BackendSamplingImplementation,
    ChannelVariant,
    DataRole,
    ForwardSamplingPolicy,
    FrameAddress,
    InPlaneVariant,
    PipelineMode,
    RasterPlan,
    RenderRequestBatch,
    RenderItem,
    TileLayout,
    channel_layout_from_format,
    expand_channel_variants,
    expand_view_variants,
    build_forward_raster_plan,
    forward_sampling_execution_record,
    forward_sampling_policy,
    require_forward_sampling,
    resolve_in_plane_variants,
)


ROOT = Path(__file__).resolve().parents[1]


def make_policy() -> ForwardSamplingPolicy:
    roles = tuple(DataRole)
    return ForwardSamplingPolicy(
        policy_id="tta-v18-forward",
        coordinate_convention="half_pixel",
        stage_order=("physical_view", "in_plane_affine", "output_raster"),
        role_kernels=(
            (DataRole.INTENSITY, "bilinear"),
            (DataRole.CATEGORICAL_GROUND_TRUTH, "nearest"),
            (DataRole.PREDICTION, "nearest"),
            (DataRole.PRESENTATION, "bilinear"),
        ),
        role_boundaries=tuple((role, "constant_zero") for role in roles),
        backend_implementations=(
            BackendSamplingImplementation(
                backend="cpu",
                implementation="reference-opencv",
                roles=roles,
                exact=True,
            ),
            BackendSamplingImplementation(
                backend="cuda",
                implementation="texture-hardware-linear",
                roles=(DataRole.INTENSITY, DataRole.PRESENTATION),
                absolute_tolerance=1.0,
            ),
        ),
    )


class UnificationContractTests(unittest.TestCase):
    def test_package_import_is_dependency_light(self) -> None:
        program = (
            "import sys; import XTA.unification; "
            "forbidden={'cv2','scipy','torch','cupy','openvino','ultralytics','jax','numpy'}; "
            "loaded={name.split('.')[0] for name in sys.modules}; "
            "assert not (forbidden & loaded), forbidden & loaded"
        )
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", program],
            cwd=ROOT,
            env={**os.environ, "YOLO_TTA_TELEMETRY": "0"},
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_channel_layout_from_format_copies_resolved_fields(self) -> None:
        layout = channel_layout_from_format(resolve_channel_format("c5s2"))
        self.assertEqual(layout.token, "C5S2")
        self.assertEqual(layout.offsets, (-4, -2, 0, 2, 4))
        # The copy keeps contract validation instead of trusting its input.
        with self.assertRaisesRegex(ValueError, "odd"):
            channel_layout_from_format(ChannelFormat("C4S1", "custom", 4, 1, (-2, -1, 1, 2)))

    def test_tta_emits_only_ascending_channel_order(self) -> None:
        layout = channel_layout_from_format(resolve_channel_format("C5S2"))
        variants = expand_channel_variants(PipelineMode.TTA, layout)
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].direction, "ascending")
        self.assertEqual(variants[0].offsets, (-4, -2, 0, 2, 4))

        lta_variants = expand_channel_variants(PipelineMode.LTA, layout)
        self.assertEqual(len(lta_variants), 1)
        self.assertEqual(lta_variants[0].direction, "ascending")

    def test_pta_adds_reversed_order_only_when_distinct(self) -> None:
        variants = expand_channel_variants(
            "pta", channel_layout_from_format(resolve_channel_format("C5S2"))
        )
        self.assertEqual([variant.direction for variant in variants], ["ascending", "reversed"])
        self.assertEqual(variants[1].offsets, (4, 2, 0, -2, -4))
        for symmetric in ("gray", "RGB", "C1S3"):
            with self.subTest(symmetric=symmetric):
                layout = channel_layout_from_format(resolve_channel_format(symmetric))
                self.assertEqual(len(expand_channel_variants("pta", layout)), 1)

    def test_pta_internal_identity_is_not_a_physical_view(self) -> None:
        self.assertEqual(resolve_in_plane_variants("pta"), (InPlaneVariant(0.0),))
        self.assertEqual(expand_view_variants("pta", []), ())
        variants = expand_view_variants("pta", ["transverse", "azimuthal"])
        self.assertEqual(len(variants), 2)
        self.assertEqual([variant.physical_view for variant in variants], ["transverse", "azimuthal"])
        self.assertTrue(all(variant.runtime_view == variant.physical_view for variant in variants))
        with self.assertRaisesRegex(ValueError, "invalid in PTA"):
            resolve_in_plane_variants("pta", [0])

    def test_tta_view_variants_wrap_existing_expansion(self) -> None:
        calls: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
        fake_geometry = ModuleType("XTA.geometry")

        def fake_existing_expansion(
            physical_views: tuple[str, ...],
            angles: tuple[float, ...],
        ) -> list[str]:
            calls.append((tuple(physical_views), tuple(angles)))
            return [f"{view}__tta_a{angle:g}" for view in physical_views for angle in angles]

        fake_geometry.expand_views_into_tta_variants = fake_existing_expansion  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"XTA.geometry": fake_geometry}):
            variants = expand_view_variants("tta", ["transverse"], [0, 120])

        self.assertEqual(calls, [(('transverse',), (0.0, 120.0))])
        self.assertEqual([variant.in_plane_variant.angle_deg for variant in variants], [0.0, 120.0])
        self.assertEqual(
            [variant.runtime_view for variant in variants],
            ["transverse__tta_a0", "transverse__tta_a120"],
        )

        with mock.patch.dict(sys.modules, {"XTA.geometry": fake_geometry}):
            lta_variants = expand_view_variants("lta", ["transverse"], [0, 120])
        self.assertEqual(
            [variant.runtime_view for variant in lta_variants],
            ["transverse__tta_a0", "transverse__tta_a120"],
        )

    def test_sampling_policy_declares_backend_role_coverage(self) -> None:
        policy = make_policy()
        self.assertEqual(policy.kernel_for("categorical_ground_truth"), "nearest")
        self.assertEqual(
            policy.implementation_for("CUDA", DataRole.INTENSITY).implementation,
            "texture-hardware-linear",
        )
        with self.assertRaises(KeyError):
            policy.implementation_for("cuda", DataRole.CATEGORICAL_GROUND_TRUTH)

    def test_production_sampling_policy_and_plan_are_mechanically_bound(self) -> None:
        policy = forward_sampling_policy()
        self.assertIs(policy, forward_sampling_policy())
        self.assertEqual(len(policy.digest), 64)
        self.assertEqual(
            require_forward_sampling("cpu", "intensity").backend,
            "cpu",
        )
        self.assertEqual(
            require_forward_sampling("cuda", "categorical_ground_truth").backend,
            "cuda",
        )

        plan = build_forward_raster_plan(
            mode="pta",
            physical_view_id="azimuthal_transverse",
            angle_deg=0,
            channel_token="C3S1",
            channel_kind="custom",
            channel_count=3,
            channel_stride=1,
            channel_offsets=(1, 0, -1),
            channel_direction="reverse",
            output_shape=(24, 32),
            tile_size=16,
            tile_stride=8,
            metadata={"runtime_job_id": "probe"},
        )
        self.assertIs(plan.sampling_policy, policy)
        self.assertEqual(plan.channel_variant.direction, "reversed")
        self.assertEqual(plan.channel_variant.layout.offsets, (-1, 0, 1))
        record = forward_sampling_execution_record(
            (("cpu", "intensity"), ("cpu", "categorical_ground_truth"))
        )
        self.assertEqual(record["policy_digest"], policy.digest)
        self.assertEqual(len(record["selected_implementations"]), 2)

    def test_raster_plan_digest_is_canonical_and_sensitive(self) -> None:
        layout = channel_layout_from_format(resolve_channel_format("C3S1"))
        channel = ChannelVariant(layout=layout)
        common = dict(
            mode=PipelineMode.PTA,
            physical_view_id="transverse",
            in_plane_variant=InPlaneVariant(0),
            channel_variant=channel,
            sampling_policy=make_policy(),
            output_shape=(64, 64),
        )
        first = RasterPlan(**common, metadata={"b": [2, 3], "a": 1})
        second = RasterPlan(**common, metadata={"a": 1, "b": [2, 3]})
        changed = RasterPlan(**common, metadata={"a": 1, "b": [2, 4]})
        self.assertEqual(first.digest, second.digest)
        self.assertNotEqual(first.digest, changed.digest)
        with self.assertRaises(TypeError):
            first.metadata["new"] = True  # type: ignore[index]

    def test_render_items_and_batches_have_stable_ids(self) -> None:
        layout = channel_layout_from_format(resolve_channel_format("gray"))
        plan = RasterPlan(
            mode="pta",
            physical_view_id="transverse",
            in_plane_variant=InPlaneVariant(),
            channel_variant=ChannelVariant(layout),
            sampling_policy=make_policy(),
            output_shape=(32, 32),
        )
        item_a = RenderItem(plan, "intensity", FrameAddress(3, mirror_u=True))
        item_b = RenderItem(plan, DataRole.INTENSITY, FrameAddress(3, mirror_u=True))
        self.assertEqual(item_a.item_id, item_b.item_id)
        batch_a = RenderRequestBatch("pta", (item_a,))
        batch_b = RenderRequestBatch(PipelineMode.PTA, (item_b,))
        self.assertEqual(batch_a.batch_id, batch_b.batch_id)
        restored = pickle.loads(pickle.dumps(batch_a))
        self.assertEqual(restored.batch_id, batch_a.batch_id)
        self.assertEqual(restored.items[0].plan.digest, plan.digest)
        self.assertEqual(RenderRequestBatch("pta", ()).items, ())
        with self.assertRaisesRegex(ValueError, "does not match"):
            RenderRequestBatch("tta", (item_a,))

    def test_render_item_identity_is_lazy_and_unchanged(self) -> None:
        import dataclasses
        from XTA.unification.contracts import _digest_record
        plan = RasterPlan(
            mode="pta",
            physical_view_id="transverse",
            in_plane_variant=InPlaneVariant(),
            channel_variant=ChannelVariant(channel_layout_from_format(resolve_channel_format("gray"))),
            sampling_policy=make_policy(),
            output_shape=(32, 32),
        )
        metadata = {"source": {"index": 4, "tags": ["a", "b"]}, "weight": 0.5}
        item = RenderItem(plan, "intensity", FrameAddress(3, mirror_u=True), metadata)
        self.assertNotIn("item_id", item.__dict__)
        self.assertNotIn("item_id", [entry.name for entry in dataclasses.fields(RenderItem)])
        before_access = pickle.loads(pickle.dumps(item))
        expected = _digest_record({"plan_digest": plan.digest, "data_role": "intensity",
            "frame_address": {"index": 3, "mirror_u": True},
            "metadata": {"source": {"index": 4, "tags": ["a", "b"]}, "weight": 0.5}})
        self.assertEqual(item.item_id, expected)
        self.assertIn("item_id", item.__dict__)
        self.assertEqual(before_access.item_id, expected)
        self.assertEqual(pickle.loads(pickle.dumps(item)).item_id, expected)
        self.assertEqual(item, before_access)
        with self.assertRaises(TypeError):
            RenderItem(object(), "intensity", FrameAddress(3))
        with self.assertRaises((TypeError, ValueError)):
            RenderItem(plan, "intensity", FrameAddress(3), {1: "non-string key"})
        with self.assertRaises(ValueError):
            RenderItem(plan, "intensity", FrameAddress(3), {"value": float("nan")})

    def test_tile_and_frame_contract_validation(self) -> None:
        self.assertEqual(TileLayout(16, 8).layout_id, "tile_16_8")
        with self.assertRaises(ValueError):
            TileLayout(8, 9)
        with self.assertRaises(ValueError):
            FrameAddress(-1)
        with self.assertRaises(TypeError):
            FrameAddress(0, mirror_u=1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
