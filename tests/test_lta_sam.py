from __future__ import annotations

import sys
import gzip
import hashlib
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np


class LtaSamBoundaryTests(unittest.TestCase):
    def test_attention_backend_environment_and_explicit_baseline(self) -> None:
        from XTA.lta_sam import resolve_sam_attention_backend

        with mock.patch.dict("os.environ", {"YOLO_TTA_SAM_ATTENTION_BACKEND": "sdpa"}):
            self.assertEqual(resolve_sam_attention_backend((9, 0)), "sdpa")
            self.assertEqual(resolve_sam_attention_backend((9, 0), "fa3_fp8"), "fa3_fp8")
        for value in ("fp8", "", "fallback"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "attention_backend"):
                resolve_sam_attention_backend((9, 0), value)

    def test_fa3_preflight_runs_both_kernel_shapes_and_rejects_bad_outputs(self) -> None:
        from XTA.lta_sam import preflight_sam_fa3

        class Scalar:
            def __init__(self, value):
                self.value = value
            def item(self):
                return self.value
            def all(self):
                return self

        class Tensor:
            is_cuda = True
            def __init__(self, shape, dtype, value=0):
                self.shape, self.dtype, self.value = shape, dtype, value
            def to(self, dtype):
                return Tensor(self.shape, dtype, self.value)
            def get_device(self):
                return 0
            def __eq__(self, value):
                return Scalar(self.value == value)

        cuda = types.SimpleNamespace(get_device_capability=lambda _device: (9, 0),
                                     synchronize=mock.Mock())
        runtime = types.SimpleNamespace(cuda=cuda, Tensor=Tensor, bfloat16="bf16", float8_e4m3fn="fp8",
            __version__="test-torch", version=types.SimpleNamespace(cuda="13.0"),
            inference_mode=nullcontext,
            zeros=lambda shape, dtype, device: Tensor(shape, dtype),
            ones=lambda shape, dtype, device: Tensor(shape, dtype, 1),
            isfinite=lambda out: Scalar(np.isfinite(out.value)))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            interface_path = root / "flash_attn_interface.py"
            extension_path = root / "_C.so"
            interface_path.write_bytes(b"interface")
            extension_path.write_bytes(b"native")
            calls = []
            def kernel(q, k, v):
                calls.append((q.shape, k.shape, q.dtype, k.dtype, v.dtype))
                return Tensor(q.shape, "bf16", 1)
            interface = types.SimpleNamespace(__file__=str(interface_path), flash_attn_func=kernel)
            extension = types.SimpleNamespace(__file__=str(extension_path))
            def sam_kernel(q, k, v):
                self.assertEqual((q.dtype, k.dtype, v.dtype), ("bf16", "bf16", "bf16"))
                return interface.flash_attn_func(q.to("fp8"), k.to("fp8"), v.to("fp8")).to(q.dtype)
            distribution = types.SimpleNamespace(version="3.0.0", files=(interface_path, extension_path),
                                                 locate_file=lambda item: item)
            modules = {"flash_attn_interface": interface, "flash_attn_3._C": extension,
                       "sam3.perflib.fa3": types.SimpleNamespace(flash_attn_func=sam_kernel)}
            with mock.patch("XTA.lta_sam.importlib.metadata.distribution", return_value=distribution), \
                 mock.patch("XTA.lta_sam.resolve_pinned_sam_runtime_provenance", return_value={}), \
                 mock.patch("XTA.lta_sam.importlib.import_module", side_effect=modules.__getitem__):
                receipt = preflight_sam_fa3(runtime, 0)
                self.assertEqual([call[0][-1] for call in calls], [64, 64, 32, 32])
                self.assertEqual(calls[-1][1][1], 256)
                self.assertTrue(all(call[2:] == ("fp8", "fp8", "fp8") for call in calls))
                self.assertEqual(receipt["extension"]["sha256"], hashlib.sha256(b"native").hexdigest())
                self.assertFalse(receipt["fallback"])
                modules["sam3.perflib.fa3"].flash_attn_func = lambda q, k, v: (q, k, v)
                with self.assertRaisesRegex(RuntimeError, "SAM FA3 wrapper head"):
                    preflight_sam_fa3(runtime, 0)
                modules["sam3.perflib.fa3"].flash_attn_func = sam_kernel
                interface.flash_attn_func = lambda q, k, v: Tensor(q.shape, "fp32", 1)
                with self.assertRaisesRegex(RuntimeError, "before model allocation.*incompatible output"):
                    preflight_sam_fa3(runtime, 0)
                interface.flash_attn_func = lambda q, k, v: Tensor(q.shape, "bf16", float("nan"))
                with self.assertRaisesRegex(RuntimeError, "known-value kernel check"):
                    preflight_sam_fa3(runtime, 0)
                distribution.version = "4.0.0"
                with self.assertRaisesRegex(RuntimeError, "expected flash-attn-3==3.0.0"):
                    preflight_sam_fa3(runtime, 0)
                distribution.version = "3.0.0"
                distribution.files = (extension_path,)
                with self.assertRaisesRegex(RuntimeError, "outside its installed distribution"):
                    preflight_sam_fa3(runtime, 0)

    def test_fa3_routing_audits_both_model_paths_and_complex_rope(self) -> None:
        from XTA.lta_sam import audit_sam_fa3_routing

        Attention = type("Attention", (), {"__module__": "sam3.model.vitdet",
                                           "use_fa3": True, "use_rope_real": False})
        Memory = type("SimpleRoPEAttention", (), {"__module__": "sam3.model.decoder",
                                                  "use_fa3": True, "use_rope_real": False})
        modules = [(f"visual.{index}", Attention()) for index in range(32)]
        modules += [(f"memory.{index}", Memory()) for index in range(8)]
        predictor = types.SimpleNamespace(model=types.SimpleNamespace(named_modules=lambda: modules))
        receipt = audit_sam_fa3_routing(predictor)
        self.assertEqual(len(receipt["image_encoder"]), 32)
        self.assertEqual(len(receipt["memory_attention"]), 8)
        modules[-1][1].use_fa3 = False
        with self.assertRaisesRegex(RuntimeError, "routing audit failed"):
            audit_sam_fa3_routing(predictor)
        modules[-1][1].use_fa3 = True
        modules[-1][1].use_rope_real = True
        with self.assertRaisesRegex(RuntimeError, "routing audit failed"):
            audit_sam_fa3_routing(predictor)
        modules.pop()
        with self.assertRaisesRegex(RuntimeError, "unexpected attention inventory"):
            audit_sam_fa3_routing(predictor)

    def test_worker_fa3_preflight_precedes_model_allocation_and_publishes_receipts(self) -> None:
        from XTA.lta_worker_adapter import build_worker_predictor

        runtime = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1,
            set_device=mock.Mock(), get_device_name=lambda _: "NVIDIA H100 80GB HBM3",
            get_device_capability=lambda _: (9, 0), empty_cache=mock.Mock()))
        order = mock.Mock()
        order.preflight.return_value = {"kernel_checks": [32, 64]}
        order.builder.return_value = object()
        order.routing.return_value = {"policy": "fp8"}
        with mock.patch.dict(sys.modules, {"torch": runtime, "cv2": types.SimpleNamespace()}), \
             mock.patch("XTA.lta_cpu.configure_worker_runtime_threads", return_value={}), \
             mock.patch("XTA.lta_sam.resolve_local_sam_bundle",
                        return_value=types.SimpleNamespace(model_version="sam3.1")), \
             mock.patch("XTA.lta_sam.resolve_pinned_sam_runtime_provenance", return_value={}), \
             mock.patch("XTA.lta_sam.preflight_sam_fa3", order.preflight), \
             mock.patch("XTA.lta_sam.build_local_sam_predictor", order.builder), \
             mock.patch("XTA.lta_sam.audit_sam_fa3_routing", order.routing), \
             mock.patch("XTA.lta_sam.install_sdpa_fallback", return_value=None):
            context = build_worker_predictor({"model_path": "local.pt", "attention_backend": "fa3_fp8"})
            self.assertEqual([call[0] for call in order.mock_calls], ["preflight", "builder", "routing"])
            self.assertEqual(context.profile["attention_preflight"], {"kernel_checks": [32, 64]})
            self.assertEqual(context.profile["attention_routing"], {"policy": "fp8"})
            self.assertEqual(order.builder.call_args.kwargs["weight_storage"], "float32")
            self.assertFalse(order.builder.call_args.kwargs["use_rope_real"])
            order.reset_mock()
            build_worker_predictor({"model_path": "local.pt", "attention_backend": "sdpa"})
            order.preflight.assert_not_called()
            order.routing.assert_not_called()
            self.assertFalse(order.builder.call_args.kwargs["use_fa3"])
            order.reset_mock()
            order.preflight.side_effect = RuntimeError("no native kernel")
            with self.assertRaisesRegex(RuntimeError, "no native kernel"):
                build_worker_predictor({"model_path": "local.pt", "attention_backend": "fa3_fp8"})
            order.builder.assert_not_called()
            order.preflight.side_effect = None
            order.reset_mock()
            predictor = types.SimpleNamespace(shutdown=mock.Mock())
            order.builder.return_value = predictor
            order.routing.side_effect = RuntimeError("incorrect model route")
            with self.assertRaisesRegex(RuntimeError, "incorrect model route"):
                build_worker_predictor({"model_path": "local.pt", "attention_backend": "fa3_fp8"})
            predictor.shutdown.assert_called_once_with()
            runtime.cuda.empty_cache.assert_called_once_with()

    def test_module_import_does_not_import_torch_or_sam(self) -> None:
        previous = {
            name: sys.modules.pop(name)
            for name in tuple(sys.modules)
            if name in {"torch", "sam3", "pkg_resources"} or name.startswith("sam3.")
        }
        try:
            sys.modules.pop("XTA.lta_sam", None)
            __import__("XTA.lta_sam")
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("sam3", sys.modules)
            self.assertNotIn("pkg_resources", sys.modules)
        finally:
            sys.modules.update(previous)

    def test_local_bundle_file_and_directory_resolution(self) -> None:
        from XTA.lta_sam import resolve_local_sam_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            bpe = root / "bpe_simple_vocab_16e6.txt.gz"
            bpe.write_bytes(b"bpe")

            direct = resolve_local_sam_bundle(checkpoint)
            bundled = resolve_local_sam_bundle(root)

        self.assertEqual(direct.checkpoint_path, checkpoint.resolve())
        self.assertEqual(direct.model_version, "sam3.1")
        self.assertIsNone(direct.bpe_path)
        self.assertEqual(bundled.checkpoint_path, checkpoint.resolve())
        self.assertEqual(bundled.bpe_path, bpe.resolve())

    def test_bundle_resolution_has_no_remote_or_ambiguous_fallback(self) -> None:
        from XTA.lta_sam import resolve_local_sam_bundle

        for value in ("facebook/sam3.1", "https://example.test/sam.pt", "hf:facebook/sam3"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "existing local filesystem path"
            ):
                resolve_local_sam_bundle(value)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "first.pt").write_bytes(b"first")
            (root / "second.pth").write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "exactly one SAM checkpoint"):
                resolve_local_sam_bundle(root)

    def test_local_bundle_revalidation_detects_checkpoint_mutation(self) -> None:
        from XTA.lta_sam import resolve_local_sam_bundle, revalidate_local_sam_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"before")
            bundle = resolve_local_sam_bundle(checkpoint)
            checkpoint.write_bytes(b"after-and-different-size")

            with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                revalidate_local_sam_bundle(bundle)

    def test_fixed_sessions_are_contiguous_nonoverlapping_and_bounded(self) -> None:
        from XTA.lta_sam import LTA_SESSION_FRAMES, plan_sam_sessions

        plans = plan_sam_sessions("sagittal__tta_a120", 65)

        self.assertEqual(LTA_SESSION_FRAMES, 30)
        self.assertEqual(
            [(plan.frame_start, plan.frame_stop) for plan in plans],
            [(0, 30), (30, 60), (60, 65)],
        )
        self.assertEqual([plan.session_index for plan in plans], [0, 1, 2])
        self.assertEqual(list(plans[-1].frame_indices), list(range(60, 65)))

    def test_shared_confidence_and_distinct_video_scores(self) -> None:
        from XTA.lta_sam import SamFramePrediction, resolve_confidence

        self.assertEqual(resolve_confidence("0.15"), 0.15)
        prediction = SamFramePrediction(
            sequence_id="transverse__tta_a0",
            session_index=0,
            frame_index=4,
            object_id=9,
            initial_detection_score=0.4,
            frame_tracker_score=0.8,
            binary_mask=object(),
        )
        self.assertEqual(prediction.initial_detection_score, 0.4)
        self.assertEqual(prediction.frame_tracker_score, 0.8)
        for invalid in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_confidence(invalid)

    def test_builder_always_receives_explicit_local_checkpoint(self) -> None:
        from XTA.lta_sam import (
            LTA_MAX_NUM_OBJECTS,
            LTA_MULTIPLEX_COUNT,
            build_local_sam_predictor,
            resolve_local_sam_bundle,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = resolve_local_sam_bundle(checkpoint)
            fake_model = type(
                "Model",
                (),
                {"score_threshold_detection": 0.5, "image_only_det_thresh": 0.5},
            )()
            fake_predictor = type(
                "Predictor",
                (),
                {"default_output_prob_thresh": 0.5, "model": fake_model},
            )()
            builder = mock.Mock(return_value=fake_predictor)

            predictor = build_local_sam_predictor(bundle, builder=builder)

        self.assertIs(predictor, builder.return_value)
        kwargs = builder.call_args.kwargs
        self.assertEqual(kwargs["checkpoint_path"], str(checkpoint.resolve()))
        self.assertEqual(kwargs["version"], "sam3.1")
        self.assertEqual(kwargs["max_num_objects"], LTA_MAX_NUM_OBJECTS)
        self.assertEqual(kwargs["multiplex_count"], LTA_MULTIPLEX_COUNT)
        self.assertFalse(kwargs["use_fa3"])
        self.assertFalse(kwargs["use_rope_real"])
        self.assertNotIn("load_from_HF", kwargs)

    def test_confidence_mapping_updates_exposed_predictor_and_model_controls(self) -> None:
        from XTA.lta_sam import configure_predictor_confidence

        model = type(
            "Model",
            (),
            {
                "score_threshold_detection": 0.5,
                "image_only_det_thresh": 0.5,
                "new_det_thresh": 0.7,
            },
        )()
        predictor = type(
            "Predictor",
            (),
            {"default_output_prob_thresh": 0.5, "model": model},
        )()

        updated = configure_predictor_confidence(predictor, 0.15)

        self.assertEqual(predictor.default_output_prob_thresh, 0.15)
        self.assertEqual(model.score_threshold_detection, 0.15)
        self.assertEqual(model.image_only_det_thresh, 0.15)
        self.assertEqual(model.new_det_thresh, 0.7)
        self.assertIn("model.score_threshold_detection", updated)

    def test_fa3_policy_targets_hopper_not_4090_ada(self) -> None:
        from XTA.lta_sam import cuda_capability_supports_fa3

        self.assertFalse(cuda_capability_supports_fa3(8, 9))
        self.assertTrue(cuda_capability_supports_fa3(9, 0))
        self.assertTrue(cuda_capability_supports_fa3(10, 0))

    def test_production_device_profiles_are_explicit_and_fail_closed(self) -> None:
        from XTA.lta_sam import resolve_sam_device_profile

        class Cuda:
            name = "NVIDIA H100 80GB HBM3"
            capability = (9, 0)

            @classmethod
            def get_device_name(cls, _device):
                return cls.name

            @classmethod
            def get_device_capability(cls, _device):
                return cls.capability

        torch = types.SimpleNamespace(cuda=Cuda)
        h100 = resolve_sam_device_profile(torch, 0, attention_backend="auto")
        self.assertEqual(h100["name"], "h100")
        self.assertEqual(h100["weight_storage"], "float32")
        self.assertTrue(h100["use_fa3"])
        self.assertFalse(h100["use_rope_real"])
        self.assertFalse(h100["compile"])
        self.assertEqual(h100["attention_backend"], "meta_fa3_fp8_e4m3fn")
        baseline = resolve_sam_device_profile(torch, 0, attention_backend="sdpa")
        self.assertFalse(baseline["use_fa3"])
        self.assertEqual(baseline["weight_storage"], "float32")

        Cuda.name = "NVIDIA GeForce RTX 4090 Laptop GPU"
        Cuda.capability = (8, 9)
        egpu = resolve_sam_device_profile(torch, 0, attention_backend="auto")
        self.assertEqual(egpu["name"], "egpu")
        self.assertEqual(egpu["weight_storage"], "bfloat16_egpu")
        self.assertEqual(egpu["attention_backend"], "torch_sdpa_flash_efficient_math")
        self.assertEqual(egpu["runtime_autocast_dtype"], "bfloat16")
        self.assertFalse(egpu["use_fa3"])
        with self.assertRaisesRegex(ValueError, "Hopper"):
            resolve_sam_device_profile(torch, 0, attention_backend="fa3_fp8")

        Cuda.name = "NVIDIA A100-SXM4-80GB"
        Cuda.capability = (8, 0)
        with self.assertRaisesRegex(ValueError, "4090"):
            resolve_sam_device_profile(torch, 0)

    def test_local_bpe_resolution_and_init_state_signature_filter(self) -> None:
        from XTA.lta_sam import (
            patch_sam_init_state_signature,
            resolve_installed_sam_bpe,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            bpe = package_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            bpe.parent.mkdir(parents=True)
            with gzip.open(bpe, "wb") as handle:
                handle.write(b'"bpe_simple_vocab_16e6.txt#version: 0.2\nsynthetic\n')
            digest = hashlib.sha256(bpe.read_bytes()).hexdigest()
            with mock.patch("XTA.lta_sam._PINNED_SAM_BPE_SHA256", digest):
                self.assertEqual(
                    resolve_installed_sam_bpe(package_root=package_root),
                    bpe.resolve(),
                )

        class Model:
            def __init__(self) -> None:
                self.kwargs = None

            def init_state(self, *, resource_path, offload_video_to_cpu=False):
                self.kwargs = {
                    "resource_path": resource_path,
                    "offload_video_to_cpu": offload_video_to_cpu,
                }
                return self.kwargs

        predictor = type("Predictor", (), {"model": Model()})()
        dropped = patch_sam_init_state_signature(predictor)
        result = predictor.model.init_state(
            resource_path="frames",
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        self.assertEqual(dropped, ("offload_state_to_cpu",))
        self.assertEqual(result["offload_video_to_cpu"], True)

        with self.assertRaisesRegex(TypeError, "unexpected_keyword"):
            predictor.model.init_state(
                resource_path="frames",
                unexpected_keyword=True,
            )

    def test_bpe_resolution_locates_package_without_executing_sam(self) -> None:
        from XTA import lta_sam

        previous = {
            name: sys.modules.pop(name)
            for name in tuple(sys.modules)
            if name == "sam3" or name.startswith("sam3.")
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "sam3"
            package.mkdir()
            (package / "__init__.py").write_text(
                "raise RuntimeError('sam3 package executed')\n", encoding="utf-8"
            )
            bpe = package / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            bpe.parent.mkdir()
            with gzip.open(bpe, "wb") as handle:
                handle.write(b'"bpe_simple_vocab_16e6.txt#version: 0.2\nsynthetic\n')
            digest = hashlib.sha256(bpe.read_bytes()).hexdigest()
            sys.path.insert(0, str(root))
            try:
                with mock.patch.object(lta_sam, "_PINNED_SAM_BPE_SHA256", digest):
                    resolved = lta_sam.resolve_installed_sam_bpe()
                self.assertEqual(resolved, bpe.resolve())
                self.assertNotIn("sam3", sys.modules)
            finally:
                sys.path.remove(str(root))
                for name in tuple(sys.modules):
                    if name == "sam3" or name.startswith("sam3."):
                        sys.modules.pop(name, None)
                sys.modules.update(previous)

    def test_builder_import_uses_scoped_pkg_resources_shim_when_absent(self) -> None:
        from XTA import lta_sam

        previous = {
            name: sys.modules.pop(name)
            for name in tuple(sys.modules)
            if name == "pkg_resources" or name == "sam3" or name.startswith("sam3.")
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "sam3"
            package.mkdir()
            (package / "__init__.py").write_text("# synthetic sam3\n", encoding="utf-8")
            (package / "model_builder.py").write_text(
                "import pkg_resources\nBOUND_PKG_RESOURCES = pkg_resources\n",
                encoding="utf-8",
            )
            expected_bpe = root / "validated-bpe.gz"
            original_find_spec = lta_sam.importlib.util.find_spec

            def find_spec(name, *args, **kwargs):
                if name == "pkg_resources":
                    return None
                return original_find_spec(name, *args, **kwargs)

            sys.path.insert(0, str(root))
            try:
                with (
                    mock.patch.object(
                        lta_sam.importlib.util, "find_spec", side_effect=find_spec
                    ),
                    mock.patch.object(
                        lta_sam,
                        "resolve_installed_sam_bpe",
                        return_value=expected_bpe,
                    ),
                ):
                    model_builder = lta_sam._import_sam_model_builder()
                    shim = model_builder.BOUND_PKG_RESOURCES
                    self.assertEqual(
                        shim.resource_filename(
                            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
                        ),
                        str(expected_bpe),
                    )
                    with self.assertRaisesRegex(RuntimeError, "accepts only"):
                        shim.resource_filename("other", "asset.bin")
                self.assertNotIn("pkg_resources", sys.modules)
                self.assertIs(model_builder.pkg_resources, shim)
            finally:
                sys.path.remove(str(root))
                for name in tuple(sys.modules):
                    if (
                        name == "pkg_resources"
                        or name == "sam3"
                        or name.startswith("sam3.")
                    ):
                        sys.modules.pop(name, None)
                sys.modules.update(previous)

    def test_builder_import_restores_both_tf32_flags(self) -> None:
        from XTA import lta_sam

        for initial in ((False, True), (True, False)):
            with self.subTest(initial=initial):
                matmul = types.SimpleNamespace(allow_tf32=initial[0])
                cudnn = types.SimpleNamespace(allow_tf32=initial[1])
                fake_torch = types.SimpleNamespace(
                    backends=types.SimpleNamespace(
                        cuda=types.SimpleNamespace(matmul=matmul), cudnn=cudnn
                    )
                )
                builder_module = types.ModuleType("sam3.model_builder")

                def import_builder(_name):
                    matmul.allow_tf32 = not initial[0]
                    cudnn.allow_tf32 = not initial[1]
                    return builder_module

                with (
                    mock.patch.dict(
                        sys.modules,
                        {"torch": fake_torch, "sam3.model_builder": None},
                    ),
                    mock.patch.object(
                        lta_sam,
                        "_sam_pkg_resources_import_compatibility",
                        return_value=nullcontext(),
                    ),
                    mock.patch.object(
                        lta_sam.importlib, "import_module", side_effect=import_builder
                    ),
                ):
                    self.assertIs(lta_sam._import_sam_model_builder(), builder_module)
                    self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)

                def failing_import(_name):
                    matmul.allow_tf32 = not initial[0]
                    cudnn.allow_tf32 = not initial[1]
                    raise RuntimeError("injected SAM import failure")

                with (
                    mock.patch.dict(
                        sys.modules,
                        {"torch": fake_torch, "sam3.model_builder": None},
                    ),
                    mock.patch.object(
                        lta_sam,
                        "_sam_pkg_resources_import_compatibility",
                        return_value=nullcontext(),
                    ),
                    mock.patch.object(
                        lta_sam.importlib, "import_module", side_effect=failing_import
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected SAM import failure"):
                        lta_sam._import_sam_model_builder()
                    self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)

                with (
                    mock.patch.dict(sys.modules, {"sam3.model_builder": builder_module}),
                    mock.patch.object(lta_sam.importlib, "import_module") as import_module,
                ):
                    self.assertIs(lta_sam._import_sam_model_builder(), builder_module)
                    import_module.assert_not_called()
                    self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)

    def test_sam31_build_restores_both_tf32_flags(self) -> None:
        from XTA import lta_sam

        for initial in ((False, True), (True, False)):
            with self.subTest(initial=initial):
                matmul = types.SimpleNamespace(allow_tf32=initial[0])
                cudnn = types.SimpleNamespace(allow_tf32=initial[1])
                fake_torch = types.SimpleNamespace(
                    backends=types.SimpleNamespace(
                        cuda=types.SimpleNamespace(matmul=matmul), cudnn=cudnn
                    ),
                    cuda=types.SimpleNamespace(empty_cache=lambda: None),
                )

                def build(**_kwargs):
                    matmul.allow_tf32 = not initial[0]
                    cudnn.allow_tf32 = not initial[1]
                    return object()

                with mock.patch.object(
                    lta_sam,
                    "_build_real_sam31_predictor_single_load_unlocked",
                    side_effect=build,
                ):
                    lta_sam._build_real_sam31_predictor_single_load(
                        builder=build,
                        kwargs={},
                        bundle=mock.sentinel.bundle,
                        runtime_torch=fake_torch,
                        weight_storage="float32",
                        construction_device="cpu",
                    )
                self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)

                def failing_build(**_kwargs):
                    matmul.allow_tf32 = not initial[0]
                    cudnn.allow_tf32 = not initial[1]
                    raise RuntimeError("injected SAM build failure")

                with mock.patch.object(
                    lta_sam,
                    "_build_real_sam31_predictor_single_load_unlocked",
                    side_effect=failing_build,
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected SAM build failure"):
                        lta_sam._build_real_sam31_predictor_single_load(
                            builder=failing_build,
                            kwargs={},
                            bundle=mock.sentinel.bundle,
                            runtime_torch=fake_torch,
                            weight_storage="float32",
                            construction_device="cpu",
                        )
                self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)

    def test_pkg_resources_compatibility_preserves_existing_module(self) -> None:
        import types

        from XTA import lta_sam

        marker = object()
        previous = sys.modules.get("pkg_resources", marker)
        existing = types.ModuleType("pkg_resources")
        sys.modules["pkg_resources"] = existing
        try:
            with lta_sam._sam_pkg_resources_import_compatibility() as installed:
                self.assertFalse(installed)
                self.assertIs(sys.modules["pkg_resources"], existing)
            self.assertIs(sys.modules["pkg_resources"], existing)
        finally:
            if previous is marker:
                sys.modules.pop("pkg_resources", None)
            else:
                sys.modules["pkg_resources"] = previous

    def test_init_state_signature_filter_survives_start_session_request_chain(self) -> None:
        from XTA.lta_sam import patch_sam_init_state_signature

        class Model:
            def init_state(self, *, resource_path, offload_video_to_cpu=False):
                return {
                    "resource_path": resource_path,
                    "offload_video_to_cpu": offload_video_to_cpu,
                }

        class Predictor:
            def __init__(self) -> None:
                self.model = Model()

            def handle_request(self, request):
                if request["type"] != "start_session":
                    raise AssertionError(request)
                state = self.model.init_state(
                    resource_path=request["resource_path"],
                    offload_video_to_cpu=request.get("offload_video_to_cpu", False),
                    offload_state_to_cpu=request.get("offload_state_to_cpu", False),
                )
                return {"session_id": "session-1", "state": state}

        predictor = Predictor()
        patch_sam_init_state_signature(predictor)
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": "frames",
                "offload_video_to_cpu": True,
            }
        )

        self.assertEqual(response["state"]["resource_path"], "frames")
        self.assertTrue(response["state"]["offload_video_to_cpu"])

    def test_builder_fails_if_pinned_runtime_cannot_apply_conf(self) -> None:
        from XTA.lta_sam import build_local_sam_predictor, resolve_local_sam_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = resolve_local_sam_bundle(checkpoint)
            with self.assertRaisesRegex(RuntimeError, "refusing to ignore --conf"):
                build_local_sam_predictor(
                    bundle,
                    builder=mock.Mock(return_value=object()),
                )

    def test_builder_rejects_unknown_internal_weight_storage(self) -> None:
        from XTA.lta_sam import build_local_sam_predictor, resolve_local_sam_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = resolve_local_sam_bundle(checkpoint)
            with self.assertRaisesRegex(ValueError, "weight_storage"):
                build_local_sam_predictor(
                    bundle,
                    builder=mock.Mock(),
                    weight_storage="float16",
                )

    def test_builder_accepts_bounded_internal_object_capacity(self) -> None:
        from XTA.lta_sam import build_local_sam_predictor, resolve_local_sam_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = resolve_local_sam_bundle(checkpoint)
            model = type(
                "Model",
                (),
                {"score_threshold_detection": 0.4, "image_only_det_thresh": 0.5},
            )()
            predictor = type(
                "Predictor",
                (),
                {"default_output_prob_thresh": 0.5, "model": model},
            )()
            builder = mock.Mock(return_value=predictor)
            build_local_sam_predictor(
                bundle,
                builder=builder,
                max_num_objects=16,
            )
            self.assertEqual(builder.call_args.kwargs["max_num_objects"], 16)
            with self.assertRaisesRegex(ValueError, "max_num_objects"):
                build_local_sam_predictor(bundle, builder=builder, max_num_objects=0)

    def test_real_sam31_patch_transaction_loads_once_and_restores_globals(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"pinned PyTorch runtime is unavailable: {exc}")
        self.addCleanup(
            setattr,
            torch.backends.cuda.matmul,
            "allow_tf32",
            bool(torch.backends.cuda.matmul.allow_tf32),
        )
        self.addCleanup(
            setattr,
            torch.backends.cudnn,
            "allow_tf32",
            bool(torch.backends.cudnn.allow_tf32),
        )
        try:
            import sam3.model_builder as sam_model_builder
            from sam3.model.sam3_multiplex_tracking import (
                Sam3MultiplexTrackingWithInteractivity,
            )
        except Exception as exc:
            self.skipTest(f"pinned SAM runtime is unavailable: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("tiny real-patch transaction requires an available CUDA device")

        from XTA.lta_sam import (
            _build_real_sam31_predictor_single_load,
            resolve_local_sam_bundle,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam3.1_multiplex.pt"
            torch.save({"probe": torch.ones(1, dtype=torch.float32)}, checkpoint)
            bundle = resolve_local_sam_bundle(checkpoint)
            tracker_calls = []

            def fake_tracker_builder(*args, **kwargs):
                tracker_calls.append(dict(kwargs))
                return object()

            def fake_builder(**kwargs):
                sam_model_builder.build_sam3_multiplex_video_model(
                    checkpoint_path=kwargs["checkpoint_path"],
                    load_from_HF=False,
                )
                drop_path_schedules = [
                    torch.linspace(0, 0.1, 32),
                    torch.linspace(0, 0.1, 32),
                ]
                self.assertTrue(
                    all(value.device.type == "cpu" for value in drop_path_schedules)
                )
                state = torch.load(
                    kwargs["checkpoint_path"],
                    map_location="cuda",
                    weights_only=False,
                    mmap=False,
                )
                model = object.__new__(Sam3MultiplexTrackingWithInteractivity)
                torch.nn.Module.__init__(model)
                model.register_parameter(
                    "probe",
                    torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)),
                )
                model.load_state_dict(state, strict=False)
                model.cuda()
                return type("Predictor", (), {"model": model})()

            original_load = torch.load
            original_state_loader = Sam3MultiplexTrackingWithInteractivity.load_state_dict
            original_cuda = Sam3MultiplexTrackingWithInteractivity.cuda
            with mock.patch.object(
                sam_model_builder,
                "build_sam3_multiplex_video_model",
                fake_tracker_builder,
            ), mock.patch.object(
                sam_model_builder,
                "build_sam3_predictor",
                fake_builder,
            ):
                predictor = _build_real_sam31_predictor_single_load(
                    builder=fake_builder,
                    kwargs={"checkpoint_path": str(checkpoint)},
                    bundle=bundle,
                    runtime_torch=torch,
                    weight_storage="float32",
                    construction_device="meta",
                )
                self.assertIsNotNone(predictor.model)
                self.assertIs(torch.load, original_load)
                self.assertIs(
                    Sam3MultiplexTrackingWithInteractivity.load_state_dict,
                    original_state_loader,
                )
                self.assertIs(
                    Sam3MultiplexTrackingWithInteractivity.cuda,
                    original_cuda,
                )
                self.assertIs(
                    sam_model_builder.build_sam3_multiplex_video_model,
                    fake_tracker_builder,
                )

            self.assertEqual(len(tracker_calls), 1)
            self.assertIsNone(tracker_calls[0]["checkpoint_path"])
            self.assertFalse(tracker_calls[0]["load_from_HF"])
            self.assertEqual(tracker_calls[0]["device"], "meta")

            def failing_builder(**kwargs):
                raise RuntimeError("injected build failure")

            with mock.patch.object(
                sam_model_builder,
                "build_sam3_multiplex_video_model",
                fake_tracker_builder,
            ), mock.patch.object(
                sam_model_builder,
                "build_sam3_predictor",
                failing_builder,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected build failure"):
                    _build_real_sam31_predictor_single_load(
                        builder=failing_builder,
                        kwargs={"checkpoint_path": str(checkpoint)},
                        bundle=bundle,
                        runtime_torch=torch,
                        weight_storage="float32",
                        construction_device="meta",
                    )
                self.assertIs(torch.load, original_load)
                self.assertIs(
                    Sam3MultiplexTrackingWithInteractivity.load_state_dict,
                    original_state_loader,
                )
                self.assertIs(
                    Sam3MultiplexTrackingWithInteractivity.cuda,
                    original_cuda,
                )

            def unexpected_buffer_builder(**kwargs):
                sam_model_builder.build_sam3_multiplex_video_model(
                    checkpoint_path=kwargs["checkpoint_path"],
                    load_from_HF=False,
                )
                torch.linspace(0, 0.1, 32)
                torch.linspace(0, 0.1, 32)
                state = torch.load(kwargs["checkpoint_path"])
                model = object.__new__(Sam3MultiplexTrackingWithInteractivity)
                torch.nn.Module.__init__(model)
                model.register_parameter(
                    "probe",
                    torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)),
                )
                model.register_buffer(
                    "unexpected_constructor_buffer",
                    torch.ones(1, dtype=torch.float32, device="cpu"),
                    persistent=False,
                )
                model.load_state_dict(state, strict=False)
                model.cuda()
                return type("Predictor", (), {"model": model})()

            with mock.patch.object(
                sam_model_builder,
                "build_sam3_multiplex_video_model",
                fake_tracker_builder,
            ), mock.patch.object(
                sam_model_builder,
                "build_sam3_predictor",
                unexpected_buffer_builder,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "non-checkpoint buffers outside the pinned causal mask"
                ):
                    _build_real_sam31_predictor_single_load(
                        builder=unexpected_buffer_builder,
                        kwargs={"checkpoint_path": str(checkpoint)},
                        bundle=bundle,
                        runtime_torch=torch,
                        weight_storage="float32",
                        construction_device="meta",
                    )
                self.assertIs(torch.load, original_load)

    def test_meta_construction_rejects_compile_and_warm_up(self) -> None:
        from XTA.lta_sam import build_local_sam_predictor, resolve_local_sam_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = resolve_local_sam_bundle(checkpoint)
            for options in ({"compile": True}, {"warm_up": True}):
                with self.subTest(options=options), self.assertRaisesRegex(
                    ValueError, "compile=False and warm_up=False"
                ):
                    build_local_sam_predictor(
                        bundle,
                        builder=mock.Mock(),
                        construction_device="meta",
                        **options,
                    )

    def test_video_session_normalizes_global_frames_and_always_closes(self) -> None:
        from XTA.lta_sam import SamPromptBox, plan_sam_sessions, run_video_session

        class FakePredictor:
            def __init__(self) -> None:
                self.requests = []

            def handle_request(self, request):
                self.requests.append(dict(request))
                if request["type"] == "start_session":
                    return {"session_id": "session-1"}
                if request["type"] == "add_prompt":
                    return {
                        "frame_index": request["frame_index"],
                        "outputs": {"frame_stats": {"num_obj_dropped": 0}},
                    }
                return {"is_success": True}

            def handle_stream_request(self, request):
                self.requests.append(dict(request))
                for frame_index in range(30):
                    populated = frame_index == 2
                    yield {
                        "frame_index": frame_index,
                        "outputs": {
                            "out_obj_ids": [7] if populated else [],
                            "out_probs": [0.35] if populated else [],
                            "out_binary_masks": (
                                np.asarray([[[True, False]]], dtype=bool)
                                if populated
                                else np.empty((0, 1, 2), dtype=bool)
                            ),
                            "frame_stats": {"num_obj_dropped": 0},
                        },
                    }

        predictor = FakePredictor()
        session = plan_sam_sessions("transverse__tta_a0", 65)[1]
        prompt = SamPromptBox(
            exemplar_id="positive",
            frame_index=30,
            xywh=(0.1, 0.2, 0.3, 0.4),
        )

        predictions = run_video_session(
            predictor,
            resource=[object()] * 30,
            session=session,
            prompt=prompt,
            conf=0.15,
            offload_video_to_cpu=True,
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].frame_index, 32)
        self.assertEqual(predictions[0].initial_detection_score, 0.35)
        self.assertIsNone(predictions[0].frame_tracker_score)
        self.assertEqual(predictor.requests[-1]["type"], "close_session")
        add_prompt = next(req for req in predictor.requests if req["type"] == "add_prompt")
        self.assertEqual(add_prompt["frame_index"], 0)
        self.assertEqual(add_prompt["output_prob_thresh"], 0.15)
        start = next(req for req in predictor.requests if req["type"] == "start_session")
        self.assertTrue(start["offload_video_to_cpu"])
        propagate = next(
            req for req in predictor.requests if req["type"] == "propagate_in_video"
        )
        self.assertEqual(propagate["max_frame_num_to_track"], 30)

    def test_video_session_rejects_resource_outside_fixed_decoded_window(self) -> None:
        from XTA.lta_sam import SamPromptBox, plan_sam_sessions, run_video_session

        session = plan_sam_sessions("transverse__tta_a0", 30)[0]
        prompt = SamPromptBox(
            exemplar_id="positive",
            frame_index=0,
            xywh=(0.1, 0.2, 0.3, 0.4),
        )
        with self.assertRaisesRegex(TypeError, "ordered list"):
            run_video_session(
                object(),
                resource="full-video.mkv",
                session=session,
                prompt=prompt,
                conf=0.15,
            )
        with self.assertRaisesRegex(ValueError, "resource length"):
            run_video_session(
                object(),
                resource=[object()] * 31,
                session=session,
                prompt=prompt,
                conf=0.15,
            )

    def test_video_session_audits_initial_prompt_multiplex_overflow(self) -> None:
        from XTA.lta_sam import SamPromptBox, plan_sam_sessions, run_video_session

        class OverflowPredictor:
            def __init__(self) -> None:
                self.closed = False

            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "session-1"}
                if request["type"] == "add_prompt":
                    return {
                        "frame_index": request["frame_index"],
                        "outputs": {"frame_stats": {"num_obj_dropped": 3}},
                    }
                if request["type"] == "close_session":
                    self.closed = True
                    return {"is_success": True}
                raise AssertionError(request)

            def handle_stream_request(self, request):
                raise AssertionError("propagation must not start after overflow")
                yield request

        predictor = OverflowPredictor()
        session = plan_sam_sessions("transverse__tta_a0", 30)[0]
        prompt = SamPromptBox(
            exemplar_id="positive",
            frame_index=0,
            xywh=(0.1, 0.2, 0.3, 0.4),
        )
        with self.assertRaisesRegex(RuntimeError, "dropped 3 object"):
            run_video_session(
                predictor,
                resource=[object()] * 30,
                session=session,
                prompt=prompt,
                conf=0.15,
            )
        self.assertTrue(predictor.closed)

    def test_video_session_preserves_primary_failure_when_close_also_fails(self) -> None:
        from XTA.lta_sam import SamPromptBox, plan_sam_sessions, run_video_session

        class BrokenPredictor:
            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "session-1"}
                if request["type"] == "add_prompt":
                    return {
                        "frame_index": request["frame_index"],
                        "outputs": {"frame_stats": {"num_obj_dropped": 0}},
                    }
                if request["type"] == "close_session":
                    raise RuntimeError("cleanup failed")
                raise AssertionError(request)

            def handle_stream_request(self, request):
                raise MemoryError("primary inference failure")
                yield request

        session = plan_sam_sessions("transverse__tta_a0", 30)[0]
        prompt = SamPromptBox(
            exemplar_id="positive",
            frame_index=0,
            xywh=(0.1, 0.2, 0.3, 0.4),
        )

        with self.assertRaisesRegex(MemoryError, "primary inference failure") as raised:
            run_video_session(
                BrokenPredictor(),
                resource=[object()] * 30,
                session=session,
                prompt=prompt,
                conf=0.15,
            )
        self.assertTrue(
            any("cleanup failed" in note for note in getattr(raised.exception, "__notes__", ()))
        )

    def test_video_session_closes_failed_stream_and_discards_partial_results(self) -> None:
        from XTA.lta_sam import SamPromptBox, plan_sam_sessions, run_video_session

        class FailedStream:
            def __init__(self) -> None:
                self.closed = False
                self.responses = iter(({"frame_index": 99, "outputs": {}},))

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.responses)

            def close(self):
                self.closed = True

        stream = FailedStream()

        class Predictor:
            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "session-1"}
                if request["type"] == "add_prompt":
                    return {
                        "frame_index": request["frame_index"],
                        "outputs": {"frame_stats": {"num_obj_dropped": 0}},
                    }
                return {"is_success": True}

            def handle_stream_request(self, request):
                return stream

        session = plan_sam_sessions("transverse__tta_a0", 30)[0]
        prompt = SamPromptBox(
            exemplar_id="positive",
            frame_index=0,
            xywh=(0.1, 0.2, 0.3, 0.4),
        )
        with self.assertRaisesRegex(RuntimeError, "out-of-range frame 99"):
            run_video_session(
                Predictor(),
                resource=[object()] * 30,
                session=session,
                prompt=prompt,
                conf=0.15,
            )
        self.assertTrue(stream.closed)

    def test_image_frame_uses_xywh_exemplar_and_shared_conf(self) -> None:
        from XTA.lta_sam import SamPromptBox, run_image_frame

        class FakeProcessor:
            def __init__(self) -> None:
                self.threshold = None
                self.box = None

            def set_confidence_threshold(self, value):
                self.threshold = value

            def set_image(self, image):
                return {"image": image}

            def add_geometric_prompt(self, *, box, label, state):
                self.box = list(box)
                self.asserted_label = label
                return {
                    **state,
                    "scores": [0.6],
                    "masks": np.asarray([[[[True]]]], dtype=bool),
                }

        processor = FakeProcessor()
        prompt = SamPromptBox(
            exemplar_id="positive",
            frame_index=5,
            xywh=(0.1, 0.2, 0.4, 0.6),
        )

        predictions = run_image_frame(
            processor,
            image=object(),
            sequence_id="sagittal__tta_a0",
            frame_index=5,
            prompt=prompt,
            conf=0.15,
        )

        self.assertEqual(processor.threshold, 0.15)
        for actual, expected in zip(processor.box, [0.3, 0.5, 0.4, 0.6]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(processor.asserted_label)
        self.assertEqual(predictions[0].object_id, 0)
        self.assertEqual(predictions[0].frame_index, 5)
        self.assertEqual(tuple(predictions[0].binary_mask.shape), (1, 1))

    def test_video_output_rejects_multiplex_overflow(self) -> None:
        from XTA.lta_sam import normalize_video_frame_output

        with self.assertRaisesRegex(RuntimeError, "dropped 2 object"):
            normalize_video_frame_output(
                {
                    "out_obj_ids": [],
                    "out_probs": [],
                    "out_binary_masks": np.empty((0, 2, 2), dtype=bool),
                    "frame_stats": {"num_obj_dropped": 2},
                },
                sequence_id="transverse__tta_a0",
                session_index=0,
                global_frame_index=0,
            )

    def test_video_output_filters_only_the_exact_removed_object_sentinel(self) -> None:
        from XTA.lta_sam import normalize_video_frame_output

        predictions = normalize_video_frame_output(
            {
                "out_obj_ids": [7, 8],
                "out_probs": [-1e4, 0.6],
                "out_binary_masks": np.asarray(
                    [[[True]], [[True]]],
                    dtype=bool,
                ),
                "frame_stats": {"num_obj_dropped": 0},
            },
            sequence_id="transverse__tta_a0",
            session_index=0,
            global_frame_index=0,
        )
        self.assertEqual(tuple(item.object_id for item in predictions), (8,))
        with self.assertRaisesRegex(ValueError, "finite value in"):
            normalize_video_frame_output(
                {
                    "out_obj_ids": [7],
                    "out_probs": [-1.0],
                    "out_binary_masks": np.asarray([[[True]]], dtype=bool),
                    "frame_stats": {"num_obj_dropped": 0},
                },
                sequence_id="transverse__tta_a0",
                session_index=0,
                global_frame_index=0,
            )

    def test_point_preview_allows_absent_drop_stats_without_fabricating_them(self) -> None:
        from XTA.lta_sam import normalize_video_frame_output

        predictions = normalize_video_frame_output(
            {
                "out_obj_ids": [0],
                "out_probs": [1.0],
                "out_binary_masks": np.asarray([[[True]]], dtype=bool),
                "frame_stats": None,
            },
            sequence_id="lta__point_smoke",
            session_index=0,
            global_frame_index=379,
            require_drop_stats=False,
        )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].object_id, 0)
        self.assertEqual(predictions[0].initial_detection_score, 1.0)


if __name__ == "__main__":
    unittest.main()
