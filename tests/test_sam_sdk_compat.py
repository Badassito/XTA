from __future__ import annotations

import gzip
import hashlib
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


class SamSdkCompatibilityTests(unittest.TestCase):
    def _run_cpu_transaction(self, mode="ok", construction_device="meta"):
        from XTA import sam_sdk_compat as sdk
        from tests._sam_sdk_transaction_double import TransactionDouble

        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"local checkpoint identity")
            double = TransactionDouble(checkpoint, construction_device=construction_device, mode=mode)
            runtime = double.runtime
            originals = (runtime.load, runtime.linspace,
                         double.model_class.load_state_dict, double.model_class.cuda,
                         double.builder_module.build_sam3_multiplex_video_model)
            with mock.patch.dict(sys.modules, {
                "sam3.model.sam3_multiplex_tracking": double.tracking_module,
            }), mock.patch.object(sdk, "_import_sam_model_builder", return_value=double.builder_module):
                try:
                    sdk._build_real_sam31_predictor_single_load(
                        builder=double.builder, kwargs={"checkpoint_path": str(checkpoint)},
                        checkpoint_path=checkpoint, runtime_torch=runtime,
                        weight_storage="float32", construction_device=construction_device)
                except BaseException as exc:
                    double.error = exc
                else:
                    double.error = None
                self.assertIs(runtime.load, originals[0])
                self.assertIs(runtime.linspace, originals[1])
                self.assertIs(double.model_class.load_state_dict, originals[2])
                self.assertIs(double.model_class.cuda, originals[3])
                self.assertIs(double.builder_module.build_sam3_multiplex_video_model, originals[4])
                self.assertFalse(runtime.backends.cuda.matmul.allow_tf32)
                self.assertTrue(runtime.backends.cudnn.allow_tf32)
            return double

    def test_cpu_double_loads_once_assigns_and_chunks_without_cuda_runtime(self):
        for device in ("cpu", "meta"):
            with self.subTest(device=device):
                double = self._run_cpu_transaction(construction_device=device)
                self.assertIsNone(double.error)
                self.assertEqual(len(double.loads), 1)
                self.assertEqual(double.loads[0][1], {
                    "map_location": "cpu", "mmap": True, "weights_only": True})
                self.assertEqual(double.states, [{"strict": False, "assign": True}])
                self.assertEqual(double.runtime.copies, [1048576, 7])
                self.assertEqual(double.tracker_calls, [{
                    "checkpoint_path": None, "load_from_HF": False, "device": device}])
                self.assertEqual(len(double.schedules), 2 if device == "meta" else 0)
                self.assertTrue(all(kwargs == {"device": "cpu"} for _, kwargs in double.schedules))
                self.assertEqual(double.shutdown_count, 0)

    def test_cpu_transaction_restores_globals_after_each_execution_failure(self):
        for mode, message in (("builder_failure", "injected builder failure"),
                              ("load_failure", "injected checkpoint load failure"),
                              ("copy_failure", "injected tensor copy failure"),
                              ("unexpected_buffer", "non-checkpoint buffers"),
                              ("schedule_shape", "unexpected torch.linspace")):
            with self.subTest(mode=mode):
                double = self._run_cpu_transaction(mode)
                self.assertIsInstance(double.error, RuntimeError)
                self.assertIn(message, str(double.error))

    def test_cpu_transaction_rejects_incomplete_or_incompatible_assembly_and_shuts_down(self):
        for mode, message in (("no_load", "exactly one local merged"),
                              ("double_load", "exactly one local merged"),
                              ("no_state", "exactly one assembled"),
                              ("double_state", "exactly one assembled"),
                              ("missing", "missing=1"),
                              ("unexpected", "unexpected=1"),
                              ("schedule_count", "exactly two CPU scalar"),
                              ("placement", "placement=['probe:cpu']"),
                              ("dtype", "dtype=['probe:float16']")):
            with self.subTest(mode=mode):
                double = self._run_cpu_transaction(mode)
                self.assertIsInstance(double.error, RuntimeError)
                self.assertIn(message, str(double.error))
                self.assertEqual(double.shutdown_count, 1)

    def test_sdk_import_is_dependency_light(self):
        import subprocess

        code = "import sys; import XTA.sam_sdk_compat; assert 'torch' not in sys.modules; assert not any(n == 'sam3' or n.startswith('sam3.') for n in sys.modules)"
        subprocess.run([sys.executable, "-B", "-c", code], check=True)

    def test_cpu_transactions_serialize_process_global_patches(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from XTA import sam_sdk_compat as sdk
        from tests._sam_sdk_transaction_double import TransactionDouble

        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"local checkpoint")
            double = TransactionDouble(checkpoint)
            entered, second_entered, release, second_started = (Event() for _ in range(4))
            calls = []

            def builder(**kwargs):
                calls.append(None)
                if len(calls) == 1:
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("first transaction did not receive release")
                else:
                    second_entered.set()
                return double.builder(**kwargs)

            double.builder_module.build_sam3_predictor = builder
            original_load = double.runtime.load
            original_cuda = double.model_class.cuda

            def construct(second=False):
                if second:
                    second_started.set()
                return sdk._build_real_sam31_predictor_single_load(
                    builder=builder, kwargs={"checkpoint_path": str(checkpoint)},
                    checkpoint_path=checkpoint, runtime_torch=double.runtime,
                    weight_storage="float32", construction_device="meta")

            with mock.patch.dict(sys.modules, {
                "sam3.model.sam3_multiplex_tracking": double.tracking_module,
            }), mock.patch.object(sdk, "_import_sam_model_builder", return_value=double.builder_module):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(construct)
                    try:
                        self.assertTrue(entered.wait(5))
                        second = pool.submit(construct, True)
                        self.assertTrue(second_started.wait(5))
                        self.assertFalse(second_entered.wait(0.1))
                    finally:
                        release.set()
                    self.assertIsNotNone(first.result(timeout=5))
                    self.assertIsNotNone(second.result(timeout=5))
                self.assertTrue(second_entered.is_set())
                self.assertIs(double.runtime.load, original_load)
                self.assertIs(double.model_class.cuda, original_cuda)
                self.assertEqual(len(double.loads), 2)

    def test_bfloat16_storage_audit_limits_fp32_to_exact_decoder_ffn_names(self):
        from XTA import sam_sdk_compat as sdk
        from tests._sam_sdk_transaction_double import Tensor

        runtime = types.SimpleNamespace(float32="float32", bfloat16="bfloat16",
                                        cuda=types.SimpleNamespace(current_device=lambda: 0))
        parameters = [("tracker.weight", Tensor(runtime, device="cuda", dtype="bfloat16")),
                      ("detector.transformer.decoder.layers.0.linear1.weight",
                       Tensor(runtime, device="cuda"))]
        model = types.SimpleNamespace(named_parameters=lambda: iter(parameters),
                                      named_buffers=lambda: iter(()))
        predictor = types.SimpleNamespace(model=model)
        sdk._audit_sam31_predictor_storage(predictor, runtime_torch=runtime,
                                           weight_storage="bfloat16_egpu")
        for name in ("tracker.weight", "detector.transformer.decoder.layers.x.linear1.weight",
                     "detector.transformer.decoder.layers.0.linear3.weight",
                     "detector.transformer.decoder.layers.0.linear1.weight.extra"):
            with self.subTest(name=name):
                parameters[:] = [(name, Tensor(runtime, device="cuda"))]
                with self.assertRaisesRegex(RuntimeError, "expected bfloat16"):
                    sdk._audit_sam31_predictor_storage(predictor, runtime_torch=runtime,
                                                       weight_storage="bfloat16_egpu")
        parameters[:] = [("detector.transformer.decoder.layers.0.linear2.bias",
                          Tensor(runtime, device="cuda", dtype="bfloat16"))]
        with self.assertRaisesRegex(RuntimeError, "expected float32"):
            sdk._audit_sam31_predictor_storage(predictor, runtime_torch=runtime,
                                               weight_storage="bfloat16_egpu")

    def test_bpe_resolution_locates_package_without_executing_sam(self) -> None:
        from XTA import sam_sdk_compat as sdk

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
                with mock.patch.object(sdk, "_PINNED_SAM_BPE_SHA256", digest):
                    resolved = sdk.resolve_installed_sam_bpe()
                self.assertEqual(resolved, bpe.resolve())
                self.assertNotIn("sam3", sys.modules)
            finally:
                sys.path.remove(str(root))
                for name in tuple(sys.modules):
                    if name == "sam3" or name.startswith("sam3."):
                        sys.modules.pop(name, None)
                sys.modules.update(previous)


    def test_builder_import_uses_scoped_pkg_resources_shim_when_absent(self) -> None:
        from XTA import sam_sdk_compat as sdk

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
            original_find_spec = sdk.importlib.util.find_spec

            def find_spec(name, *args, **kwargs):
                if name == "pkg_resources":
                    return None
                return original_find_spec(name, *args, **kwargs)

            sys.path.insert(0, str(root))
            try:
                with (
                    mock.patch.object(
                        sdk.importlib.util, "find_spec", side_effect=find_spec
                    ),
                    mock.patch.object(
                        sdk,
                        "resolve_installed_sam_bpe",
                        return_value=expected_bpe,
                    ),
                ):
                    model_builder = sdk._import_sam_model_builder()
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
        from XTA import sam_sdk_compat as sdk

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
                        sdk,
                        "_sam_pkg_resources_import_compatibility",
                        return_value=nullcontext(),
                    ),
                    mock.patch.object(
                        sdk.importlib, "import_module", side_effect=import_builder
                    ),
                ):
                    self.assertIs(sdk._import_sam_model_builder(), builder_module)
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
                        sdk,
                        "_sam_pkg_resources_import_compatibility",
                        return_value=nullcontext(),
                    ),
                    mock.patch.object(
                        sdk.importlib, "import_module", side_effect=failing_import
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected SAM import failure"):
                        sdk._import_sam_model_builder()
                    self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)

                with (
                    mock.patch.dict(sys.modules, {"sam3.model_builder": builder_module}),
                    mock.patch.object(sdk.importlib, "import_module") as import_module,
                ):
                    self.assertIs(sdk._import_sam_model_builder(), builder_module)
                    import_module.assert_not_called()
                    self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)


    def test_sam31_build_restores_both_tf32_flags(self) -> None:
        from XTA import sam_sdk_compat as sdk

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
                    sdk,
                    "_build_real_sam31_predictor_single_load_unlocked",
                    side_effect=build,
                ):
                    sdk._build_real_sam31_predictor_single_load(
                        builder=build,
                        kwargs={},
                        checkpoint_path=mock.sentinel.checkpoint,
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
                    sdk,
                    "_build_real_sam31_predictor_single_load_unlocked",
                    side_effect=failing_build,
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected SAM build failure"):
                        sdk._build_real_sam31_predictor_single_load(
                            builder=failing_build,
                            kwargs={},
                            checkpoint_path=mock.sentinel.checkpoint,
                            runtime_torch=fake_torch,
                            weight_storage="float32",
                            construction_device="cpu",
                        )
                self.assertEqual((matmul.allow_tf32, cudnn.allow_tf32), initial)


    def test_pkg_resources_compatibility_preserves_existing_module(self) -> None:
        import types

        from XTA import sam_sdk_compat as sdk

        marker = object()
        previous = sys.modules.get("pkg_resources", marker)
        existing = types.ModuleType("pkg_resources")
        sys.modules["pkg_resources"] = existing
        try:
            with sdk._sam_pkg_resources_import_compatibility() as installed:
                self.assertFalse(installed)
                self.assertIs(sys.modules["pkg_resources"], existing)
            self.assertIs(sys.modules["pkg_resources"], existing)
        finally:
            if previous is marker:
                sys.modules.pop("pkg_resources", None)
            else:
                sys.modules["pkg_resources"] = previous


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

        from XTA.sam_sdk_compat import _build_real_sam31_predictor_single_load
        from XTA.lta_sam import resolve_local_sam_bundle

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
                    checkpoint_path=bundle.checkpoint_path,
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
                        checkpoint_path=bundle.checkpoint_path,
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
                        checkpoint_path=bundle.checkpoint_path,
                        runtime_torch=torch,
                        weight_storage="float32",
                        construction_device="meta",
                    )
                self.assertIs(torch.load, original_load)
