"""CPU-only SAM construction double; no torch import or CUDA initialization."""

from pathlib import Path
from types import ModuleType, SimpleNamespace


class Device:
    def __init__(self, kind, index=0):
        self.type, self.index = kind, index

    def __str__(self):
        return self.type

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Tensor:
    def __init__(self, runtime, size=1, device="cpu", dtype="float32", shape=None):
        self.runtime, self.size = runtime, size
        self.device, self.dtype = Device(device), dtype
        self.shape = shape or (size,)

    @property
    def is_meta(self):
        return self.device.type == "meta"

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def numel(self):
        return self.size

    def element_size(self):
        return 4

    def is_contiguous(self):
        return True

    def is_floating_point(self):
        return True

    def get_device(self):
        return self.device.index

    def view(self, *args):
        return self

    def __getitem__(self, item):
        return Tensor(self.runtime, item.stop-item.start, str(self.device), self.dtype)

    def copy_(self, source, *, non_blocking):
        assert not non_blocking
        self.runtime.copies.append(source.numel())
        if self.runtime.fail_copy:
            raise RuntimeError("injected tensor copy failure")
        return self


class TransactionDouble:
    def __init__(self, checkpoint: Path, *, construction_device="meta", mode="ok"):
        self.checkpoint, self.construction_device, self.mode = checkpoint, construction_device, mode
        self.loads, self.tracker_calls, self.states, self.schedules = [], [], [], []
        self.shutdown_count = 0
        owner = self
        self.runtime = SimpleNamespace(
            float32="float32", bfloat16="bfloat16", copies=[], fail_copy=mode == "copy_failure",
            cuda=SimpleNamespace(current_device=lambda: 0, empty_cache=lambda: None),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                     cudnn=SimpleNamespace(allow_tf32=True)),
            device=Device,
        )

        def load(file, *args, **kwargs):
            self.loads.append((file, kwargs))
            if mode == "load_failure":
                raise RuntimeError("injected checkpoint load failure")
            return {"probe": Tensor(self.runtime, 1048583)}

        def linspace(*args, **kwargs):
            self.schedules.append((args, kwargs))
            return object()

        self.runtime.load, self.runtime.linspace = load, linspace
        self.runtime.empty_like = lambda tensor, device: Tensor(
            self.runtime, tensor.numel(), str(device), tensor.dtype)

        class Model:
            def __init__(self):
                self.probe = Tensor(owner.runtime, device=construction_device)
                self.buffers = []
                if mode == "unexpected_buffer":
                    self.buffers = [("unexpected", Tensor(owner.runtime, device="meta"))]

            def load_state_dict(self, state, **kwargs):
                owner.states.append(kwargs)
                self.probe = state["probe"]
                if mode == "placement":
                    self.probe.device = Device("cpu")
                if mode == "dtype":
                    self.probe.dtype = "float16"
                return SimpleNamespace(missing_keys=["missing"] if mode == "missing" else [],
                                       unexpected_keys=["extra"] if mode == "unexpected" else [])

            def cuda(self, device=None):
                raise AssertionError("SDK transaction did not intercept cuda")

            def named_buffers(self):
                return iter(self.buffers)

            def named_parameters(self):
                return iter([("probe", self.probe)])

            def _apply(self, move):
                self.probe = move(self.probe)
                return self

        self.model_class = Model
        self.builder_module = ModuleType("sam3.model_builder")

        def tracker(**kwargs):
            self.tracker_calls.append(kwargs)

        def build(**kwargs):
            self.runtime.backends.cuda.matmul.allow_tf32 = True
            self.runtime.backends.cudnn.allow_tf32 = False
            if mode == "builder_failure":
                raise RuntimeError("injected builder failure")
            self.builder_module.build_sam3_multiplex_video_model(
                checkpoint_path=kwargs["checkpoint_path"], load_from_HF=True)
            if construction_device == "meta":
                for _ in range(1 if mode == "schedule_count" else 2):
                    self.runtime.linspace(0, 0.2 if mode == "schedule_shape" else 0.1, 32)
            state = None
            for _ in range(0 if mode == "no_load" else 2 if mode == "double_load" else 1):
                state = self.runtime.load(kwargs["checkpoint_path"], map_location="cuda",
                                          mmap=False, weights_only=False)
            model = Model()
            if state is not None and mode != "no_state":
                for _ in range(2 if mode == "double_state" else 1):
                    model.load_state_dict(state, strict=False)
                if mode != "placement":
                    model.cuda()
            self.predictor = SimpleNamespace(model=model, shutdown=self.shutdown)
            return self.predictor

        self.builder = build
        self.builder_module.build_sam3_predictor = build
        self.builder_module.build_sam3_multiplex_video_model = tracker
        self.tracking_module = ModuleType("sam3.model.sam3_multiplex_tracking")
        self.tracking_module.Sam3MultiplexTrackingWithInteractivity = Model

    def shutdown(self):
        self.shutdown_count += 1
