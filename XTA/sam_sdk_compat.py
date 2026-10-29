"""Compatibility boundary for the audited, pinned SAM SDK.

Imports stay dependency-light. Runtime imports, local vocabulary validation,
and process-global construction patches live here so their locks and exact
restoration rules have one owner. Model/session policy remains in ``lta_sam``.
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import threading
import types
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

PINNED_SAM_VERSION = "0.1.0"
PINNED_SAM_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
PINNED_SAM_SOURCE_URL = "https://github.com/facebookresearch/sam3.git"
PINNED_SAM_PACKAGE_FILE_COUNT = 153
PINNED_SAM_PACKAGE_TREE_SHA256 = (
    "6addad49cab67f65895cea072512e7d9cd0f5cfb162e102a60c56a4299075aaf"
)
_PINNED_SAM_BPE_SHA256 = (
    "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
)
_PINNED_SAM_BPE_PREFIX = b'"bpe_simple_vocab_16e6.txt#version: 0.2\n'
_SAM_BPE_RESOURCE = "assets/bpe_simple_vocab_16e6.txt.gz"

# The prepared-frame loader in lta_sam also replaces pinned SDK globals. It
# shares this reentrant lock with builder imports and construction transactions.
SAM_SDK_PATCH_LOCK = threading.RLock()
_SAM_PKG_RESOURCES_COMPAT_LOCK = threading.RLock()


@contextmanager
def _preserve_torch_tf32(runtime_torch: object) -> Iterator[None]:
    """Undo SAM's process-wide TF32 changes after an import or construction."""

    matmul = runtime_torch.backends.cuda.matmul
    cudnn = runtime_torch.backends.cudnn
    original_matmul = bool(matmul.allow_tf32)
    original_cudnn = bool(cudnn.allow_tf32)
    try:
        yield
    finally:
        matmul.allow_tf32 = original_matmul
        cudnn.allow_tf32 = original_cudnn


def _canonical_sam_package_fingerprint(distribution: object) -> tuple[str, int, Path]:
    """Fingerprint pinned runtime files independently of wheel metadata."""

    try:
        package_root = Path(distribution.locate_file("sam3")).resolve(strict=True)
    except Exception as exc:
        raise RuntimeError("sam3 installation has no locatable package source tree") from exc
    if not package_root.is_dir():
        raise RuntimeError(f"sam3 package source root is not a directory: {package_root}")
    files = sorted(
        (
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() != ".pyc"
        ),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        payload = path.read_bytes()
        if path.suffix.lower() == ".py":
            payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\n")
    return digest.hexdigest(), len(files), package_root


def resolve_pinned_sam_runtime_provenance(
    *,
    pinned_version: str = PINNED_SAM_VERSION,
    pinned_commit: str = PINNED_SAM_COMMIT,
    pinned_source_url: str = PINNED_SAM_SOURCE_URL,
    pinned_package_file_count: int = PINNED_SAM_PACKAGE_FILE_COUNT,
    pinned_package_tree_sha256: str = PINNED_SAM_PACKAGE_TREE_SHA256,
    bpe_resolver: Optional[Callable[[], Path]] = None,
) -> dict[str, object]:
    """Fail closed unless the installed private-API runtime is exactly pinned."""

    try:
        distribution = importlib.metadata.distribution("sam3")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("the pinned local sam3 distribution is not installed") from exc
    version = str(distribution.version)
    if version != str(pinned_version):
        raise RuntimeError(
            "sam3 runtime does not match the pinned adapter: "
            f"version={version!r}, expected={str(pinned_version)!r}"
        )
    direct_text = distribution.read_text("direct_url.json")
    direct: Mapping[str, object] = {}
    if direct_text:
        try:
            parsed = json.loads(direct_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            direct = parsed
    vcs_info = direct.get("vcs_info")
    recorded_vcs = (
        str(vcs_info.get("vcs", "")).strip().lower()
        if isinstance(vcs_info, Mapping)
        else ""
    )
    recorded_commit = (
        str(vcs_info.get("commit_id", "")).strip()
        if isinstance(vcs_info, Mapping)
        else ""
    )
    if recorded_vcs and recorded_vcs != "git":
        raise RuntimeError(f"sam3 installation records unsupported VCS {recorded_vcs!r}")
    if recorded_commit and recorded_commit != str(pinned_commit):
        raise RuntimeError(
            "sam3 runtime does not match the pinned adapter: "
            f"version={version!r}, commit={recorded_commit!r}"
        )
    tree_digest, package_file_count, package_root = _canonical_sam_package_fingerprint(
        distribution
    )
    spec = importlib.util.find_spec("sam3")
    if spec is None or spec.origin is None:
        raise RuntimeError("the installed sam3 runtime is not importable")
    try:
        Path(spec.origin).resolve(strict=True).relative_to(package_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "the importable sam3 runtime is outside the audited distribution package"
        ) from exc
    if (
        package_file_count != int(pinned_package_file_count)
        or tree_digest != str(pinned_package_tree_sha256)
    ):
        raise RuntimeError(
            "sam3 installed package tree does not match the pinned adapter: "
            f"files={package_file_count}, sha256={tree_digest!r}"
        )
    resolve_bpe = resolve_installed_sam_bpe if bpe_resolver is None else bpe_resolver
    bpe = Path(resolve_bpe()).resolve(strict=True)
    return {
        "distribution_version": version,
        "git_commit": recorded_commit or None,
        "pinned_git_commit": str(pinned_commit),
        "provenance_method": (
            "pep610_vcs_plus_source_tree"
            if recorded_commit
            else "pinned_installed_source_tree"
        ),
        "source_url": str(direct.get("url") or "").strip() or None,
        "pinned_source_url": str(pinned_source_url),
        "direct_url_present": bool(direct_text),
        "package_root": str(package_root),
        "package_file_count": package_file_count,
        "package_tree_sha256": tree_digest,
        "bpe_path": str(bpe),
        "bpe_sha256": hashlib.sha256(bpe.read_bytes()).hexdigest(),
    }


def _resolve_installed_sam_package_root() -> Path:
    """Locate the installed package without executing ``sam3.__init__``."""

    try:
        spec = importlib.util.find_spec("sam3")
    except (ImportError, AttributeError, ValueError) as exc:
        raise RuntimeError("the pinned local sam3 package is not locatable") from exc
    locations = () if spec is None else tuple(spec.submodule_search_locations or ())
    if len(locations) != 1:
        raise RuntimeError(
            "the pinned local sam3 package must have exactly one source root; "
            f"found {len(locations)}"
        )
    try:
        root = Path(locations[0]).resolve(strict=True)
    except Exception as exc:
        raise RuntimeError("the pinned local sam3 package source root is missing") from exc
    if not root.is_dir():
        raise RuntimeError(f"the pinned local sam3 package root is not a directory: {root}")
    return root


def resolve_installed_sam_bpe(*, package_root: Optional[Path] = None) -> Path:
    """Resolve the pinned SAM package's local BPE asset without network access."""

    if package_root is None:
        package_root = _resolve_installed_sam_package_root()
    candidate = Path(package_root) / _SAM_BPE_RESOURCE
    try:
        resolved = candidate.resolve(strict=True)
    except Exception as exc:
        raise RuntimeError(f"local SAM BPE asset is missing: {candidate}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"local SAM BPE asset is not a file: {resolved}")
    return _validate_pinned_sam_bpe(resolved)


def _sam_pkg_resources_shim() -> types.ModuleType:
    """Provide only the one legacy resource lookup used by pinned SAM 3.1."""

    shim = types.ModuleType("pkg_resources")

    def resource_filename(package: object, resource: object) -> str:
        package_name = str(package)
        resource_name = str(resource).replace("\\", "/")
        if package_name != "sam3" or resource_name != _SAM_BPE_RESOURCE:
            raise RuntimeError(
                "the SAM pkg_resources compatibility shim accepts only the pinned "
                f"BPE resource; got package={package_name!r}, resource={resource_name!r}"
            )
        return str(resolve_installed_sam_bpe())

    shim.resource_filename = resource_filename  # type: ignore[attr-defined]
    return shim


@contextmanager
def _sam_pkg_resources_import_compatibility() -> Iterator[bool]:
    """Temporarily satisfy SAM's obsolete top-level import under Setuptools 82+.

    The official builder retains its reference to the narrow module object. The
    temporary ``sys.modules`` entry is removed immediately after the builder
    import so unrelated libraries never observe a project-wide replacement for
    ``pkg_resources``.
    """

    with _SAM_PKG_RESOURCES_COMPAT_LOCK:
        if "pkg_resources" in sys.modules:
            if sys.modules["pkg_resources"] is None:
                raise RuntimeError("pkg_resources has an invalid null module entry")
            yield False
            return
        try:
            pkg_resources_spec = importlib.util.find_spec("pkg_resources")
        except (ImportError, AttributeError, ValueError) as exc:
            raise RuntimeError("could not inspect pkg_resources availability") from exc
        if pkg_resources_spec is not None:
            yield False
            return
        shim = _sam_pkg_resources_shim()
        installed = sys.modules.setdefault("pkg_resources", shim)
        if installed is not shim:
            if installed is None:
                raise RuntimeError("pkg_resources acquired an invalid null module entry")
            yield False
            return
        try:
            yield True
        finally:
            if sys.modules.get("pkg_resources") is shim:
                sys.modules.pop("pkg_resources", None)


def _import_sam_model_builder() -> object:
    """Import the pinned builder without retaining its global TF32 changes."""

    with SAM_SDK_PATCH_LOCK:
        cached = sys.modules.get("sam3.model_builder")
        if cached is not None:
            return cached
        import torch as runtime_torch  # type: ignore

        with _preserve_torch_tf32(runtime_torch), _sam_pkg_resources_import_compatibility():
            return importlib.import_module("sam3.model_builder")


def _validate_pinned_sam_bpe(path: Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != _PINNED_SAM_BPE_SHA256:
        raise RuntimeError(
            "local SAM BPE asset does not match the pinned vocabulary: "
            f"{resolved}"
        )
    try:
        with gzip.open(resolved, "rb") as handle:
            prefix = handle.read(len(_PINNED_SAM_BPE_PREFIX))
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"local SAM BPE asset is not readable gzip: {resolved}") from exc
    if prefix != _PINNED_SAM_BPE_PREFIX:
        raise RuntimeError(f"local SAM BPE vocabulary header is invalid: {resolved}")
    return resolved


def _same_local_path(value: object, expected: Path) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    try:
        return Path(value).resolve(strict=True) == Path(expected).resolve(strict=True)
    except Exception:
        return False


def _approved_egpu_fp32_name(name: str) -> bool:
    parts = str(name).split(".")
    return (
        len(parts) == 7
        and parts[:4] == ["detector", "transformer", "decoder", "layers"]
        and parts[4].isdigit()
        and parts[5] in {"linear1", "linear2"}
        and parts[6] in {"weight", "bias"}
    )


def _audit_sam31_predictor_storage(
    predictor: object,
    *,
    runtime_torch: object,
    weight_storage: str,
) -> None:
    model = getattr(predictor, "model", None)
    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        raise RuntimeError("the assembled SAM 3.1 predictor exposes no tensor inventory")
    tensors = tuple(named_parameters()) + tuple(named_buffers())
    if not tensors:
        raise RuntimeError("the assembled SAM 3.1 predictor tensor inventory is empty")
    current_device = int(runtime_torch.cuda.current_device())
    placement_errors = []
    dtype_errors = []
    for name, tensor in tensors:
        device = getattr(tensor, "device", None)
        device_type = str(getattr(device, "type", ""))
        get_device = getattr(tensor, "get_device", None)
        tensor_device = int(get_device()) if callable(get_device) else -1
        if bool(getattr(tensor, "is_meta", False)) or (
            device_type != "cuda" or tensor_device != current_device
        ):
            placement_errors.append(f"{name}:{device}")
            continue
        is_floating_point = getattr(tensor, "is_floating_point", None)
        if not callable(is_floating_point) or not bool(is_floating_point()):
            continue
        dtype = getattr(tensor, "dtype", None)
        if weight_storage == "float32":
            if dtype != runtime_torch.float32:
                dtype_errors.append(f"{name}:{dtype}")
        elif _approved_egpu_fp32_name(name):
            if dtype != runtime_torch.float32:
                dtype_errors.append(f"{name}:{dtype} (expected float32)")
        elif dtype != runtime_torch.bfloat16:
            dtype_errors.append(f"{name}:{dtype} (expected bfloat16)")
    if placement_errors or dtype_errors:
        raise RuntimeError(
            "assembled SAM 3.1 tensor audit failed: "
            f"placement={placement_errors[:12]}, dtype={dtype_errors[:12]}"
        )


def _build_real_sam31_predictor_single_load_unlocked(
    *,
    builder: Callable[..., object],
    kwargs: Mapping[str, object],
    checkpoint_path: Path,
    runtime_torch: object,
    weight_storage: str,
    construction_device: str,
) -> object:
    """Build the merged predictor without the upstream throwaway checkpoint load.

    The official SAM 3.1 builder first loads the merged 3.5 GiB predictor state
    into a tracker-only model, where every ``tracker.*``/``detector.*`` key is
    incompatible, then loads the same file again into the assembled predictor.
    That redundant materialization can terminate a 16 GiB Windows host.  This
    adapter leaves model construction upstream-owned, skips only that known
    incompatible first load, builds the parameter shell on CPU, assigns the
    memory-mapped assembled state without a second copy, audits its exact
    state-dict result, and chunks the final host-to-CUDA tensor copies so a
    Windows eGPU does not need one large BAR1 transfer mapping.
    """

    try:
        sam_model_builder = _import_sam_model_builder()
        from sam3.model.sam3_multiplex_tracking import (  # type: ignore
            Sam3MultiplexTrackingWithInteractivity,
        )
    except Exception as exc:  # pragma: no cover - real dependency gated
        raise RuntimeError("the pinned SAM 3.1 builder internals are unavailable") from exc

    official_builder = getattr(sam_model_builder, "build_sam3_predictor", None)
    tracker_builder = getattr(
        sam_model_builder, "build_sam3_multiplex_video_model", None
    )
    torch_load = getattr(runtime_torch, "load", None)
    if builder is not official_builder or not callable(tracker_builder) or not callable(torch_load):
        raise RuntimeError("the pinned SAM 3.1 builder identity has changed")

    checkpoint = Path(checkpoint_path).resolve(strict=True)
    original_state_loader = Sam3MultiplexTrackingWithInteractivity.load_state_dict
    original_cuda = Sam3MultiplexTrackingWithInteractivity.cuda
    torch_linspace = getattr(runtime_torch, "linspace", None)
    if not callable(torch_linspace):
        raise RuntimeError("the pinned PyTorch runtime exposes no linspace constructor")
    checkpoint_load_count = 0
    meta_drop_path_linspace_count = 0
    state_audits: list[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []

    def tracker_without_merged_checkpoint(*args: object, **inner: object) -> object:
        requested = inner.get("checkpoint_path")
        if not _same_local_path(requested, checkpoint):
            raise RuntimeError(
                "SAM 3.1 tracker construction did not receive the selected local checkpoint"
            )
        inner["checkpoint_path"] = None
        inner["load_from_HF"] = False
        inner["device"] = construction_device
        return tracker_builder(*args, **inner)

    def cpu_scalar_linspace(*args: object, **inner: object) -> object:
        nonlocal meta_drop_path_linspace_count
        # The pinned ViT constructor materializes its stochastic-depth schedule
        # with ``[x.item() for x in torch.linspace(0, 0.1, 32)]``.  A meta
        # tensor cannot service ``item()``, so permit exactly that scalar-only
        # constructor call on CPU.  Do not turn this into a general escape hatch
        # for tensors that should remain meta until the checkpoint is assigned.
        if (
            len(args) != 3
            or inner
            or float(args[0]) != 0.0
            or not math.isclose(float(args[1]), 0.1, rel_tol=0.0, abs_tol=1e-12)
            or isinstance(args[2], bool)
            or int(args[2]) != 32
        ):
            raise RuntimeError(
                "unexpected torch.linspace call during pinned SAM 3.1 meta construction"
            )
        meta_drop_path_linspace_count += 1
        inner["device"] = "cpu"
        return torch_linspace(*args, **inner)

    def mmap_local_checkpoint(file: object, *args: object, **inner: object) -> object:
        nonlocal checkpoint_load_count
        if _same_local_path(file, checkpoint):
            checkpoint_load_count += 1
            inner["mmap"] = True
            inner["map_location"] = "cpu"
            inner["weights_only"] = True
        return torch_load(file, *args, **inner)

    def audited_state_load(self: object, *args: object, **inner: object) -> object:
        if construction_device == "meta":
            state_dict = args[0] if args else inner.get("state_dict")
            if not isinstance(state_dict, Mapping):
                raise RuntimeError(
                    "SAM meta construction received no mapping state dictionary"
                )
            named_buffers = getattr(self, "named_buffers", None)
            if not callable(named_buffers):
                raise RuntimeError("SAM meta construction exposes no buffer inventory")
            state_keys = frozenset(str(key) for key in state_dict)
            constructor_only_buffers = [
                (
                    str(name),
                    tuple(int(value) for value in tensor.shape),
                    str(getattr(tensor, "device", "")),
                )
                for name, tensor in named_buffers()
                if int(tensor.numel()) > 0 and str(name) not in state_keys
            ]
            expected_constructor_only = [
                (
                    "detector.backbone.language_backbone.encoder.attn_mask",
                    (32, 32),
                    "meta",
                )
            ]
            if constructor_only_buffers not in ([], expected_constructor_only):
                raise RuntimeError(
                    "SAM meta construction created non-checkpoint buffers outside the "
                    f"pinned causal mask: {constructor_only_buffers[:20]}"
                )
        inner["assign"] = True
        result = original_state_loader(self, *args, **inner)
        missing = tuple(str(value) for value in getattr(result, "missing_keys", ()))
        unexpected = tuple(str(value) for value in getattr(result, "unexpected_keys", ()))
        state_audits.append((missing, unexpected))
        if weight_storage == "bfloat16_egpu":
            cast = getattr(self, "bfloat16", None)
            if not callable(cast):
                raise RuntimeError("the assembled SAM 3.1 model cannot cast to bfloat16")
            cast()
            detector = getattr(self, "detector", None)
            transformer = getattr(detector, "transformer", None)
            decoder = getattr(transformer, "decoder", None)
            layers = tuple(getattr(decoder, "layers", ()))
            if not layers:
                raise RuntimeError("the pinned SAM decoder exposes no FFN layers")
            fp32_modules = 0
            for layer in layers:
                for name in ("linear1", "linear2"):
                    module = getattr(layer, name, None)
                    to_float = getattr(module, "float", None)
                    if not callable(to_float):
                        raise RuntimeError(
                            f"the pinned SAM decoder layer exposes no {name} FP32 boundary"
                        )
                    to_float()
                    fp32_modules += 1
            if fp32_modules != len(layers) * 2:
                raise RuntimeError("the pinned SAM decoder FFN FP32 audit failed")
        return result

    def chunked_cuda(self: object, device: object = None) -> object:
        cuda = getattr(runtime_torch, "cuda")
        torch_device = getattr(runtime_torch, "device")
        target = torch_device(
            "cuda",
            int(cuda.current_device()) if device is None else int(device),
        )
        chunk_bytes = 4 * 1024 * 1024
        if construction_device == "meta":
            named_buffers = getattr(self, "named_buffers", None)
            residual = []
            if callable(named_buffers):
                residual = [
                    (str(name), tuple(int(value) for value in tensor.shape), str(tensor.dtype))
                    for name, tensor in named_buffers()
                    if bool(getattr(tensor, "is_meta", False)) and int(tensor.numel()) > 0
                ]
            expected_causal = (
                "detector.backbone.language_backbone.encoder.attn_mask",
                (32, 32),
            )
            residual_identity = [(name, shape) for name, shape, _dtype in residual]
            if residual_identity == [expected_causal]:
                encoder = self.detector.backbone.language_backbone.encoder
                build_causal_mask = getattr(encoder, "build_causal_mask", None)
                if not callable(build_causal_mask):
                    raise RuntimeError("SAM text encoder exposes no causal-mask builder")
                with runtime_torch.device("cpu"):
                    causal_mask = build_causal_mask()
                if weight_storage == "bfloat16_egpu":
                    causal_mask = causal_mask.to(dtype=runtime_torch.bfloat16)
                encoder._buffers["attn_mask"] = causal_mask
                residual = []
            if residual:
                raise RuntimeError(
                    "SAM meta construction left initialized buffers absent from the checkpoint: "
                    f"{residual[:20]}"
                )

        def move_tensor(tensor: object) -> object:
            if bool(getattr(tensor, "is_cuda", False)):
                return tensor.to(device=target)
            if bool(getattr(tensor, "is_meta", False)) and int(tensor.numel()) > 0:
                raise RuntimeError(
                    "SAM meta construction left an initialized tensor absent from the checkpoint"
                )
            destination = runtime_torch.empty_like(tensor, device=target)
            numel = int(tensor.numel())
            if numel == 0:
                return destination
            if int(tensor.element_size()) * numel <= chunk_bytes:
                destination.copy_(tensor, non_blocking=False)
                return destination
            if not bool(tensor.is_contiguous()) or not bool(destination.is_contiguous()):
                destination.copy_(tensor, non_blocking=False)
                return destination
            chunk_elements = max(1, chunk_bytes // int(tensor.element_size()))
            source_flat = tensor.view(-1)
            destination_flat = destination.view(-1)
            for start in range(0, numel, chunk_elements):
                stop = min(numel, start + chunk_elements)
                destination_flat[start:stop].copy_(
                    source_flat[start:stop],
                    non_blocking=False,
                )
            return destination

        apply = getattr(self, "_apply", None)
        if not callable(apply):
            raise RuntimeError("the assembled SAM 3.1 model has no tensor apply boundary")
        return apply(move_tensor)

    try:
        setattr(
            sam_model_builder,
            "build_sam3_multiplex_video_model",
            tracker_without_merged_checkpoint,
        )
        setattr(runtime_torch, "load", mmap_local_checkpoint)
        if construction_device == "meta":
            setattr(runtime_torch, "linspace", cpu_scalar_linspace)
        setattr(
            Sam3MultiplexTrackingWithInteractivity,
            "load_state_dict",
            audited_state_load,
        )
        setattr(Sam3MultiplexTrackingWithInteractivity, "cuda", chunked_cuda)
        if construction_device == "meta":
            with runtime_torch.device("meta"):
                predictor = builder(**dict(kwargs))
        else:
            predictor = builder(**dict(kwargs))
    finally:
        setattr(sam_model_builder, "build_sam3_multiplex_video_model", tracker_builder)
        setattr(runtime_torch, "load", torch_load)
        setattr(runtime_torch, "linspace", torch_linspace)
        setattr(
            Sam3MultiplexTrackingWithInteractivity,
            "load_state_dict",
            original_state_loader,
        )
        setattr(Sam3MultiplexTrackingWithInteractivity, "cuda", original_cuda)

    try:
        if construction_device == "meta" and meta_drop_path_linspace_count != 2:
            raise RuntimeError(
                "the pinned SAM 3.1 meta constructor did not materialize exactly two "
                "CPU scalar drop-path schedules; observed "
                f"{meta_drop_path_linspace_count}"
            )
        if checkpoint_load_count != 1:
            raise RuntimeError(
                "the pinned SAM 3.1 builder did not perform exactly one local merged "
                f"checkpoint load; observed {checkpoint_load_count}"
            )
        if len(state_audits) != 1:
            raise RuntimeError(
                "the pinned SAM 3.1 builder did not expose exactly one assembled "
                f"state-dict audit; observed {len(state_audits)}"
            )
        missing, unexpected = state_audits[0]
        if missing or unexpected:
            raise RuntimeError(
                "local SAM 3.1 checkpoint does not exactly match the assembled predictor: "
                f"missing={len(missing)} {list(missing[:12])}, "
                f"unexpected={len(unexpected)} {list(unexpected[:12])}"
            )
        _audit_sam31_predictor_storage(
            predictor,
            runtime_torch=runtime_torch,
            weight_storage=weight_storage,
        )
    except BaseException:
        failed_predictor = predictor
        predictor = None
        shutdown = getattr(failed_predictor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        del failed_predictor
        gc.collect()
        cuda = getattr(runtime_torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        raise
    return predictor


def _build_real_sam31_predictor_single_load(
    *,
    builder: Callable[..., object],
    kwargs: Mapping[str, object],
    checkpoint_path: Path,
    runtime_torch: object,
    weight_storage: str,
    construction_device: str,
) -> object:
    """Serialize the process-global pinned SAM builder patch transaction."""

    with SAM_SDK_PATCH_LOCK, _preserve_torch_tf32(runtime_torch):
        try:
            return _build_real_sam31_predictor_single_load_unlocked(
                builder=builder,
                kwargs=kwargs,
                checkpoint_path=checkpoint_path,
                runtime_torch=runtime_torch,
                weight_storage=weight_storage,
                construction_device=construction_device,
            )
        except BaseException:
            gc.collect()
            cuda = getattr(runtime_torch, "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
            raise
