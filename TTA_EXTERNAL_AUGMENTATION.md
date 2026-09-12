# External GPU policies in TTA — v22 development branch

TTA reuses the unchanged GPU policy files shipped for PTA. This development
branch is based on the last committed source and intentionally leaves the
concurrent LTA changes for a later merge. Package and launcher version numbers
remain unchanged until release integration.

The native projection schedule is chosen before policy fan-out. TTA defaults
to certified `--projection_sampling coverage`; `dense` retains the older
frame schedule. See [PROJECTION_SAMPLING.md](PROJECTION_SAMPLING.md) for domain
guarantees, frame reductions and admission fallbacks.

## Run

Keep the existing model, input, channel, view, angle, device and save arguments.
Add:

```text
--augmentation XTA/examples/external_augmentations/GPU_light.py
--augmentation_ratio 3
--augmentation_granularity slice
--augmentation_coverage packed
```

Select `nrrd` in `--save` for independent editable layers. GPU-only prediction is
required for N > 1; CPU and hybrid prediction are out of scope. The shipped
policy adapter requires GPU retina flattening and a compatible CuPy/NVRTC
runtime (the existing `cuda12` or `cuda13` extras). Worker setup checks the fused
kernels and fails clearly if they cannot run, instead of silently choosing the
slow Torch inverse. Custom `apply_tta_batch` hooks own their implementation and
are exempt from this CuPy requirement. Use a writable `CUPY_CACHE_DIR` on hosts
whose default cache location is unavailable.

`GPU_light.py`, `GPU_baseline.py`, `GPU_heavy.py` and `GPU_superheavy.py` are
unchanged. Their parameter sampling, affine matrices, elastic fields, and
photometry remain authoritative. PTA and TTA share `--augmentation` and
`--augmentation_ratio`; no duplicate TTA policy directory is needed.

## Passes and output ownership

`--augmentation_ratio N` means one unaugmented pass and N-1 policy passes.
TTA requires a finite integer N >= 1. PTA retains its fractional-ratio behavior.
N=1 uses the established non-policy path and its fast routes.

Pass 0 keeps the existing view name. Additional passes append `__policy_001`,
`__policy_002`, and so on. Angle, physical view, tile configuration and mask
category remain separate identities. A three-pass full-frame run includes:

```text
Transverse_TTA_a0_fullframe_yolo.seg.nrrd
Transverse_TTA_a0__policy_001_fullframe_yolo.seg.nrrd
Transverse_TTA_a0__policy_002_fullframe_yolo.seg.nrrd
```

Actual filenames also contain the input/model naming components. Each pass
retains its own full-frame NRRD even when empty. An empty tile configuration
receives an explicit empty YOLO layer; nonempty tile acceptance categories keep
their existing separate layers. Policy IDs identify outputs; tile raster plans
use the explicit numerical angle, so policy suffixes do not enter angle parsing.

Pass layers are published independently and then contribute to the terminal
physical-view/native union. Only the base pass can create interpolation bridges
or supply base interpolation endpoints. Augmented masks also skip 2-D hole
filling, which would otherwise turn unknown support holes into foreground.
Confidence/radius removal remains available.

There are N full-frame YOLO layers per existing view/angle, plus independent
policy tile layers. The entire output directory is not multiplied by N: bridge
layers remain base-only and global union/checkpoint files remain singular.
Canonical saved input images remain the base rendered inputs.

## Spatial inversion and coverage

Every channel receives the same spatial transform. Photometry is delegated to
the policy and is not undone in masks. Future policies can change channel
photometry without changing the shared spatial contract.

The forward sampler maps augmented coordinate y to original coordinate x:

```text
x = inverse(A) * y + d(y)
```

A fused CUDA kernel builds forward/inverse grids and support without allocating
intermediate tensors for each Newton operation and strip. It retains all 16
Newton iterations, exact bilinear local derivatives, the 0.05-pixel convergence
tolerance, Jacobian orientation/singularity tests, and frame bounds. Fast-math
is disabled. Binary masks and confidence planes are restored with nearest-neighbor
sampling before the existing output-to-processing affine and backprojection.
Invalid inverse support yields zero foreground with its unknown status retained
separately. CUDA bit packing and quantization preserve the policy's uint8
boundary, including rounding behavior. A Torch reference remains available to
math tests and comparison tools.

This is a conservative numerical inverse, not recovery of cropped information
or proof that an arbitrary elastic field is globally one-to-one.

`augmentation_manifest.json` records policy settings, planned output groups,
completed task receipts and support file identities. `augmentation_support/`
contains the policy snapshot, its descriptor, and a compressed `.npz` file for
each augmented task/pass. Support files contain `validity_bits`, `seeds`,
`global_destinations`, `mirror_azimuthal_u` and scalar JSON `metadata`:

```python
import json
import numpy as np

with np.load('support.npz', allow_pickle=False) as archive:
    metadata = json.loads(str(archive['metadata'].item()))
    height, width = metadata['raster_shape']
    valid = np.unpackbits(archive['validity_bits'], axis=-1,
                          count=width, bitorder='big').astype(bool)
```

A valid pixel with no prediction is distinguishable from an invalid pixel.
These maps use the unaugmented model raster before the recorded processing
affine and azimuthal seam mirror. Base support is implicitly the full model
canvas. Source-volume acquisition bounds are not encoded. Future voting must
compose support through the view geometry and combine it with acquisition
coverage; this change does not implement new voting or native support NRRDs.

## Re-roll scope and scheduling

`--augmentation_granularity` accepts:

- `slice` (default): a separate seed for each logical destination slice.
- `slab`: one seed per `--augmentation_slab_slices` logical slices (default 32),
  independent of worker lease boundaries.
- `lease`: one seed per scheduled GPU task; results depend on lease boundaries.
- `view`: one seed per view/angle/tile trajectory.

Every augmented pass has its own seed. Seeds derive from the policy SHA-256,
`--augmentation_seed` (default 0), trajectory, pass, and scope. The policy is
checked during planning, worker execution and successful publication. Recipes
are reproducible; cross-device bitwise neural-network results are not promised.

`--augmentation_cache_mib` defaults to 512 per worker; 0 disables retained map
caching. Repeated seeds within a batch still share one replay. This limit covers
retained maps, not active images, photometric workspaces, or the model.
`--augmentation_coverage packed` is the default; `none` keeps recipes but omits
raster support sidecars.

One rendered GPU batch is reused for base and policy inference before advancing
to the next batch. Radial, Azimuthal and Spherical views are not re-rendered for
policy copies. Existing CPU render fallback can upload once; no GPU-image
round trip is made for CPU augmentation.

At most two mask retirement futures and two pinned support-batch transfers are
pending per active task. CUDA events fence background support writes; duplicate
replays within a batch share packing. Batch occupancy/bounding-box metadata is
merged by task-local offset so later cleanup/publication retains its scan
shortcuts. Missing metadata from a seam or host path selects the safe scan path.

Task-end mask retirement and support compression return through the existing
deferred worker publication mechanism. Two task permits bound outstanding
policy publication buffers per GPU worker, allowing the next task to infer while
the previous task finishes saving. Failure paths drain copies and mask work
before releasing maps. Existing NRRD publication remains asynchronous.

For N > 1, affine-only D1/direct-union and the resident TensorRT ring remain
bypassed. Full-frame planning charges all policy-parent canvases, and tile
admission charges all siblings. Large ratios/view/angle sets can hit the
retained-dense memory guard. Grouped policy tasks cannot be dynamically split.

## Qualification

GPU mathematics, without a model or volume:

```bash
python tools/tta_augmentation_smoke.py --device 0 --channel_format RGB \
  --batch 4 --imgsz 256 --report /scratch/tta/math.json
```

The probe compares forward geometry before stochastic photometry, compares
photometry using identical spatial input, checks deterministic replay, validates
inverse residuals and rejects unknown-support foreground. A post-photometry
original-policy difference is diagnostic: Poisson draws can amplify tiny grid
roundoff. Use `--compile` to exercise the policy's requested Torch compilation
mode. The policy itself may fall back when Torch compilation is unavailable.
Exit 2 means CUDA was unavailable, not a pass. Channel tokens follow the main
CLI (`gray`/`grey`, `RGB`, `C5S2`, etc.); synthetic input tests channel count,
not neighboring-volume-slice sampling.

A real small Cartesian/Azimuthal run with tiling, base interpolation, independent
NRRD decoding and exact component-union verification:

```bash
python tools/qualify_tta_augmentation.py --model /models/best.pt \
  --device 0 --channel_format gray --batch 4 --imgsz 128 \
  --output /scratch/tta/tiled-qualification
```

Use a fresh output directory and a model matching the requested channel format.
The synthetic fixture tests orchestration and publication, not segmentation
quality. Header parsing requires pynrrd. The completed-run checker also supports
real user outputs:

```bash
python tools/check_tta_augmentation_run.py /path/to/completed/output
```

The checker requires planned full-frame/tile groups to match completed execution
and all published policy passes. It rejects an entirely missing tile group,
missing pass, duplicate filename or augmented interpolation layer. Older r1/r2
manifests lack the required planning record and must be regenerated. The small
qualification tool additionally decodes all gzip members and checks exact OR
recomposition; the manifest-only checker does not judge mask content.

The repeatable adapter benchmark heatsoaks the GPU and writes all evidence to
its explicit output directory:

```bash
python tools/benchmark_tta_augmentation.py --device 0 --channel_format RGB \
  --batch 4 --imgsz 3072 --reference --output /scratch/tta/benchmark
```

It compares fresh slice seeds and cached transforms with the original Torch
solver. Timings include augmentation, packed support transfer and one binary
mask inverse; they exclude the model, volume rendering and NRRD compression.
Validate the actual model/backend, projection families and production volumes
on the target cluster before treating local timings as pipeline throughput.

Implementation owners: `augmentation_policy` and `tta_augmentation_config` own
shared identity/settings; `tta_augmentation` adapts policies; `tta_augmentation_cuda`
owns fused kernels; `tta_augmentation_retirement` owns bounded transfers and
metadata; `tta_augmentation_runtime` drives batch fan-out. An optional policy
hook `apply_tta_batch(images=..., seeds=...)` returns normalized NCHW images,
float32 NHW2 inverse grids (align-corners coordinates), and boolean NHW validity
on the same device. Unknown policies without the shipped contract or this hook
fail clearly.
