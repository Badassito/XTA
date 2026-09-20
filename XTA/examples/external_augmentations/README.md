# External augmentation policies

Each `CPU_*.py` and `GPU_*.py` file is self-contained. CPU files export
`build_augmentation()`; GPU files export
`build_gpu_augmentation(device=..., batch_size=...)`. Both support PTA and
TTA. `baseline` supplies the standard augmentation magnitudes.

Select policies with grouped `--augmentation` values. CPU inference requires
the CPU policy, GPU inference requires the GPU policy, and hybrid inference
requires both:

```text
--augmentation cpu:XTA/examples/external_augmentations/CPU_baseline.py
--augmentation gpu:XTA/examples/external_augmentations/GPU_baseline.py
--augmentation cpu:XTA/examples/external_augmentations/CPU_baseline.py gpu:XTA/examples/external_augmentations/GPU_baseline.py
```

For example:

```text
python -m XTA --mode pta --input INPUT_DIRECTORY --output OUTPUT_DIRECTORY --augmentation cpu:XTA/examples/external_augmentations/CPU_baseline.py --augmentation_ratio 4 --augmentation_execution offline
python -m XTA --mode tta --input INPUT_VIDEO --model gpu:MODEL.engine --augmentation gpu:XTA/examples/external_augmentations/GPU_baseline.py --augmentation_ratio 4
```

## Profile settings

| Parameter | Light | Baseline | Heavy | Superheavy |
| --- | ---: | ---: | ---: | ---: |
| Rotation | ±35° | ±45° | ±55° | ±70° |
| Scale | 0.667–1.5× | 0.5–2× | 0.4–2.5× | 0.333–3× |
| Translation per axis | ±7.5% | ±10% | ±12.5% | ±17.5% |
| Shear per axis | ±22.5° | ±30° | ±37.5° | ±42° |
| Elastic RMS per axis / shorter dimension | 0.5% | 1% | 1.5% | 2% |
| Elastic RMS per axis at 512 pixels | 2.56 px | 5.12 px | 7.68 px | 10.24 px |
| Brightness multiplier | 0.85–1.15× | 0.8–1.2× | 0.75–1.25× | 0.65–1.35× |
| Gaussian blur sigma | 0–3.5 | 0–5 | 0–6.5 | 0–8 |
| Additive Gaussian noise sigma | 0–0.35 | 0–0.5 | 0–0.65 | 0–0.85 |
| Shot-noise strength | 0–0.035 | 0–0.05 | 0–0.065 | 0–0.085 |
| Multiplicative noise | 0.65–1.35× | 0.5–1.5× | 0.35–1.65× | 0.15–1.85× |
| Salt-and-pepper amount | 0–3.5% | 0–5% | 0–6.5% | 0–8.5% |
| CLAHE clip limit, uniform | 1–2 | 1–4 | 2–6 | 3–8 |
| CLAHE tile grid | 8×8 | 8×8 | 8×8 | 8×8 |

D4 rotation/reflection is uniformly selected. Elastic activates for 30% of
augmented copies, brightness for 50%, blur for 25%, salt-and-pepper for 25%,
and CLAHE for 1%. One of the three primary noise families is selected uniformly
for every augmented copy. The superheavy shear limit stays below 45° to keep
the composed two-axis shear away from its singular endpoint.

The elastic field is mean-free and normalized separately on each axis after
smoothing and resizing. Smoothing sigma is 8% of the shorter dimension. The
coarse noise grid has a shorter edge of at most 128 pixels, and the final RMS
scales with the shorter image dimension. Single-row/column inputs have zero
elastic displacement.

## Bit-depth frequencies and mapping

The stage activates at 20% / 30% / 40% / 50% for light through superheavy.
When active, it uniformly chooses one of that profile's available depths.
These frequencies are **per augmented sample**, excluding unaugmented originals:

| Preset | Stage off | 8-bit stretch | 4-bit | 2-bit | 1-bit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Light | 80% | 20% | — | — | — |
| Baseline | 70% | 15% | 15% | — | — |
| Heavy | 60% | 13.333…% | 13.333…% | 13.333…% | — |
| Superheavy | 50% | 12.5% | 12.5% | 12.5% | 12.5% |

Order: spatial resampling → CLAHE → blur/brightness/noise → clamp → adaptive
bit-depth emulation → output conversion. Source zeros are identified immediately
after spatial sampling in the 8-bit bin domain and remain protected throughout
photometry. Segmentation labels only undergo the spatial transforms.

Each image or contextual stack has one local histogram and mapping shared
across channels. Samples in a batch are processed independently. The 8-bit
option stretches the positive local range to 0–255; for example,
`[0,10,20,30,40] → [0,0,85,170,255]`. It preserves 8-bit storage rather than
reducing the number of bits. The 4/2/1-bit options use histogram quantiles and
palettes of multiples of 17, `{0,85,170,255}`, or `{0,255}`, respectively.
Equal intensities stay together. Repeated quantile thresholds can leave palette
levels unused. All-zero and single-positive-bin inputs preserve their eligible
values; constant inputs may therefore be off-palette. Later interpolation or
lossy encoding can introduce intermediate values.

## CLAHE and backend behavior

Baseline uses the numerical defaults from
[Ultralytics' Albumentations CLAHE](https://docs.ultralytics.com/integrations/albumentations/#contrast-limited-adaptive-histogram-equalization-clahe):
1% activation, a uniform clip limit from 1 to 4, and 8×8 tiles. Light uses the
weaker 1–2 clip range. GPU histogram construction, clipping, redistribution,
LUT interpolation, and bit-depth mapping remain on the device.

Grayscale CLAHE follows OpenCV, with possible one-unit rounding differences.
Contextual channels share pooled tile histograms normalized to one tile's area;
replicated grayscale channels reproduce the single-channel result. Channels
retain their correspondence when reordered. RGB is processed through these
shared histograms without a Lab conversion. Protected zeros are restored after
photometry.

CPU and GPU policies select the same parameters for the same profile and integer
seed. Bit-depth and CLAHE use independent seed streams, preserving the established
D4, affine, elastic, brightness, blur, and noise draw order. Backend resamplers
and noise generators can produce different pixels. GPU Gaussian filtering uses
separable horizontal/vertical passes with reflect padding, or replicate padding
when a dimension is too small for the kernel radius.

The GPU policies require CUDA-enabled PyTorch. Set `PTA_GPU_TORCH_COMPILE=0`
to disable optional compilation of the fused pointwise kernel. Numerical tests
run on CPU; actual CUDA policy execution is enabled with:

```text
XTA_RUN_EXTERNAL_AUGMENTATION_CUDA=1 python -m unittest tests.test_external_augmentation_examples -v
```
