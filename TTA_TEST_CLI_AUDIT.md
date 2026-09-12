# Test-tool CLI audit — external TTA r2

Scope: static review of argument declarations and related call sites in all 58
Python tools and the one test module shipped in the r1 archive, compared with
`XTA/config.py` (especially `resolve_channel_format` and `build_argparser`).
The new r2 CLI regression module is additional. This is a CLI audit, not a
qualification of those older tools, their dependencies or their fixtures.

## Requested options

| Input | Main TTA CLI | r1 augmentation probe | r2 augmentation probe |
| --- | --- | --- | --- |
| GPU selector | `--device 0` (also supports multi-device/backend selection) | `--device cuda:0` | `--device 0` (one logical GPU only) |
| Channel layout | `--channel_format grey`, `RGB`, or `C5S2` | `--channels 3` | Same flag and token grammar as main |
| Square image size | `--imgsz 256` | `--size 256` | `--imgsz 256` |
| Batch | `--batch 4` (also supports tagged backend values) | `--batch 4` | `--batch 4` (integer math-probe batch only) |

The probe reuses the existing lightweight channel-format resolver. The custom
channel count must be odd and positive; stride must be a positive integer.
`grey` canonicalizes to `gray`. Inputs remain independently randomized synthetic
channels. Only their count comes from the token; stride is validated/recorded,
not used for neighboring-slice sampling. RGB duplication is not exercised.
Probe defaults remain GPU 0, 256 x 256, batch 4, three channels (`RGB`). The
minimum raster size remains 16 and the minimum batch remains 2.

The probe's JSON report now uses `device`, `imgsz`, `channel_format`,
`channel_count` and `channel_stride`. The old `size` and numeric `channels`
report fields are replaced. Old CLI spellings are not compatibility aliases.

## Same discrepancies elsewhere

**Device syntax:** The other 20 tools declaring `--device` already use an integer
logical index, normally defaulting to 0. No other declared `--device` option uses
`cuda:0` as its CLI value/default. Internal Python calls using `cuda:0` are
PyTorch device descriptors, not CLI discrepancies, and should not be renamed.
Some older wrappers hard-code GPU 0 instead of exposing a device option.

**Channel naming:** No other tool declares a numeric `--channels` option.
`qualify_native_trt_pipeline.py` and `qualify_radial_owner_pipeline.py` construct
main-pipeline invocations with `--channel_format grey` already, but hard-code
that choice rather than exposing a channel-format selector. Other fixed-channel
or engine-specific fixtures are not general channel-configuration tests.

**Size naming:** `tools/benchmark_topology_runs.py:22` still declares `--size`
(default 3072). It sizes synthetic square label planes for a topology/adjacency
benchmark, not model inference. This is the remaining same-name inconsistency
for a square raster. It is reported here, not changed in this narrowly scoped
revision. The five existing tools that expose model input size already use
`--imgsz`: `export_local_trt_engine.py`, `qualify_native_trt_lease.py`,
`qualify_native_trt_pipeline.py`, `qualify_radial_owner_pipeline.py`, and
`qualify_spherical_optimization.py`.

## Broader pre-existing naming differences

These are outside the three requested arguments and are left unchanged.

`--engine` supplies a TensorRT artifact where the main pipeline uses a tagged
`--model gpu:PATH`:

- `tools/capture_native_proto_outputs.py:94`
- `tools/qualify_native_trt_lease.py:55`
- `tools/qualify_native_trt_pipeline.py:86`

`--output-dir` is used instead of `--output` in 16 tools. Depending on the
tool this is a benchmark/evidence directory or export directory, not necessarily
the pipeline's final output directory. All occurrences:

- `tools/benchmark_owner_publication.py:40`
- `tools/benchmark_radial_columns.py:26`
- `tools/benchmark_radial_graph_dispatch.py:58`
- `tools/benchmark_radial_host_contention.py:129`
- `tools/benchmark_radial_owner.py:25`
- `tools/benchmark_radial_setup.py:44`
- `tools/build_source_release.py:23`
- `tools/capture_native_proto_outputs.py:95`
- `tools/export_local_trt_engine.py:15`
- `tools/qualify_bf16_proto_union.py:154`
- `tools/qualify_cropped_upload_pipeline.py:141`
- `tools/qualify_native_trt_lease.py:56`
- `tools/qualify_native_trt_pipeline.py:87`
- `tools/qualify_radial_owner_pipeline.py:22`
- `tools/qualify_spherical_large_address.py:28`
- `tools/qualify_spherical_optimization.py:196`

`--input-root` in several LTA harnesses takes a directory used to discover a
source, not the main TTA `--input` file path. LTA tile/window options likewise
control harness-specific fixtures rather than the TTA model's square raster.
They should not be blindly renamed to `--input` or `--imgsz` without reviewing
their meanings.

## Different quantities, not drop-in renames

`tools/hgx_selftest.py` and `tools/d1_ipc_selftest.py` use `--gpus` for a *count*
of devices, not a logical device index. In `hgx_selftest.py`, 0 means all visible
GPUs. It is not equivalent to `--device 0`.

`tools/hgx_selftest.py` uses independent `--height` and `--width` (defaults 129
and 131) for intentionally rectangular fixtures. `--imgsz` would lose that
independent-shape control. Patch sizes, slab depths, synthetic volume shapes,
byte sizes such as `--size-mib`, and other workload dimensions are also not model
image-size aliases.

The augmentation probe retains `--profiles`, `--compile` and `--report` as
probe-only controls. `--profiles` selects a set of shipped policies for math
checks, unlike main `--augmentation`, which selects a policy file for a run.
`--report` selects a JSON file, not a pipeline output directory. The completed-run
checker accepts positional `output_dir` to read existing artifacts; it does not
configure an inference output destination.

## Files changed in r2 relative to r1

Only `tools/tta_augmentation_smoke.py`, `tests/test_tta_augmentation_smoke_cli.py`,
`TTA_EXTERNAL_AUGMENTATION.md`, and this audit document change. No `XTA/` module,
example policy, older test tool, mask handling, scheduling, interpolation or
NRRD publication behavior changes.
