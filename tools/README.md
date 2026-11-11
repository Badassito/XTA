# Tools index

Run tools from the source checkout with the intended project environment and
inspect `--help` before execution. Generated output belongs in a task-specific
Scratch directory. Scripts under `share/xta/tools` in a wheel keep their sibling
helpers, but research and qualification tools can require the complete checkout,
test oracles, model bundles, retained evidence, or specific hardware.

## User utilities

| Tool | Purpose |
| --- | --- |
| `xta_native_filter.py` | Transfer Slicer cleanup evidence to native-resolution labels |
| `xta_auto.py` | Assist the native cleanup workflow |
| `run_transport.py` | Pack, verify, and recover stopped-run evidence |
| `prepare_pta_input.py` | Validate and publish a PTA input pair from authoritative edited shape or strict native recovery |
| `prebuild_yolo_label_cache.py` | Build `yolo train`'s label cache for a generated dataset on a process pool; `--self-test` checks byte identity with the installed Ultralytics |
| `slicer/xta_slicerrc.py` | Slicer toolbar and edit/export workflow |
| `slicer/xta_upscale_preview.py` | Native/LQ alignment and preview support |
| `xta_cut_axial_pair.py` | Crop a Slicer NPZ edit bundle and its grayscale MKV at a marker, or cut a single XTA NRRD and MKV |

See [Slicer cleanup](../docs/slicer_pta_cleanup.md) and
[run transport](../docs/run_transport.md) for workflows and required inputs.
Mixed-confidence native transfer is described in
[native union recovery](../docs/native_union_recovery.md).

For a marker placed on an XTA low-quality preview, check that the axial view
has **L** on its right edge. After **Export XTA Edits**, give the cutter the
directory containing `edits.json`, `source_manifest.json` and the numbered
NPZ files. From the tools directory:

```text
python xta_cut_axial_pair.py --edits /path/to/xta_edits --mkv M3_30_5_2_2026.mkv --ras -488 -280 60 --output-dir M3_30_cut
```

Bundle input defaults to a physical crop. The bundle supplies both grid sizes.
The new output directory contains the narrower MKV, the cropped low-quality
Global in `preview/<video-stem>.nrrd`, every named cropped layer in `edits/`
and the cut geometry in `cut.json`. The marker plane cuts every row and frame.
For a native-resolution training mask, use the updated `prepare_pta_input.py`
with `--mode edited_shape --bundle-dir M3_30_cut/edits --source-video
M3_30_cut/M3_30_5_2_2026.mkv` in the normal preparation command. The derived
bundle records both original grids so reconstruction samples the same positions
in the retained region. Strict native prediction recovery requires the original bundle.

For a single NRRD, supply `--nrrd FILE` with `--preview-manifest FILE` or
`--preview-scale 0.2`. Use `--mode crop` for a narrower pair or `--mode zero`
to clear the removed region. For a cropped preview union, `--nrrd-grid preview`
places its NRRD origin on the original preview grid. `--dry-run` prints the
cut columns before writing. The cutter uses NumPy, FFmpeg and ffprobe. When
copying the source, include `xta_cut_edits.py`, `xta_cut_nrrd.py`,
`xta_cut_video.py` and `prepare_pta_input.py` alongside it.

For manual edits that define the desired shape and separation, use
[edited-shape reconstruction](../docs/edited_shape_reconstruction.md). It does
not restore original native prediction components that disagree with the edit.

## Maintenance and release tools

| Tool | Purpose |
| --- | --- |
| `prepare_reconciliation_release.py` | Draft or replace the general current-source inventory |
| `verify_package_inventory.py` | Check maintained paths, versions, and canonical digests |
| `check_release_version.py` | Validate release-number transitions and local tag identities |
| `build_source_release.py` | Build an identified complete-source bundle |
| `qualify_release.py` | Run complete checkout qualification and preserve evidence |

The [maintenance guide](../docs/maintenance.md) explains these boundaries and
the difference between source identity, packaging, and numerical qualification.

## Focused diagnostics and qualification

- `qualify_*` scripts validate a named backend or contract. They can require
  test reference implementations, CUDA, native codecs, or local model assets.
- `benchmark_*` and `profile_*` scripts measure a specific path; retain their
  receipts and check that the requested backend actually ran. The maintained
  tilted projection benchmark is `benchmark_native_pull_projection.py`.
  `benchmark_tilted_azimuthal_projection.py` reports retirement of its old
  scatter backend rather than measuring current native destination pull.
- `audit_*`, `analyze_*`, `inspect_*`, and `report_*` tools examine saved runs or
  dataset evidence. A historical filename does not make its assumptions valid
  for every dataset; read its help and input contract.
- `lta_*`, `sam_*`, and `smoke_sam_*` cover propagation, crop strategies, or
  model initialization. Some preserve older algorithms as explicit reference
  experiments; they are not alternative production defaults.
- `verify_ptx_*` inspect compiled GPU code and must target the maintained
  backend. Their source/static checks differ from live GPU execution.

The PTA smoke-label and F3 inspection scripts now accept explicit paths and
keep file I/O behind `main()`. They require an external output directory for
generated reports and montages; importing them does not inspect a personal
dataset or write into the source tree.

Shared tool infrastructure lives in `gpu_reservation.py` for cooperative GPU
ownership and `pta_audit_common.py` for diagnostic output containment. New
scripts should reuse those boundaries rather than adding another lock or
hardcoded personal output path.
