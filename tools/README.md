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
| `slicer/xta_slicerrc.py` | Slicer toolbar and edit/export workflow |
| `slicer/xta_upscale_preview.py` | Native/LQ alignment and preview support |

See [Slicer cleanup](../docs/slicer_pta_cleanup.md) and
[run transport](../docs/run_transport.md) for workflows and required inputs.
Mixed-confidence native transfer is described in
[native union recovery](../docs/native_union_recovery.md).

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
