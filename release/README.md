# Release inventory

## v25.0.2 SAM bridge development candidate

The package and versioned launcher identify this uncommitted candidate as
`25.0.2`. Its separate `v25_0_2_sam_bridges_development_review` begins at the
exact tagged `v25.0.1` source (`d34b9ef7270f395efd87bc54e7b7803a467a9eda`),
including every prior development appendix. It preserves all historical
receipts and reviews changed source statements and inherited/new research tools.
The candidate has `kind=development`, `released=false`, and
`source_review_only=true`; this record makes no test or model qualification claim.

```text
python -B tools/prepare_reconciliation_release.py --development sam-bridges-v25.0.2 --output-dir ../Scratch/Experiments/SAM_v25_0_2_20261003/release/inventory
```

After source and validation tools are frozen, append the reviewed draft using
the same command with `--write`. Default preparation rejects overwriting an
existing appendix. Tagged receipts and all inherited history stay immutable.
If validation reveals a defect in this still-uncommitted source-only candidate,
preserve the withdrawn candidate and its pins in Scratch, verify that removing
only its key restores the exact tagged `v25.0.1` inventory, and restore only those
new candidate fields before regenerating the corrected audit. This recovery
must also verify unchanged `v25.0.1` HEAD and absence of a `v25.0.2` tag.
Execute and retain actual validation
with `tools/qualify_release.py --snapshot`, or `--snapshot --cpu-only` when only
the reduced CPU gate was run. These receipts state their coverage and exact source
identity. Complete source bundles of an uncommitted checkout use
`tools/build_source_release.py --snapshot`. Git commits and tags require a
separate request.

`_package_inventory.json` is a development audit record. It is included in the
source distribution and complete source bundle, but is not needed by the
installed `XTA` package or wheel.

Each released appendix is immutable. The `v24.0.4` tag authenticates the
legacy configuration cleanup and compiled CPU backend policy. The `24.0.5`
successor records the TTA/PTA throughput changes against that tagged receipt.
`tools/prepare_reconciliation_release.py` creates a draft under Scratch and
requires an explicit `--write` to append it after source and validation tools
are frozen. The new appendix independently pins predecessor module snapshots
and top-level retirements. `tools/verify_package_inventory.py` checks the
historical chain and current source, including protected Radial arithmetic.

The `25.0.0` successor adds single-backend SAM interpolation and bounded LTA
crop work. Its predecessor is the exact tagged `v24.0.5` source and inventory.
Preparation retains every historical record and independently pins changed
source-module predecessors, statement positions, explicitly reviewed
retirements, and inherited/new validation tools. It does not certify model
quality or claim an unexecuted workload passed.
The release also records the opt-in experimental SAM tracking crop selector;
whole-crop tracking stays the default and the public backend choices remain
`sdf` and `sam`.

```text
python -B tools/prepare_reconciliation_release.py --release 25.0.0 --output-dir ../Scratch/Releases/v25.0.0-review
```

Inspect the draft and freeze source and validation tools before using the same
command with `--write`. The operation appends a new receipt and updates only
its independently authenticated current-source pins. It cannot rewrite a
tagged release. Git commits and tags remain separate, explicitly requested
operations.

The release gate is `tools/qualify_release.py`. It runs the full test suite from
the repository root and checks the inventory before building a source bundle.

## Development after the v25.0.0 tag

`v25.0.0` at `edbb1d20df41a3be6450cd30a9d358f2f795d230` authenticates the released
inventory. Its appendix and independent release pins stay immutable. Subsequent
outer-crop work keeps the package and launcher version at `25.0.0` and receives
a separately labelled development audit, rather than a claimed `25.0.1` release.

```text
python -B tools/prepare_reconciliation_release.py --development outer-crop --output-dir ../Scratch/Experiments/SAM_Outer_Crop_20261001/development-review
```

The default writes `development_review_draft.json`, predecessor source pins, and
a summary under the requested Scratch directory. Inspect the draft and freeze
source and validation tools before repeating the command with `--write`.
`v25_outer_crop_development_review` then extends the exact tagged inventory with
source/statement/import-seam and tool predecessor links. Its `kind` is
`development`, `released` is false, and `package_version` remains `25.0.0`.
Only the development pins are updated; the tagged release records are retained.

Focused development tests can run while code evolves. Full inventory verification
requires a truthful review of the final frozen tree; unreviewed source changes
remain failures. Run `tools/qualify_release.py --snapshot` for a development
qualification and preserve its source identity and limitations. The audit does
not qualify a model, grant a new release, or authorize another commit or tag.

The subsequent guarded-rescue and broader TTA SAM view-routing work has its own
development audit. It does not replace the reviewed outer-crop appendix or the
immutable `v25.0.0` release:

```text
python -B tools/prepare_reconciliation_release.py --development guarded-rescue --predecessor-archive QUALIFIED_OUTER_CROP.zip --output-dir ../Scratch/Experiments/SAM_Guarded_Rescue_20261001/audit
```

Use the qualified outer-crop source archive as the explicit predecessor. The
draft pins that archive and its payload identities; inspect it before freezing
the guarded-rescue source, documentation, and checks, then repeat with `--write`.
The new `v25_guarded_rescue_development_review` is a distinct successor with
`kind=development`, `released=false`, and package/launcher version `25.0.0`.
The version string alone therefore does not identify whether a cluster checkout
contains this patch. Keep the development source identity and resolved SAM
policy identity with its run receipts.

The qualified predecessor is `XTA_v25.0.0_complete_source.zip`, with SHA-256
`6ca3e8158c42aaaba1b09fbf4e8c62495f2dbfb221fedf09e7c41d9b2e8bb4a6`.
It contains 670 source payload identities plus `RELEASE_MANIFEST`; its reviewed
outer-crop predecessor identity is
`62dd162866f8b55b8a5cbb30555797cab14c8669aaef49c8ffaa2c72de26b461`.
Pass the full archive wherever it was copied; the shipped verifier does not
depend on a local Scratch path.

This audit authenticates a source tree. It does not certify a complete cluster
workload, relax resource admission, or establish that memory-refused groups
were generated or rescued. The initial guarded rescue supports only single-edge
groups; stock multi-edge behavior remains unchanged. No new version, release,
commit, or tag is implied.

## v25.0.1 projection coverage patch

The patch keeps the released predecessor `v25.0.0` at
`edbb1d20df41a3be6450cd30a9d358f2f795d230` and every intervening development
appendix unchanged. Its source baseline is the reviewed development archive
with SHA-256
`626cf097a9c6467b9507059822380bb02586ce31fc9a5e7fc4c48072ae3a1415`, containing
699 source identities. This archive's original strict qualification gate failed
a trailing-newline source check after the CPU tests passed; the exact input was
restored and a transparent independent follow-up was audited. The release
review records `full_qualification=false` and
`qualification_status=strict_gate_failed_followup_audited` for that predecessor.

Prepare the release appendix from the pinned archive wherever it was copied:

```sh
python -B tools/prepare_reconciliation_release.py --release 25.0.1 --predecessor-archive REVIEWED_V25_SOURCE.zip --output-dir ../Scratch/Experiments/Projection_Coverage_v25_0_1_20261001/release/audit
```

Use the same arguments with `--write` only after implementation, tests and docs
freeze. The new `v25_0_1_release_review` preserves the development receipts and
links their source/tool identities to the patch. Source qualification and bundle
creation then run against the final published pins; the prior gate is never
retroactively promoted to success. See [projection coverage](../docs/projection_coverage.md)
for the patch's supported view and sampling contracts.
