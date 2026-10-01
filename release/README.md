# Release inventory

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
