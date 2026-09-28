# Release inventory

`_package_inventory.json` is a development audit record. It is included in the
source distribution and complete source bundle, but is not needed by the
installed `XTA` package or wheel.

Each released appendix is immutable. The `v24.0.1` tag authenticates its own
review; fixes made after that tag are recorded in the `24.0.2` successor rather
than rewriting the released receipt. `tools/prepare_reconciliation_release.py`
creates a draft under Scratch and requires an explicit `--write` to append it.
`tools/verify_package_inventory.py` checks both the historical chain and the
current source, including protected Radial arithmetic.

The release gate is `tools/qualify_release.py`. It runs the full test suite from
the repository root and checks the inventory before building a source bundle.
