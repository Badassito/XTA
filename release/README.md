# Release inventory

`_package_inventory.json` authenticates reviewed source identity and its
predecessor chain. It is included in the source distribution and complete source
bundle, but is not needed by the installed `XTA` package or wheel. Keep this
machine-readable inventory in the repository; run-specific stories,
measurements and qualification records belong in the workspace History.

## Source review lifecycle

`tools/prepare_reconciliation_release.py` writes a draft under the requested
output directory. Choose a supported development review or release with
`--help`, then replace the capitalized placeholders below with actual values:

```text
python -B tools/prepare_reconciliation_release.py --help
python -B tools/prepare_reconciliation_release.py --development DEVELOPMENT_REVIEW --predecessor-archive QUALIFIED_PREDECESSOR.zip --output-dir ../Scratch/Releases/REVIEW_NAME/inventory
python -B tools/prepare_reconciliation_release.py --release VERSION --predecessor-archive QUALIFIED_PREDECESSOR.zip --output-dir ../Scratch/Releases/REVIEW_NAME/inventory
```

Some reviews derive their predecessor from an exact Git tag; archive-based
reviews require the explicit predecessor archive. The archive can be supplied
from any location. Preparation pins its identity rather than depending on a
machine-local artifact path. Distinguish a qualified development archive from a
released Git predecessor; neither a version string nor a source-only audit
changes that status.

Inspect the draft and freeze source, documentation, tests and validation tools
before repeating the supported command with `--write`. The operation appends a
new review and updates only its independently authenticated current-source pins.
Tagged release appendices and inherited history are immutable. Source/module
predecessors, statement positions, reviewed retirements and validation-tool
identities must remain attributable. `tools/verify_package_inventory.py` checks
the historical chain and current source, including protected numerical contracts.

A development review records `kind=development`, `released=false` and its source
review scope. It authenticates a source tree; it does not certify model quality,
a complete cluster workload, execution of memory-refused groups or an unexecuted
check. Git commits and tags are separate operations requiring an explicit request.

If validation invalidates an uncommitted candidate, preserve its source bundle,
receipts, appendix and current-source pins before withdrawing it. Remove only
that candidate's inventory fields, verify the immutable predecessor inventory
and Git state, and regenerate the review after the correction is frozen. Do not
rewrite a failed or reduced-scope gate as a historical success.

## Qualification and source bundles

The release gate runs the test suite from the repository root, verifies the
inventory and constructs a complete source bundle:

```text
python -B tools/qualify_release.py --output-dir ../Scratch/Releases/REVIEW_NAME/qualification
python -B tools/qualify_release.py --snapshot --output-dir ../Scratch/Releases/REVIEW_NAME/development-qualification
python -B tools/build_source_release.py --snapshot --output-dir ../Scratch/Releases/REVIEW_NAME/source
```

The default gate requires a clean Git checkout and bundles committed Git bytes.
`--snapshot` validates and labels an uncommitted development tree. Each step must
complete without changing the source tree. `--snapshot --cpu-only` records an
explicitly reduced CPU gate; it does not establish full GPU or release
qualification. Qualification receipts retain exact source identity, executed and
skipped coverage, logs and limitations. Full source bundles preserve the
maintained tools, tests, native sources, documentation and packaging configuration.

Store generated reviews, logs and bundles in a task-specific Scratch directory.
Preserve release history and completed run evidence in
`Scratch/Data/XTA/History`; historical artifact paths and job identifiers are not
part of this reusable release procedure.
