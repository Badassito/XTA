# Maintaining the source tree

Keep source, tests, native extensions, documentation, and reusable development
tools in the repository. Write generated logs, reports, archives, datasets, and
validation output to a task-specific directory under the sibling `Scratch`
directory. Preserve `.git` and `__pycache__`. If a build creates `build/`, `dist/`,
or `*.egg-info/` in the checkout, preserve needed artifacts in Scratch and remove
those generated directories after validation.

Name maintained modules, tests and guides by responsibility rather than release
number. The versioned launcher is the exception. Keep historical numerical
identity in fixture metadata and preserve the recorded data when renaming a
fixture; immutable release archives and historical evidence retain their names.

## Three separate inventories

| File | Purpose | Maintenance |
| --- | --- | --- |
| `release/_package_inventory.json` | Canonical digests of every maintained source file | Regenerate from the stable reviewed tree; do not edit digests manually |
| `MANIFEST.in` | Extra source-distribution members, including test helpers and fixtures | Maintain directory patterns when adding a new resource category |
| `pyproject.toml` data files | Documentation and companion scripts installed with a wheel | Preserve installed paths; directory patterns include new sibling scripts |

The flat source inventory includes tracked and nonignored source files, tests,
tools, and repository rules. It is an identity record, not a claim that every
workload passed qualification. UTF-8 text digests normalize CRLF to LF; binary
files remain exact. The inventory does not hash itself, while Git and a complete
source bundle cover its bytes. Verification also checks that files did not
change during the check.

LTA run telemetry separately records package Python sources under the explicit
scope `xta.package_python_sources/1`. That fingerprint includes shared SAM and
geometry modules and future Python modules automatically. It does not identify
native binaries, dependencies, or model weights. Historical SAM receipt readers
also authenticate selected numerical helper implementations; those compatibility
identities are separate from both packaging and the current-source inventory.

Standard wheels and source distributions are packaging products. They have
generated metadata and differ from the complete source tree; do not present
them as a verified complete-source bundle. Use `build_source_release.py` when
the complete maintained source and its manifest are required. An installed
qualification script may still require a source checkout, tests, models, or
specific hardware; installing its `.py` file does not supply those inputs.

## Routine change workflow

1. Inspect Git status and the owning module before changing code. Treat historical
   version names as clues, not proof that a path is unused.
2. Add or update tests for changed behavior or a meaningful boundary. Run the
   relevant regression group in the intended dependency environment, keeping
   temporary files, caches, logs, and JUnit reports in the task's Scratch folder.
3. Review the diff, test results, and newly added files. `git diff --check` does
   not inspect untracked files; the staged check before a requested commit does.
4. Stop source edits, then refresh and verify the inventory once. Concurrent
   changes intentionally make inventory generation or verification fail.
5. Check the launcher, output containment, and Git status after cleanup. A
   successful test command does not authorize a commit or release.

From the repository root, replacing `CHANGE` with the task's directory name:

```text
python -B tools/prepare_reconciliation_release.py --output-dir ../Scratch/Experiments/CHANGE/inventory
python -B tools/prepare_reconciliation_release.py --output-dir ../Scratch/Experiments/CHANGE/inventory --write
python -B tools/verify_package_inventory.py
python -B -m XTA --version
```

Despite its historical name, `prepare_reconciliation_release.py` is the general
inventory writer. Without `--write` it produces a reviewable draft in Scratch;
with `--write` it atomically replaces the current flat inventory. It does not
change versions, commit, tag, publish, or rewrite History.

An uncommitted cleanup can retain the current source version while in
development. Its changed bytes are not equivalent to the tagged release. Use
`--snapshot` when explicitly creating a development bundle or running the
qualification tool on that state; do not silently reuse a release identity.

## Packaging checks

The packaging integration tests build an sdist, build a wheel from that sdist,
and install it into an isolated target outside the checkout. They check that
test helpers and fixtures survive source packaging, companion scripts accompany
installed user tools, and `tests` is not installed as a runtime Python package.
Keep this artifact check when changing packaging patterns: a list of filenames
in metadata does not prove the built package contains a working tool.

## GPU tools and qualification

Developer tools use the shared [GPU reservation helper](../tools/gpu_reservation.py).
Acquisition atomically creates `Scratch/Temp/GPU_LOCK` with task, PID, start time,
and an ownership token. An occupied reservation is waited on up to the tool's
timeout; cleanup must not delete another owner's reservation. Keep exceptional
retention behavior explicit when a worker may still own CUDA resources.

For CPU-only checks, set `CUDA_VISIBLE_DEVICES=-1` before starting Python.
An empty PowerShell environment assignment can remove the variable and leave
the GPU visible. Tests that automatically use an available CUDA device still
require a reservation when run with devices visible.

Heatsoak the CPU/GPU before timed local benchmarks as appropriate. Local tests
establish correctness and accelerated-path availability on this machine; they
do not establish cluster throughput, multi-GPU safety, or model behavior on
representative volumes.

`qualify_release.py` is the complete qualification gate. Its clean-release path
requires CUDA and the required numerical cases. `--snapshot --cpu-only` records
reduced development coverage and is not release qualification.

## Release boundaries

Follow [AGENTS.md](../AGENTS.md) before changing a release version or creating a
requested commit/tag. In particular:

```text
python -B tools/check_release_version.py --target-version MAJOR.MINOR.PATCH
```

Check the intended ancestry, local tag collisions, and historical identity map.
Check remote refs when publishing. Do not fill gaps by inventing releases or
rewrite tags to hide them. A requested release commit, tag, source version,
and launcher must agree; verify the committed result with the checker without
`--target-version`. Commits and tags occur only on request.

Authenticated historical SAM receipt readers and their source hashes are
compatibility contracts. Refactor them only while preserving accepted historical
fixtures and rejection of malformed or incompatible evidence. Likewise, keep
copyable augmentation policy examples self-contained unless deliberately changing
that interface.
