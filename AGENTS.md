# Release numbering

Before changing an actual release version, creating a release commit/tag, or
publishing, check the proposed transition against the latest released version
on the intended base ancestry. Check all local release tags for collisions and
consult the History map below for existing aliases. When publishing, check
remote refs as well; report remote state as unverified if unavailable.

Run `python -B tools/check_release_version.py --target-version MAJOR.MINOR.PATCH`
before making a release commit/tag. Omit `--target-version` to check committed
HEAD. Qualification and source-bundle creation run this check automatically.
Only pass `--acknowledge-gap vFROM:vTO --reason "..."` when the user has already
accepted that exact deviation; never invent a reason to make the check pass.

- Normal increments from `M.m.p` are `M.m.(p+1)`, `M.(m+1).0`, or `(M+1).0.0`.
  A normal minor/major increment does not require filling unused patch numbers.
- Warn the user before a proposed operation introduces an unexplained gap,
  starts a series above `.0`/`.0.0`, or reuses a release label for different
  source. State the previous version, proposed version, and missing/conflicting
  identities; distinguish another-branch, local-only, and development versions.
  Pause that versioning operation until the user acknowledges the deviation,
  unless that exact deviation was already explicitly accepted. Continue any
  independent work. Normal increments require no extra confirmation.
- A development version, source ZIP, or uncommitted snapshot does not count as
  a committed/tagged release. Keep development and release provenance distinct.
- Do not renumber, retag, rewrite history, or invent/backfill a checkpoint to
  conceal a gap. Those actions require explicit user direction. Do not repeat
  warnings about already documented/accepted historical gaps unrelated to the
  proposed operation.
- Before reporting a release complete, verify that its source version, launcher,
  commit message and tag agree, and state whether the commit/tag is local or
  published. Commits and tags still occur only when requested.

# Historical release identities

Before interpreting History, Results, old qualification reports, or release
baselines, read the [version/commit map](../Scratch/Data/XTA/History/release/version_migrations.json)
and [interpretation rules](../Scratch/Data/XTA/History/release/README.md#historical-release-identities).
In a worktree, resolve these from the canonical History directory:
`C:\Users\Bry\Documents\ChatGPT\Scratch\Data\XTA\History`.

On 2026-10-05, the listed old v24 releases became v23 and old v25 releases became
v24. Old evidence retains its original labels, paths, bytes, and hashes. A bare
v24 label can belong to either numbering scheme; never resolve it by name alone.
Use the exact commit mapping and retained source manifests. Development/dirty
run labels do not establish source equivalence, even when their base commit maps.
In new analysis, state both the original label and its renamed release equivalent
when known. Do not apply this migration to future releases or unrelated versions.
