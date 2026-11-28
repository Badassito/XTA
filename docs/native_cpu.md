# Native CPU kernels

XTA builds its bulk CPU kernels ahead of time as a C++17 extension for CPython.
Platform wheels include the extension, so launching an installed wheel needs no
compiler. The CPU extension is independent of the native detector scheduler and
CUDA. Numerical operations stop with a diagnostic if a compatible CPU extension
cannot be loaded. CLI help and version discovery do not need it.

## Running from source

Use the same Python interpreter that will run XTA. Install setuptools, the
CPython development headers, and a C++17 compiler. On Windows, use MSVC Build
Tools and the Windows SDK. The CPU extension uses the Python buffer protocol and
does not need NumPy headers. The Unix CPU extension requests `-O3` and keeps the
strict floating-point flags. The build recipe adds no target ISA flags.

```text
python -B tools/build_native_cpu.py
```

The builder uses the platform's user cache by default. On Windows this is
`%LOCALAPPDATA%\XTA\native-cpu`. On other platforms it is
`$XDG_CACHE_HOME/xta/native-cpu`, or `~/.cache/xta/native-cpu` with no override.
To use an external location such as the task's Scratch directory, pass
`--artifacts-dir OUTSIDE_CHECKOUT` and set `XTA_NATIVE_CPU_ARTIFACTS` to that same
directory in the launch environment.

The tool prints the extension path. You can also select that path with
`XTA_NATIVE_CPU_BINARY` in a matching source checkout. Keep the adjacent
`native_cpu_build.json` receipt together with the extension. The source loader
checks the native build source fingerprint, CPython ABI, and binary digest.
Rebuild with the selected interpreter if any native build input changes.
Both artifact identities include `setup.py`, so a change to that shared recipe
requires new CPU and scheduler artifacts for source launches.
Launches load an existing matching artifact and never start a CPU compiler.

## Building a wheel

```text
python -m pip wheel . --no-deps --wheel-dir PATH_OUTSIDE_CHECKOUT
```

A standard platform wheel includes the CPU extension and the native TTA
scheduler. Release qualification explicitly builds the CPU extension in its
own evidence directory before running repository tests. See the
[maintenance guide](maintenance.md) for packaging and qualification checks.
