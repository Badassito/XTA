"""Build the required native CPU kernels and TTA scheduler, plus optional Intel support.

The scheduler requires a C++17 compiler. Source builds on an Intel accelerator image
discover QATzip/QPL through pkg-config and the kernel DSA user ABI through
``linux/idxd.h``.  Explicit requests fail during the build instead of silently
producing a wheel without the requested backend.
"""

from __future__ import annotations

import os
import platform
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from setuptools import Extension, setup


SUPPORTED_MACHINE = platform.machine().strip().lower() in {"x86_64", "amd64"}
SUPPORTED_PLATFORM = sys.platform.startswith("linux") and SUPPORTED_MACHINE
BACKENDS = ("qat", "qpl", "dsa")


def _prepared_scheduler() -> Extension:
    if sys.platform == "win32":
        compile_args = ["/std:c++17", "/EHsc"]
        link_args = []
        libraries = ["ws2_32"]
    else:
        compile_args = ["-std=c++17", "-pthread"]
        link_args = ["-pthread"]
        libraries = []
    return Extension(
        "XTA._prepared_scheduler",
        sources=["native/prepared_scheduler.cpp"],
        depends=["native/prepared_scheduler.hpp", "native/prepared_scheduler_handles.hpp", "native/device_stage.hpp"],
        language="c++",
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )


def _cpu_kernels() -> Extension:
    recipe = runpy.run_path(str(Path(__file__).resolve().parent / "XTA" / "native_cpu_build.py"))
    source_sha256 = recipe["cpu_source_identity"]()
    return Extension(
        "_xta_cpu_kernels",
        sources=recipe["cpu_extension_sources"](),
        depends=recipe["cpu_extension_dependencies"](),
        language="c++",
        define_macros=[
            ("XTA_CPU_SOURCE_SHA256", f'"{source_sha256}"'),
            *((f"XTA_CPU_DIGEST_{domain}", f'"{digest}"')
              for domain, digest in recipe["cpu_domain_identities"]().items()),
        ],
        extra_compile_args=(
            ["/std:c++17", "/EHsc", "/fp:strict"] if sys.platform == "win32" else
            ["-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off", "-pthread"]
        ),
        extra_link_args=[] if sys.platform == "win32" else ["-pthread"],
    )


def _pkg_config(
    package: str,
    *,
    minimum_version: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    executable = shutil.which("pkg-config")
    if executable is None:
        return None
    exists = subprocess.run(
        [executable, "--exists", package],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode != 0:
        return None
    if minimum_version is not None:
        compatible = subprocess.run(
            [executable, f"--atleast-version={minimum_version}", package],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if compatible.returncode != 0:
            return None
    completed = subprocess.run(
        [executable, "--cflags", "--libs", package],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result: Dict[str, object] = {
        "include_dirs": [],
        "library_dirs": [],
        "libraries": [],
        "define_macros": [],
        "extra_compile_args": [],
        "extra_link_args": [],
    }
    for token in shlex.split(completed.stdout):
        if token.startswith("-I") and len(token) > 2:
            result["include_dirs"].append(token[2:])  # type: ignore[union-attr]
        elif token.startswith("-L") and len(token) > 2:
            result["library_dirs"].append(token[2:])  # type: ignore[union-attr]
        elif token.startswith("-l") and len(token) > 2:
            result["libraries"].append(token[2:])  # type: ignore[union-attr]
        elif token.startswith("-D") and len(token) > 2:
            definition = token[2:]
            key, separator, value = definition.partition("=")
            result["define_macros"].append(  # type: ignore[union-attr]
                (key, value if separator else None)
            )
        elif token in {"-pthread", "-pthreads"}:
            result["extra_compile_args"].append(token)  # type: ignore[union-attr]
            result["extra_link_args"].append(token)  # type: ignore[union-attr]
        elif token.startswith("-Wl,"):
            result["extra_link_args"].append(token)  # type: ignore[union-attr]
        else:
            result["extra_compile_args"].append(token)  # type: ignore[union-attr]
    return result


def _header_available(relative: str) -> bool:
    candidates = [Path("/usr/include"), Path("/usr/local/include")]
    for variable in ("CPATH", "C_INCLUDE_PATH"):
        candidates.extend(
            Path(entry)
            for entry in os.environ.get(variable, "").split(os.pathsep)
            if entry
        )
    return any((base / relative).is_file() for base in candidates)


def _requested_backends() -> tuple[set[str], bool]:
    raw = os.environ.get("XTA_BUILD_INTEL", "auto").strip().lower()
    explicit = raw != "auto"
    if raw in {"", "auto"}:
        selected: set[str] = set()
    elif raw in {"none", "0", "off", "false"}:
        selected = set()
    elif raw in {"all", "1", "on", "true"}:
        selected = set(BACKENDS)
    else:
        selected = {part.strip() for part in raw.split(",") if part.strip()}
        unknown = selected.difference(BACKENDS)
        if unknown:
            raise RuntimeError(
                "XTA_BUILD_INTEL accepts auto, none, all, or a comma-separated "
                f"subset of {BACKENDS}; unknown values: {sorted(unknown)}"
            )
    return selected, explicit


def _extension_from_pkg_config(
    *,
    backend: str,
    package: str,
    minimum_version: str,
    module: str,
    source: str,
    forced: bool,
) -> Optional[Extension]:
    config = _pkg_config(package, minimum_version=minimum_version)
    if config is None:
        if forced:
            raise RuntimeError(
                f"Intel {backend} extension was explicitly requested, but "
                f"pkg-config could not resolve {package!r} at version "
                f"{minimum_version} or newer"
            )
        return None
    compile_args = list(config["extra_compile_args"])  # type: ignore[arg-type]
    if "-std=c11" not in compile_args:
        compile_args.append("-std=c11")
    return Extension(
        module,
        sources=[source],
        include_dirs=list(config["include_dirs"]),  # type: ignore[arg-type]
        library_dirs=list(config["library_dirs"]),  # type: ignore[arg-type]
        libraries=list(config["libraries"]),  # type: ignore[arg-type]
        define_macros=list(config["define_macros"]),  # type: ignore[arg-type]
        extra_compile_args=compile_args,
        extra_link_args=list(config["extra_link_args"]),  # type: ignore[arg-type]
    )


def _intel_extensions() -> List[Extension]:
    selected, explicit_policy = _requested_backends()
    if not SUPPORTED_PLATFORM:
        if selected:
            raise RuntimeError(
                "Intel QAT/QPL/DSA extensions require Linux on x86-64; "
                f"this build is {sys.platform}/{platform.machine()}"
            )
        return []

    auto = os.environ.get("XTA_BUILD_INTEL", "auto").strip().lower() in {"", "auto"}
    qat_config = (
        _pkg_config("qatzip", minimum_version="1.3.2")
        if auto and "qat" not in selected else None
    )
    qpl_config = (
        _pkg_config("qpl", minimum_version="1.9.0")
        if auto and "qpl" not in selected else None
    )
    dsa_detected = _header_available("linux/idxd.h")
    if auto:
        if qat_config is not None:
            selected.add("qat")
        if qpl_config is not None:
            selected.add("qpl")
        if dsa_detected:
            selected.add("dsa")

    extensions: List[Extension] = []
    if "qat" in selected:
        extension = _extension_from_pkg_config(
            backend="QAT/QATzip",
            package="qatzip",
            minimum_version="1.3.2",
            module="XTA._qat_codec",
            source="native/qat_codec.c",
            forced=bool(explicit_policy),
        )
        if extension is not None:
            if "-pthread" not in extension.extra_compile_args:
                extension.extra_compile_args.append("-pthread")
            if "-pthread" not in extension.extra_link_args:
                extension.extra_link_args.append("-pthread")
            extensions.append(extension)
    if "qpl" in selected:
        extension = _extension_from_pkg_config(
            backend="IAA/QPL",
            package="qpl",
            minimum_version="1.9.0",
            module="XTA._qpl_codec",
            source="native/qpl_codec.c",
            forced=bool(explicit_policy),
        )
        if extension is not None:
            for argument in ("-pthread",):
                if argument not in extension.extra_compile_args:
                    extension.extra_compile_args.append(argument)
                if argument not in extension.extra_link_args:
                    extension.extra_link_args.append(argument)
            for library in ("dl", "stdc++"):
                if library not in extension.libraries:
                    extension.libraries.append(library)
            extensions.append(extension)
    if "dsa" in selected:
        if not dsa_detected:
            if explicit_policy:
                raise RuntimeError(
                    "Intel DSA extension was explicitly requested, but linux/idxd.h "
                    "was not found in the compiler include path"
                )
        else:
            extensions.append(
                Extension(
                    "XTA._dsa_copy",
                    sources=["native/dsa_copy.c"],
                    define_macros=[("_GNU_SOURCE", "1")],
                    extra_compile_args=["-O3", "-std=c11", "-pthread"],
                    extra_link_args=["-pthread"],
                )
            )
    return extensions


def _extensions() -> List[Extension]:
    selected = os.environ.get("XTA_BUILD_EXTENSION", "") if sys.argv[1:2] == ["build_ext"] else ""
    if selected == "cpu":
        return [_cpu_kernels()]
    if selected == "scheduler":
        return [_prepared_scheduler()]
    if selected:
        raise RuntimeError(f"Unknown XTA native build target: {selected!r}")
    return [_cpu_kernels(), _prepared_scheduler(), *_intel_extensions()]


setup(ext_modules=_extensions(), zip_safe=False)
