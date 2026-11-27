"""Prepare an AOT CPU extension for a source checkout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile


MODULE = "_xta_cpu_kernels"
ABI_VERSION = 1
BINARY_ENV = "XTA_NATIVE_CPU_BINARY"
ARTIFACTS_ENV = "XTA_NATIVE_CPU_ARTIFACTS"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT = "native_cpu_build.json"
_REGISTRY = "native/cpu_domains.inc"
_BUILD_RECIPES = ["setup.py", "pyproject.toml", "XTA/native_cpu_build.py"]
_DOMAIN = re.compile(r"XTA_CPU_DOMAIN\(([a-z][a-z0-9_]*)\)")
_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


class NativeCpuBuildError(RuntimeError):
    """The checkout does not have a usable ahead-of-time CPU extension."""


def is_source_checkout() -> bool:
    return (SOURCE_ROOT / "setup.py").is_file() and (SOURCE_ROOT / _REGISTRY).is_file()


def _domains() -> list[str]:
    domains = []
    for entry in (SOURCE_ROOT / _REGISTRY).read_text(encoding="ascii").splitlines():
        entry = entry.strip()
        if not entry or entry.startswith("//"):
            continue
        match = _DOMAIN.fullmatch(entry)
        if match is None:
            raise NativeCpuBuildError(f"Invalid CPU kernel registry entry: {entry!r}")
        domains.append(match[1])
    if len(domains) != len(set(domains)):
        raise NativeCpuBuildError("Duplicate native CPU kernel domain")
    return domains


def cpu_extension_sources() -> list[str]:
    return ["native/cpu_kernels.cpp", *(f"native/cpu_{domain}.cpp" for domain in _domains())]


def _local_sources(relative_paths: list[str]) -> dict[str, bytes]:
    pending = list(relative_paths)
    contents = {}
    while pending:
        relative = pending.pop()
        if relative in contents:
            continue
        try:
            source = (SOURCE_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        except OSError as error:
            raise NativeCpuBuildError(f"Native CPU build input unavailable: {relative}") from error
        contents[relative] = source
        if not relative.endswith((".cpp", ".hpp", ".h")):
            continue
        for local in _LOCAL_INCLUDE.findall(source.decode("utf-8")):
            included = ((SOURCE_ROOT / relative).parent / local).resolve()
            if not included.is_relative_to(SOURCE_ROOT / "native"):
                raise NativeCpuBuildError(f"Native CPU local include escapes source tree: {local}")
            pending.append(included.relative_to(SOURCE_ROOT).as_posix())
    return contents


def cpu_extension_dependencies() -> list[str]:
    return [
        *_BUILD_RECIPES,
        *sorted(relative for relative in _local_sources(cpu_extension_sources())
                if not relative.endswith(".cpp")),
    ]


def _fingerprint(contents: dict[str, bytes]) -> str:
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode("ascii")).hexdigest()


def cpu_source_identity() -> str:
    contents = _local_sources([*_BUILD_RECIPES, *cpu_extension_sources()])
    return _fingerprint(contents)


def cpu_domain_identities() -> dict[str, str]:
    identities = {}
    for domain in _domains():
        contents = _local_sources(["native/cpu_kernels.cpp", f"native/cpu_{domain}.cpp"])
        # The registry lists membership only. Adding another numerical domain
        # does not change the implementation or input contract of this one.
        contents.pop(_REGISTRY, None)
        identities[domain] = _fingerprint(contents)
    return identities


def binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(source_sha256: str | None = None) -> dict:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if sys.implementation.name != "cpython" or not suffix:
        raise NativeCpuBuildError("The native CPU kernels require a CPython extension ABI")
    return {
        "schema": "xta.native_cpu_build/1",
        "filename": MODULE + suffix,
        "cpu_kernel_abi": ABI_VERSION,
        "python_abi": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "source_sha256": source_sha256 or cpu_source_identity(),
    }


def _artifacts_root(artifacts_dir: str | Path | None = None) -> Path:
    configured = artifacts_dir if artifacts_dir is not None else os.environ.get(ARTIFACTS_ENV)
    if configured:
        root = Path(configured)
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "XTA" / "native-cpu"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "xta" / "native-cpu"
    root = root.expanduser().resolve()
    if root.is_relative_to(SOURCE_ROOT):
        raise NativeCpuBuildError(f"{ARTIFACTS_ENV} must point outside the XTA checkout: {root}")
    return root


def _destination(root: Path, identity: dict) -> Path:
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("ascii")).hexdigest()
    return root / key


@dataclass(frozen=True)
class CpuArtifact:
    path: Path
    sha256: str


def _validated_binary(path: Path, identity: dict) -> CpuArtifact:
    path = path.resolve()
    try:
        if path.is_relative_to(SOURCE_ROOT) or path.name != identity["filename"]:
            raise ValueError("unexpected extension path or CPython ABI filename")
        receipt = json.loads((path.parent / _RECEIPT).read_text(encoding="utf-8"))
        digest = binary_sha256(path)
        if receipt != {**identity, "binary_sha256": digest}:
            raise ValueError("source, ABI or extension digest changed")
    except (OSError, ValueError) as error:
        raise NativeCpuBuildError(f"Invalid native CPU artifact {path}: {error}") from error
    return CpuArtifact(path, digest)


def find_native_cpu_binary(*, source_sha256: str | None = None) -> CpuArtifact | None:
    """Find existing build output without launching any commands."""
    identity = _identity(source_sha256)
    explicit = os.environ.get(BINARY_ENV)
    if explicit:
        return _validated_binary(Path(explicit).expanduser(), identity)
    destination = _destination(_artifacts_root(), identity)
    if destination.exists():
        return _validated_binary(destination / identity["filename"], identity)
    return None


def _compile(directory: Path, identity: dict) -> Path:
    build_lib = directory / "out"
    temporary = directory / "temp"
    temporary.mkdir()
    environment = dict(
        os.environ, XTA_BUILD_EXTENSION="cpu", TMPDIR=str(temporary),
        TMP=str(temporary), TEMP=str(temporary),
    )
    command = [
        sys.executable, "-X", "utf8", "-B", "setup.py", "build_ext",
        "--build-lib", str(build_lib), "--build-temp", str(temporary / "objects"),
    ]
    result = subprocess.run(
        command, cwd=SOURCE_ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    binary = build_lib / identity["filename"]
    if result.returncode or not binary.is_file():
        raise NativeCpuBuildError(
            "Could not build the native CPU extension with setuptools, CPython headers, and a C++17 compiler.\n"
            + result.stdout
        )
    check = (
        "import sys; sys.path.insert(0, sys.argv[1]); import _xta_cpu_kernels as m; "
        "assert m.ABI_VERSION == int(sys.argv[2]); assert m.SOURCE_SHA256 == sys.argv[3]"
    )
    loaded = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-B", "-c", check, str(build_lib),
         str(ABI_VERSION), identity["source_sha256"]],
        cwd=directory, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    if loaded.returncode:
        raise NativeCpuBuildError(f"Native CPU extension failed its import check:\n{loaded.stdout}")
    return binary


def build_native_cpu(*, artifacts_dir: str | Path | None = None) -> Path:
    """Build or reuse a verified artifact in an outside-checkout directory."""
    identity = _identity()
    root = _artifacts_root(artifacts_dir)
    destination = _destination(root, identity)
    binary = destination / identity["filename"]
    if destination.exists():
        return _validated_binary(binary, identity).path
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as stage:
            built = _compile(Path(stage), identity)
            if _identity() != identity:
                raise NativeCpuBuildError("Native CPU sources changed during the build. Retry the build.")
            publish = built.parent
            (publish / _RECEIPT).write_text(
                json.dumps({**identity, "binary_sha256": binary_sha256(built)}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                publish.rename(destination)
            except OSError:
                # Another explicit build may have published the same identity.
                if not destination.is_dir():
                    raise
    except OSError as error:
        raise NativeCpuBuildError(f"Native CPU extension build failed at {root}: {error}") from error
    return _validated_binary(binary, identity).path
