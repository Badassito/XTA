"""Dependency-free inspection shared by PTA and TTA external policies."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

@dataclass(frozen=True)
class AugmentationDefinition:
    """Import-free description of an external augmentation policy.

    Deferred mode deliberately parses only Python syntax and export names.  It
    therefore does not import Albumentations/Torch, allocate transform
    objects, or execute arbitrary policy-file code during dataset generation.
    """

    path: Path
    content_sha256: str
    export_name: str


def inspect_augmentation_definition(path_arg: str) -> AugmentationDefinition:
    path = Path(path_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"--augmentation file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"--augmentation must point to a Python file, not a directory: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"--augmentation must point to a .py file: {path}")
    content = path.read_bytes()
    content_sha256 = hashlib.sha256(content).hexdigest()
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Could not parse augmentation file {path}: {exc}") from exc

    supported = {
        "custom_transforms",
        "augmentation",
        "build_augmentation",
        "build_gpu_augmentation",
    }
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(str(node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    defined.add(str(target.id))
    recognized = sorted(supported & defined)
    if not recognized:
        raise ValueError(
            f"Augmentation file {path} must define exactly one of: "
            "custom_transforms, augmentation, build_augmentation, build_gpu_augmentation"
        )
    if len(recognized) != 1:
        raise ValueError(
            f"Augmentation file {path} defines multiple supported exports {recognized}; define exactly one"
        )
    return AugmentationDefinition(path, content_sha256, recognized[0])


def assert_augmentation_definition_unchanged(
    definition: Optional[AugmentationDefinition],
) -> None:
    """Reject successful publication if the external policy changed mid-run."""

    if definition is None:
        return
    try:
        current = inspect_augmentation_definition(str(definition.path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Augmentation policy changed during execution; refusing a complete "
            f"manifest: path={definition.path}, unavailable_or_invalid={exc}"
        ) from exc
    changed: List[str] = []
    if current.path != definition.path:
        changed.append("path")
    if current.content_sha256 != definition.content_sha256:
        changed.append("sha256")
    if current.export_name != definition.export_name:
        changed.append("export")
    if changed:
        raise RuntimeError(
            "Augmentation policy changed during execution; refusing a complete "
            f"manifest: path={definition.path}, fields={changed}"
        )


def resolve_augmentation_definitions(
    values: Sequence[str] | str | None,
) -> dict[str, AugmentationDefinition]:
    """Inspect backend-tagged policy files without importing their runtimes."""
    if values is None:
        return {}
    tokens = [values] if isinstance(values, str) else list(values)
    resolved: dict[str, AugmentationDefinition] = {}
    for value in tokens:
        raw = str(value).strip()
        backend, separator, payload = raw.partition(':')
        backend = backend.lower()
        if backend in {'cpu', 'gpu'} and separator:
            if not payload.strip():
                raise ValueError(f'--augmentation {backend}: requires a Python policy path')
            definition = inspect_augmentation_definition(payload.strip())
            actual = 'gpu' if definition.export_name == 'build_gpu_augmentation' else 'cpu'
            if actual != backend:
                raise ValueError(f'--augmentation {backend}: requires a {backend.upper()} policy export')
        elif len(tokens) == 1:
            definition = inspect_augmentation_definition(raw)
            backend = 'gpu' if definition.export_name == 'build_gpu_augmentation' else 'cpu'
        else:
            raise ValueError('--augmentation requires cpu:POLICY.py and/or gpu:POLICY.py entries')
        if backend in resolved:
            raise ValueError(f'--augmentation contains duplicate {backend}: entries')
        resolved[backend] = definition
    return resolved
