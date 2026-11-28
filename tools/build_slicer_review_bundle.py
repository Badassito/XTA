"""Build a portable XTA native-review toolbar for Slicer's Python environment."""
import argparse
from importlib.util import find_spec
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build(destination):
    tools_dir = Path(__file__).resolve().parent
    package = tools_dir.parent / "XTA"
    if not package.is_dir():
        spec = find_spec("XTA")
        if spec is None or not spec.origin:
            raise RuntimeError("Cannot locate the installed XTA package")
        package = Path(spec.origin).parent
    modules = sorted(package.glob("native_review*.py"))
    if not modules:
        raise ValueError("This XTA checkout does not contain the native-window modules")
    # Keep the portable toolbar independent of the inference runtime.
    shared = ("__init__.py", "slicer_native_window.py", "slicer_preview_review.py", "slicer_reconciliation.py",
              "review_archive.py", "reconciliation.py", "reconciliation_components.py",
              "reconciliation_geometry.py", "reconciliation_io.py", "reconciliation_policy.py",
              "json_publication.py", "run_transport.py", "artifact_archive.py", "env_flags.py", "file_hashing.py")
    members = {f"XTA/{name}": package / name for name in shared}
    members.update({f"XTA/{p.name}": p for p in modules})
    examples = package / "examples"
    members["XTA/examples/__init__.py"] = examples / "__init__.py"
    members.update({f"XTA/examples/external_reconciliation/{p.name}": p
                    for p in sorted((examples / "external_reconciliation").glob("*.py"))})
    for name in ("xta_slicerrc.py", "xta_toolbar_startup.py"):
        members[name] = tools_dir / "slicer" / name
    # Read all sources before opening the destination. A missing package
    # dependency should never leave a half-built support bundle.
    sources = {name: path.read_bytes() for name, path in members.items()}
    sources["XTA/_slicer_review_bundle.py"] = b'"""Identifies the portable review-only package."""\n'
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        try:
            with ZipFile(output, "w", compression=ZIP_DEFLATED) as bundle:
                for name, source in sources.items():
                    bundle.writestr(name, source)
        except BaseException:
            output.close()
            destination.unlink(missing_ok=True)
            raise
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New .zip file to copy to the Slicer workstation")
    print(build(parser.parse_args().output).resolve())
