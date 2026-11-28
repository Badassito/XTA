"""Module implementation identities use the release inventory's canonical source bytes.

A Windows working tree whose files were rewritten with CRLF must report the same
IMPLEMENTATION_SHA256 values as the LF release and cluster trees, so saved SAM
receipts keep validating. Run provenance keeps recording actual bytes.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path

import pytest

from XTA.file_hashing import canonical_source_bytes, sha256_file, source_sha256
from tools.verify_package_inventory import canonical_source_bytes as inventory_canonical_source_bytes

ROOT = Path(__file__).resolve().parents[1]
SELF_IDENTITIES = (
    ("sam_cyclic", "IMPLEMENTATION_SHA256"),
    ("sam_policy", "_POLICY_IMPLEMENTATION_SHA256"),
    ("sam_branch_selection", "IMPLEMENTATION_SHA256"),
    ("sam_filtering", "IMPLEMENTATION_SHA256"),
    ("sam_mask_reader", "IMPLEMENTATION_SHA256"),
    ("sam_gpu_rendering", "IMPLEMENTATION_SHA256"),
    ("spherical_rendering_cpu", "IMPLEMENTATION_SHA256"),
    ("radial_rendering_cpu", "IMPLEMENTATION_SHA256"),
    ("sam_canvas_rendering", "IMPLEMENTATION_SHA256"),
    ("sam_azimuthal_batch", "IMPLEMENTATION_SHA256"),
    ("sam_extrapolation_policy", "IMPLEMENTATION_SHA256"),
    ("sam_crop_retry", "IMPLEMENTATION_SHA256"),
    ("sam_resources", "IMPLEMENTATION_SHA256"),
    ("sam_integration", "_GRAY_CONSUMER_IMPLEMENTATION_SHA256"),
    ("lta_sam", "_SAM_INPUT_PREPARATION_SHA256"),
)
SAMPLES = (b"", b"a\nb\n", b"a\r\nb\r\n", b"a\r\nb\nc\rd", b"lone\rcarriage\r", b"nul\0\r\n",
           b"\xff\xfe\r\n", "unicode é\r\n".encode("utf-8"))


@pytest.mark.parametrize("data", SAMPLES)
def test_canonical_bytes_match_the_release_inventory(data):
    assert canonical_source_bytes(data) == inventory_canonical_source_bytes(data)


def test_crlf_and_lf_copies_share_one_source_identity(tmp_path):
    text = b'"""Fixture."""\nVALUE = 1\n\n\ndef helper():\n    return VALUE\n'
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(text)
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    assert source_sha256(lf) == source_sha256(crlf) == hashlib.sha256(text).hexdigest()
    assert sha256_file(lf) != sha256_file(crlf)


@pytest.mark.parametrize("module_name,attribute", SELF_IDENTITIES)
def test_module_identities_are_canonical_and_line_ending_stable(tmp_path, module_name, attribute):
    module = importlib.import_module(f"XTA.{module_name}")
    path = Path(module.__file__)
    identity = getattr(module, attribute)
    assert identity == source_sha256(path)
    canonical = canonical_source_bytes(path.read_bytes())
    copy = tmp_path / path.name
    copy.write_bytes(canonical.replace(b"\n", b"\r\n"))
    assert source_sha256(copy) == identity


def test_no_package_module_hashes_its_own_raw_bytes():
    offenders = []
    for path in sorted((ROOT / "XTA").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_bytes"
                    and "__file__" in ast.unparse(node.func.value)):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert offenders == []


def test_previous_release_identities_stay_readable():
    from XTA import sam_branch_selection, sam_cyclic

    assert sam_branch_selection._QUALIFIED_PACKED_OWNER_IMPLEMENTATIONS[
        "b7131c1e76e3a557354a8f7c3181233d669ad203751936d0a518174e94b27ce1"] == "tracked_known"
    assert "f65c83b547ec3b434eb0d25555685a022089984f9d7b76e5a79531d9c446f9bd" in sam_cyclic.QUALIFIED_IMPLEMENTATIONS
