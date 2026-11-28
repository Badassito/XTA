"""Quality-version validation cannot silently turn off component filtering."""
import numpy as np
import pytest

from XTA import sam_filtering as filtering


class _Bundle:
    groups = {"g": {"interpolation_min_radius": 2.0}}
    runs = {"r": {"group_id": "g"}}

    def __init__(self):
        self.raw = np.zeros((20, 30), bool)
        self.raw[3:14, 3:14] = True
        self.raw[4:7, 23:26] = True

    def raw_mask(self, run_id, frame):
        return self.raw

    def candidate_mask(self, run_id, frame):
        return self.raw


@pytest.mark.parametrize("version", [2, 3, 4, 5])
@pytest.mark.parametrize("declaration", ["version", "name", "filter_identity", "filter_setting"])
def test_modern_receipt_missing_filter_cannot_use_legacy_unfiltered_path(version, declaration):
    receipt = {"schema": "xta.sam_selection/1"}
    if declaration == "version":
        receipt["resolved_policy"] = {"version": version}
    elif declaration == "name":
        receipt["policy_name"] = f"sam_conservative_v{version}"
    elif declaration == "filter_identity":
        receipt["component_filter_implementation_sha256"] = filtering.IMPLEMENTATION_SHA256
    else:
        receipt["resolved_policy"] = {"enforce_interpolation_min_radius": False}
    with pytest.raises(ValueError, match="missing its component filter spec"):
        filtering.effective_candidate_mask(_Bundle(), "r", 0, receipt)


@pytest.mark.parametrize("implementation,issued_native_source", [
    # The pre-AOT source guard, plus the retained pre-checkpoint and tagged AOT sources.
    ("6dd89b0929e8d3407750654cdf9606d2dfd3fe1689c3549e0debd04a95152bdc", None),
    ("434ec05db044c2c1383e4e58b5f099e76945018b2e60b87cf2fe8b6d03ad5cf3",
     "23c8ccdcb70eb666a65c32476c15119a6c7853a1d0d7df13157ef387d8dd6109"),
    ("434ec05db044c2c1383e4e58b5f099e76945018b2e60b87cf2fe8b6d03ad5cf3",
     "31c3e5ead5e932db22c4704875dc10935ff768a195a986cc70eaa591294d0fa7"),
    ("0" * 64, None),
])
@pytest.mark.parametrize("enabled", (False, True))
def test_noncurrent_filter_spec_requires_regeneration(monkeypatch, implementation,
                                                       issued_native_source, enabled):
    bundle = _Bundle()
    spec = filtering.build_mask_filter(bundle, enabled=enabled)
    assert implementation != filtering.IMPLEMENTATION_SHA256
    spec["implementation_sha256"] = implementation
    if enabled:
        if issued_native_source is None:
            spec.pop("native_implementation")
        else:
            spec["native_implementation"]["source_sha256"] = issued_native_source
    _refingerprint(spec)
    monkeypatch.setattr(filtering, "native_cpu_provenance",
                        lambda _: pytest.fail("An incompatible module must fail before native verification"))
    with pytest.raises(ValueError, match="implementation differs.*regenerate"):
        filtering.effective_candidate_mask(bundle, "r", 0, {"mask_filter": spec})


def test_current_disabled_filter_preserves_raw_masks_without_native_receipt(monkeypatch):
    bundle = _Bundle()
    monkeypatch.setattr(filtering, "native_cpu_provenance",
                        lambda _: pytest.fail("A disabled filter must not load the native extension"))
    spec = filtering.build_mask_filter(bundle, enabled=False)
    assert "native_implementation" not in spec
    np.testing.assert_array_equal(filtering.effective_candidate_mask(bundle, "r", 0, spec), bundle.raw)


def _refingerprint(spec):
    spec["sha256"] = filtering._fingerprint({key: value for key, value in spec.items() if key != "sha256"})


def test_real_legacy_unfiltered_receipt_keeps_its_meaning():
    bundle = _Bundle()
    receipt = {"schema": "xta.sam_selection/1", "resolved_policy": {"version": 1},
               "policy_name": "sam_conservative_v1"}
    np.testing.assert_array_equal(filtering.effective_candidate_mask(bundle, "r", 0, receipt), bundle.raw)


def test_current_filter_receipt_records_native_source_and_issuing_binary():
    bundle = _Bundle()
    spec = filtering.build_mask_filter(bundle)
    native = spec["native_implementation"]
    assert native == filtering.native_cpu_provenance("retina")
    assert all(len(native[key]) == 64 for key in ("source_sha256", "binary_sha256"))
    assert filtering._spec(spec) is spec

    # Source and FP contract identify replay semantics. The issuing binary and
    # compiler stay in the receipt even when another platform performs replay.
    portable = dict(spec, native_implementation={**native, "binary_sha256": "a" * 64, "compiler": "other-build"})
    _refingerprint(portable)
    assert filtering._spec(portable) is portable

    for field, replacement in (("source_sha256", "0" * 64), ("fp_contract", "changed"),
                               ("binary_sha256", None), ("binary_sha256", "g" * 64), ("compiler", "")):
        changed = dict(spec, native_implementation={**native, field: replacement})
        _refingerprint(changed)
        with pytest.raises(ValueError, match="native implementation differs.*regenerate"):
            filtering._spec(changed)
    missing = dict(spec)
    del missing["native_implementation"]
    _refingerprint(missing)
    with pytest.raises(ValueError, match="native implementation differs.*regenerate"):
        filtering._spec(missing)
