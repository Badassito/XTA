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


def _legacy_spec(bundle):
    result = filtering.build_mask_filter(bundle)
    result["implementation_sha256"] = filtering._QUALIFIED_LEGACY_IMPLEMENTATION
    result["sha256"] = filtering._fingerprint({key: value for key, value in result.items() if key != "sha256"})
    return result


@pytest.mark.parametrize('implementation', [filtering._QUALIFIED_LEGACY_IMPLEMENTATION,
                                           filtering._QUALIFIED_PREVIOUS_IMPLEMENTATION,
                                           filtering._QUALIFIED_RADIUS_ONLY_IMPLEMENTATION,
                                           filtering._QUALIFIED_PRE_BOOLEAN_IMPLEMENTATION])
def test_qualified_legacy_filter_has_exact_current_pixels(implementation):
    bundle = _Bundle()
    current = filtering.effective_candidate_mask(bundle, "r", 0, filtering.build_mask_filter(bundle))
    spec = _legacy_spec(bundle)
    spec['implementation_sha256'] = implementation
    spec['sha256'] = filtering._fingerprint({key: value for key, value in spec.items() if key != 'sha256'})
    old = filtering.effective_candidate_mask(bundle, "r", 0, {"mask_filter": spec})
    np.testing.assert_array_equal(old, current)
    assert current[3:14, 3:14].all()
    assert not current[4:7, 23:26].any()


def test_old_compatibility_rejects_changed_numerical_function(monkeypatch):
    bundle = _Bundle()
    spec = _legacy_spec(bundle)

    def changed_filter(*args, **kwargs):
        return bundle.raw.copy(), {}

    monkeypatch.setattr(filtering, "filter_sam_components", changed_filter)
    with pytest.raises(ValueError, match="implementation differs"):
        filtering.effective_candidate_mask(bundle, "r", 0, spec)


def test_unknown_implementation_is_not_accepted_as_compatible():
    bundle = _Bundle()
    spec = _legacy_spec(bundle)
    spec["implementation_sha256"] = "0" * 64
    spec["sha256"] = filtering._fingerprint({key: value for key, value in spec.items() if key != "sha256"})
    with pytest.raises(ValueError, match="implementation differs"):
        filtering.effective_candidate_mask(bundle, "r", 0, spec)


def test_compatibility_rejects_changed_component_radius_helper(monkeypatch):
    bundle = _Bundle()
    spec = _legacy_spec(bundle)

    def changed_radii(raw, labels, count):
        return np.full(count, 1.e9)

    monkeypatch.setattr(filtering, '_component_inscribed_radii', changed_radii)
    with pytest.raises(ValueError, match='implementation differs'):
        filtering.effective_candidate_mask(bundle, 'r', 0, spec)


def test_real_legacy_unfiltered_receipt_keeps_its_meaning():
    bundle = _Bundle()
    receipt = {"schema": "xta.sam_selection/1", "resolved_policy": {"version": 1},
               "policy_name": "sam_conservative_v1"}
    np.testing.assert_array_equal(filtering.effective_candidate_mask(bundle, "r", 0, receipt), bundle.raw)
