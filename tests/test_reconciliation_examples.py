"""Behavior and portable loading of the curated external reconciliation presets."""
from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np

from XTA.reconciliation_policy import resolve_reconciliation, load_reconciliation_policy
from XTA.examples.external_reconciliation._confidence_core import _decide_slice
from XTA.examples.external_reconciliation._hybrid import decide_slice as hybrid_slice


EXAMPLES = Path(__file__).resolve().parents[1] / 'XTA/examples/external_reconciliation'
EXPECTED = {'union', 'confidence_core_rescue', 'confidence_anchored', 'quorum3',
            'largest_island', 'hybrid_with_fill'}


def load(path):
    return load_reconciliation_policy(resolve_reconciliation(
        SimpleNamespace(reconciliation=path, reconciliation_memory_mib=64)))


def block(shape=(1, 17, 21)):
    return dict(candidate=np.zeros(shape, bool), score=np.zeros(shape, np.float64),
                support=np.zeros(shape, np.uint16), prediction_support=np.zeros(shape, np.uint16),
                anchored=np.zeros(shape, bool), z0=0, z1=shape[0])


def test_curated_preset_roster_and_public_identity():
    paths = {path.stem: path for path in EXAMPLES.glob('*.py') if not path.name.startswith('_')}
    assert set(paths) == EXPECTED
    for name, path in paths.items():
        assert load(path)['name'] == name


def test_snapshotted_custom_presets_load_outside_the_example_directory(tmp_path):
    sample = block()
    sample['candidate'][:, 3:14, 3:18] = True
    sample['score'][sample['candidate']] = 3.
    sample['support'][sample['candidate']] = 3
    sample['prediction_support'][sample['candidate']] = 3
    sample['anchored'][sample['candidate']] = True
    for name in ('confidence_core_rescue', 'hybrid_with_fill'):
        source = EXAMPLES / (name + '.py')
        snapshot = tmp_path / name / 'policy.py'
        snapshot.parent.mkdir()
        shutil.copyfile(source, snapshot)
        np.testing.assert_array_equal(load(snapshot)['decide'](sample), load(source)['decide'](sample))


def test_core_keeps_coherent_anchors_and_bounds_rescue_through_observed_support():
    sample = block()
    candidate = sample['candidate'][0]
    candidate[5:10, 3:8] = True
    candidate[7, 8:17] = True
    sample['score'][0, 5:10, 3:8] = 1.2
    sample['support'][0, 5:10, 3:8] = 2
    sample['prediction_support'][0, 5:10, 3:8] = 2
    sample['anchored'][0, 5:10, 3:8] = True
    sample['score'][0, 7, 8:17] = .7
    sample['support'][0, 7, 8:17] = 1
    sample['prediction_support'][0, 7, 8:17] = 1
    before = {name: value.copy() for name, value in sample.items() if isinstance(value, np.ndarray)}
    keep = _decide_slice(candidate, sample['score'][0], sample['support'][0],
                         sample['prediction_support'][0], sample['anchored'][0],
                         core_radius=1, rescue_radius=3)
    assert keep[7, 10] and not keep[7, 11]
    assert not np.any(keep & ~candidate)
    for name, value in before.items():
        np.testing.assert_array_equal(sample[name], value)
    # An unobserved candidate breaks the path; it is never assigned a score.
    sample['score'][0, 7, 9] = 0
    sample['prediction_support'][0, 7, 9] = 0
    keep = _decide_slice(candidate, sample['score'][0], sample['support'][0],
                         sample['prediction_support'][0], sample['anchored'][0],
                         core_radius=1, rescue_radius=3)
    assert not keep[7, 9:].any()


def test_custom_presets_are_independent_of_slab_boundaries_and_never_expand_candidates():
    rng = np.random.default_rng(418)
    sample = block((5, 29, 31))
    sample['candidate'][:] = rng.random(sample['candidate'].shape) < .8
    sample['score'][:] = rng.uniform(0, 4, sample['score'].shape)
    sample['support'][:] = rng.integers(0, 5, sample['support'].shape)
    sample['prediction_support'][:] = sample['support']
    sample['anchored'][:] = rng.random(sample['anchored'].shape) < .5
    for value in sample.values():
        if isinstance(value, np.ndarray): value.flags.writeable = False
    for name in ('confidence_core_rescue', 'hybrid_with_fill'):
        decide = load(EXAMPLES / (name + '.py'))['decide']
        whole = decide(sample)
        separate = []
        for z in range(5):
            slab = {key: value[z:z+1] if isinstance(value, np.ndarray) else value for key, value in sample.items()}
            slab.update(z0=z, z1=z+1)
            separate.append(decide(slab))
        np.testing.assert_array_equal(whole, np.concatenate(separate))
        assert not np.any(whole & ~sample['candidate'])


def test_hybrid_fill_cannot_bypass_a_failed_small_island_vote():
    candidate = np.zeros((25, 30), bool)
    candidate[3:18, 3:18] = True
    candidate[7:10, 23:26] = True
    support = candidate.astype(np.uint16) * 3
    support[7:10, 23:26] = 2
    score = support.astype(np.float64)
    kept = hybrid_slice(candidate, score, support, support, satellites=True, rescue=True, fill=True)
    assert kept[5:16, 5:16].all()
    assert not kept[7:10, 23:26].any()
