"""Deterministic topology cases for compiled-versus-reference CPU benchmarks.

The adjacency cases call the compiled dispatcher directly. Coherent planes admit its
Numba row-run algorithm; fragmented planes refuse that route and use its bounded
Numba hash. The reference is the NumPy pair-code implementation. Union cases call
the Numba batch kernel and Python graph loop on identical prepared pairs.
"""

from __future__ import annotations

import numpy as np

from XTA import topology
from tests.reference_backends.topology import union_pair_codes_python

from .contracts import Case


def _adjacency_case(
    name: str,
    previous: np.ndarray,
    current: np.ndarray,
    *,
    expected_compiled_path: str,
    max_capacity: int | None = None,
) -> Case:
    offsets = topology._adjacent_xy_offsets_for_3d_connectivity(26)
    previous.setflags(write=False)
    current.setflags(write=False)

    def reference() -> np.ndarray:
        return topology._adjacent_gid_pair_codes_numpy(previous, current, offsets)

    def compiled() -> np.ndarray:
        kwargs = {} if max_capacity is None else {
            'initial_capacity': min(128, max_capacity), 'max_capacity': max_capacity,
        }
        return topology._compiled_adjacent_gid_pair_codes(
            previous, current, offsets, **kwargs,
        )

    return Case(
        name=name,
        reference=reference,
        compiled=compiled,
        work_units=int(previous.size),
        unit='pixels per plane',
        metadata={
            'subsystem': 'topology-adjacency',
            'shape': list(previous.shape),
            'dtype': str(previous.dtype),
            'connectivity': 26,
            'expected_compiled_path': expected_compiled_path,
            'max_hash_capacity': max_capacity,
        },
    )


def _coherent_planes(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.ogrid[:size, :size]
    radius = size // 3
    disk = ((y - size // 2) ** 2 + (x - size // 2) ** 2) < radius ** 2
    previous = disk.astype(np.uint16)
    return previous, np.roll(previous, 2, axis=1)


def _patch_planes(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.ogrid[:size, :size]
    previous = ((x // 64 + (y // 64) * (size // 64 + 1)) % 30000 + 1).astype(np.uint16)
    previous[(x % 64 < 4) | (y % 64 < 4)] = 0
    return previous, np.roll(previous, 3, axis=1)


def _fragmented_planes(size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7407)
    previous = rng.integers(0, 100, size=(size, size), dtype=np.uint16)
    return previous, np.roll(previous, 1, axis=1)


def _spill_planes(size: int) -> tuple[np.ndarray, np.ndarray]:
    previous = np.arange(1, size * size + 1, dtype=np.uint32).reshape(size, size)
    return previous, np.roll(previous, 1, axis=1)


def _canonical_union_result(uf: topology._UnionFind, node_count: int) -> np.ndarray:
    """Compare partitions and boundary reachability, independent of root naming."""
    roots = uf.root_map().astype(np.int64, copy=False)
    ids = np.arange(node_count + 1, dtype=np.int64)
    representative = np.full(int(uf.parent.size), node_count + 1, dtype=np.int64)
    np.minimum.at(representative, roots, ids)
    partition = representative[roots]
    boundary = uf.touches_boundary[roots].astype(np.int64, copy=False)
    return np.column_stack((partition, boundary))


def _union_case(name: str, node_count: int, a_ids: np.ndarray, b_ids: np.ndarray) -> Case:
    a_ids = np.ascontiguousarray(a_ids, dtype=np.int64)
    b_ids = np.ascontiguousarray(b_ids, dtype=np.int64)
    a_ids.setflags(write=False)
    b_ids.setflags(write=False)
    boundary_nodes = np.arange(1, node_count + 1, max(1, node_count // 97), dtype=np.int64)
    initial = topology._UnionFind()
    initial.new_ids(node_count)
    initial.touches_boundary[boundary_nodes] = True

    def initial_union() -> topology._UnionFind:
        uf = topology._UnionFind()
        uf.parent = initial.parent.copy()
        uf.rank = initial.rank.copy()
        uf.touches_boundary = initial.touches_boundary.copy()
        uf._size = node_count + 1
        return uf

    def reference() -> np.ndarray:
        uf = initial_union()
        union_pair_codes_python(uf, a_ids, b_ids)
        return _canonical_union_result(uf, node_count)

    def compiled() -> np.ndarray:
        if topology._numba_union_find_batch_kernel is None:
            raise RuntimeError('Numba union-find batch kernel is unavailable')
        uf = initial_union()
        topology._numba_union_find_batch_kernel(
            uf.parent, uf.rank, uf.touches_boundary, a_ids, b_ids,
        )
        return _canonical_union_result(uf, node_count)

    return Case(
        name=name,
        reference=reference,
        compiled=compiled,
        work_units=int(a_ids.size),
        unit='union pairs',
        metadata={
            'subsystem': 'topology-union-find',
            'nodes': node_count,
            'pairs': int(a_ids.size),
            'marked_boundary_nodes': int(boundary_nodes.size),
            'compiled_path': '_numba_union_find_batch_kernel',
            'reference_path': 'tests.reference_backends.topology.union_pair_codes_python',
            'timed_common_work': 'copy initial UF arrays and canonicalize output',
        },
    )


def make_cases(workload: str = 'smoke') -> list[Case]:
    """Build bounded representative cases without executing benchmark work."""
    if workload not in ('smoke', 'scaled'):
        raise ValueError(f'Unknown topology workload: {workload!r}')

    run_size = 512 if workload == 'smoke' else 2048
    fragmented_size = 512 if workload == 'smoke' else 1024
    spill_size = 128 if workload == 'smoke' else 256
    union_nodes = 8192 if workload == 'smoke' else 131072

    coherent = _coherent_planes(run_size)
    patches = _patch_planes(run_size)
    fragmented = _fragmented_planes(fragmented_size)
    spill = _spill_planes(spill_size)

    chain = np.arange(1, union_nodes, dtype=np.int64)
    chain_a = np.concatenate((chain, chain, chain[:-1]))
    chain_b = np.concatenate((chain + 1, chain + 1, chain[:-1] + 2))

    rng = np.random.default_rng(7419)
    shuffled = rng.permutation(np.arange(1, union_nodes + 1, dtype=np.int64))
    disjoint_a = shuffled[::2]
    disjoint_b = shuffled[1::2]
    sparse_a = np.concatenate((disjoint_a, disjoint_a))
    sparse_b = np.concatenate((disjoint_b, disjoint_b))

    return [
        _adjacency_case('adjacency-coherent-run', *coherent, expected_compiled_path='numba-row-runs'),
        _adjacency_case('adjacency-patches-run', *patches, expected_compiled_path='numba-row-runs'),
        _adjacency_case('adjacency-fragmented-hash', *fragmented, expected_compiled_path='numba-hash-after-run-refusal'),
        _adjacency_case(
            'adjacency-hash-spills', *spill, expected_compiled_path='numba-hash-spill',
            max_capacity=1024,
        ),
        _union_case('union-chain-and-repeats', union_nodes, chain_a, chain_b),
        _union_case('union-disjoint-and-repeats', union_nodes, sparse_a, sparse_b),
    ]
