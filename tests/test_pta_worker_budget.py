"""PTA uses actual affinity capacity and only budgets overlap that can occur."""
from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA.pta_scheduler import plan_pta_cpu_budget, resolve_pta_pipeline_depth


LOCAL = tuple(range(16)) + tuple(range(104, 120))
ALLOWED = LOCAL + tuple(range(32, 48)) + tuple(range(136, 152))
MASKS = tuple(tuple(range(rank, 16, 4)) + tuple(range(104 + rank, 120, 4)) for rank in range(4))


def plan(**changes):
    values = dict(worker_budget=64, requested_frame_workers=0, allowed_cpus=ALLOWED,
                  worker_cpu_order=LOCAL + tuple(cpu for cpu in ALLOWED if cpu not in LOCAL),
                  gpu_count=4, gpu_cpu_sets=MASKS, topology_aware=True, overlapping=False)
    values.update(changes)
    return plan_pta_cpu_budget(**values)


def extracted_function(name, namespace):
    source = (Path(__file__).resolve().parents[1] / 'XTA' / 'pta.py').read_text(encoding='utf-8')
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<actual PTA budgeting function>', 'exec'), namespace)
    return namespace[name]


class PtaWorkerBudgetTests(unittest.TestCase):
    def test_single_volume_default_depth_uses_all_planning_cpus_and_caps_renderers(self):
        depth = resolve_pta_pipeline_depth(requested_depth=2, volume_count=1)
        budget = plan(overlapping=depth.overlapping)
        self.assertEqual(depth.effective_depth, 1)
        self.assertFalse(budget.overlapping)
        self.assertEqual(budget.frame_workers, 4)
        self.assertEqual(budget.gpu_render_threads_by_owner, (8, 8, 8, 8))
        self.assertEqual(budget.gpu_render_threads, 8)
        self.assertEqual(budget.gpu_cpu_sets, MASKS)
        self.assertEqual((budget.planning_workers, budget.io_workers), (64, 16))
        self.assertEqual((budget.bootstrap_workers, budget.bootstrap_io_workers), (64, 16))

    def test_multiple_volumes_reserve_actual_unique_affinity_not_nominal_threads(self):
        depth = resolve_pta_pipeline_depth(requested_depth=2, volume_count=2,
                                          resident_estimates=(10, 10), available_bytes=100)
        budget = plan(overlapping=depth.overlapping)
        self.assertTrue(budget.overlapping)
        self.assertEqual(budget.render_cpu_count, 32)
        self.assertEqual(budget.gpu_cpu_sets, MASKS)
        self.assertEqual((budget.planning_workers, budget.io_workers), (32, 16))
        self.assertEqual(set(budget.planning_cpu_order), set(ALLOWED) - set(LOCAL))
        self.assertEqual(budget.bootstrap_workers, 64)

    def test_memory_reduced_depth_and_explicit_depth_one_keep_full_planning(self):
        for requested, estimates, available in ((2, (40, 40), 100), (1, (1, 1), 100)):
            with self.subTest(requested=requested):
                depth = resolve_pta_pipeline_depth(requested_depth=requested, volume_count=2,
                    resident_estimates=estimates, available_bytes=available)
                budget = plan(overlapping=depth.overlapping)
                self.assertEqual(depth.effective_depth, 1)
                self.assertEqual((budget.planning_workers, budget.io_workers), (64, 16))
        self.assertTrue(resolve_pta_pipeline_depth(requested_depth=2, volume_count=2,
                       resident_estimates=(40, 40), available_bytes=100).memory_limited)

    def test_all_local_cpus_leave_a_real_disjoint_planning_reservation(self):
        allowed = tuple(range(64))
        masks = tuple(tuple(range(rank, 64, 4)) for rank in range(4))
        budget = plan(allowed_cpus=allowed, worker_cpu_order=allowed, gpu_cpu_sets=masks, overlapping=True)
        self.assertEqual(budget.gpu_render_threads_by_owner, (12, 12, 12, 12))
        self.assertEqual(budget.render_cpu_count, 48)
        self.assertEqual(budget.planning_workers, 16)
        for before, after in zip(masks, budget.gpu_cpu_sets):
            self.assertTrue(set(after) <= set(before))
        self.assertFalse(set(budget.planning_cpu_order) & set().union(*map(set, budget.gpu_cpu_sets)))

    def test_heterogeneous_affinity_has_per_owner_limits_without_global_narrowing(self):
        masks = (tuple(range(2)), tuple(range(2, 8)), tuple(range(8, 20)), tuple(range(20, 32)))
        budget = plan(worker_budget=32, allowed_cpus=tuple(range(32)), worker_cpu_order=tuple(range(32)),
                      gpu_cpu_sets=masks, overlapping=True)
        self.assertEqual(budget.gpu_render_threads_by_owner, (2, 6, 8, 8))
        self.assertEqual(budget.gpu_render_threads, 8)
        self.assertEqual((budget.render_cpu_count, budget.planning_workers), (24, 8))
        self.assertTrue(all(threads <= len(cpus) for threads, cpus in zip(budget.gpu_render_threads_by_owner, budget.gpu_cpu_sets)))

    def test_disabled_topology_budgets_threads_without_setting_any_affinity(self):
        budget = plan(topology_aware=False, overlapping=True)
        self.assertEqual(budget.gpu_cpu_sets, ((), (), (), ()))
        self.assertEqual(budget.gpu_render_threads_by_owner, (12, 12, 12, 12))
        self.assertEqual(budget.planning_workers, 16)
        self.assertEqual(budget.planning_cpu_order, ())
        self.assertEqual(budget.bootstrap_cpu_order, ())
        self.assertEqual(budget.render_cpu_order, ())

    def test_explicit_worker_and_frame_requests_respect_allowed_and_local_capacity(self):
        budget = plan(worker_budget=16, requested_frame_workers=128, overlapping=True)
        self.assertEqual(budget.worker_budget, 16)
        self.assertEqual(budget.gpu_render_threads_by_owner, (3, 3, 3, 3))
        self.assertEqual(budget.planning_workers, 4)
        self.assertEqual(len(budget.planning_cpu_order), 4)
        self.assertEqual(budget.bootstrap_workers, 16)
        self.assertTrue(set(budget.render_cpu_order) <= set(ALLOWED))
        self.assertEqual(plan(worker_budget=128).worker_budget, 64)
        self.assertEqual(plan(requested_frame_workers=8).gpu_render_threads_by_owner, (2, 2, 2, 2))

    def test_too_small_concurrent_budget_disables_overlap_without_losing_owners(self):
        budget = plan(worker_budget=4, overlapping=True)
        self.assertFalse(budget.overlapping)
        self.assertEqual(budget.frame_workers, 4)
        self.assertEqual(budget.gpu_render_threads_by_owner, (1, 1, 1, 1))
        self.assertEqual(budget.planning_workers, 4)
        smaller = plan(worker_budget=2, overlapping=True)
        self.assertFalse(smaller.overlapping)
        self.assertEqual(smaller.worker_budget, 2)
        self.assertEqual(smaller.frame_workers, 4)
        self.assertEqual(smaller.gpu_render_threads_by_owner, (1, 1, 1, 1))
        self.assertEqual((smaller.planning_workers, smaller.io_workers), (2, 2))
        self.assertTrue(all(set(cpus) <= set(original) for cpus, original in zip(smaller.gpu_cpu_sets, MASKS)))

    def test_cpu_only_rendering_uses_actual_overlap_as_well(self):
        single = plan(gpu_count=0, gpu_cpu_sets=())
        overlap = plan(gpu_count=0, gpu_cpu_sets=(), overlapping=True)
        self.assertEqual((single.frame_workers, single.planning_workers), (64, 64))
        self.assertEqual((overlap.frame_workers, overlap.planning_workers), (48, 16))
        self.assertFalse(set(overlap.render_cpu_order[:48]) & set(overlap.planning_cpu_order))

    def test_actual_planning_wrapper_selects_bootstrap_overlap_and_restored_idle_budget(self):
        budget = plan(overlapping=True)
        progress = {}
        called = mock.Mock(side_effect=lambda index, workers: (index, workers))
        affinity = mock.Mock(side_effect=lambda _cpus: contextlib.nullcontext())
        namespace = dict(progress_by_gen=progress, cpu_budget=budget, planning_workers=budget.planning_workers,
                         planning_cpu_order=budget.planning_cpu_order, args=SimpleNamespace(topology_aware=True),
                         _planning_phase_affinity=affinity, _plan_volume_with_budget=called)
        wrapper = extracted_function('_plan_volume', namespace)
        self.assertEqual(wrapper(0), (0, 64))
        self.assertEqual(affinity.call_args.args[0], ALLOWED)
        progress[0] = SimpleNamespace(pending_a=1, pending_b=0)
        self.assertEqual(wrapper(1), (1, 32))
        self.assertEqual(affinity.call_args.args[0], budget.planning_cpu_order)
        progress[0].pending_a = 0
        self.assertEqual(wrapper(2), (2, 64))

    def test_actual_affinity_scope_restores_parent_even_if_planning_fails(self):
        bind = mock.Mock(return_value=True)
        namespace = dict(contextlib=contextlib, _allowed_cpu_tuple=lambda: ALLOWED,
                         bind_current_thread_to_cpus=bind)
        affinity = extracted_function('_planning_phase_affinity', namespace)
        with self.assertRaisesRegex(RuntimeError, 'planning failed'):
            with affinity(ALLOWED[-8:]):
                raise RuntimeError('planning failed')
        self.assertEqual(bind.call_args_list, [mock.call(ALLOWED[-8:]), mock.call(ALLOWED)])
        bind.reset_mock()
        with affinity(()):
            pass
        bind.assert_not_called()


if __name__ == '__main__':
    unittest.main()
