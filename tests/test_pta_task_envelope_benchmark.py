"""Keep the task-envelope experiment honest without importing GPU runtimes."""
from copy import deepcopy
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_pta_task_envelope.py"
    spec = importlib.util.spec_from_file_location("pta_task_envelope_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abba_order_balances_first_and_last_positions():
    tool = load_tool()
    assert tool.abba_order(2) == ["baseline", "candidate", "candidate", "baseline"] * 2


def test_policy_partition_records_source_boundaries_originals_and_order():
    tool = load_tool()
    assert tool.partition_record(((None, 4, 7), (None, 2, 8))) == [[None, 4, 7], [None, 2, 8]]
    assert tool.partition_record(((None, 4), (7,))) != tool.partition_record(((None,), (4, 7)))


def test_output_inventory_checks_bytes_and_relative_names(tmp_path):
    tool = load_tool()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "image.jpg").write_bytes(b"same")
    (second / "image.jpg").write_bytes(b"same")
    assert tool.output_manifest(first) == tool.output_manifest(second)
    (second / "image.jpg").write_bytes(b"drift")
    assert tool.output_manifest(first) != tool.output_manifest(second)


def test_comparator_separates_batch_contract_from_output_equivalence():
    tool = load_tool()
    baseline = dict(files={"image.jpg": {"bytes": 4, "sha256": "a"}},
                    policy_partitions=[[[None, 3], [None, 9]]],
                    totals={"written": 4, "flips": 0},
                    candidate_states=[[0, True, True, ""]])
    actual = deepcopy(baseline)
    assert all(tool.compare_runs(baseline, actual).values())
    actual["policy_partitions"] = [[[None, 3]], [[None, 9]]]
    result = tool.compare_runs(baseline, actual)
    assert result["exact_files"] and result["exact_totals"] and result["exact_candidate_states"]
    assert not result["exact_policy_partitions"]
    actual["candidate_states"][0][1] = False
    assert not tool.compare_runs(baseline, actual)["exact_candidate_states"]


def test_default_ratio_three_has_a_custom_policy_partition_counterexample():
    """A wider envelope cannot be promoted from per-seed image parity alone."""
    from XTA.pta_scheduler import iter_compatible_work_batches
    from types import SimpleNamespace

    @dataclass
    class Frame:
        items: tuple

    @dataclass
    class Work:
        candidates: tuple
        image: object
        mask: object
        output_size: tuple = (2, 2)
        channel_kind: str = "gray"
        channel_count: int = 1

    source = (Path(__file__).resolve().parents[1] / "XTA/pta_workers.py").read_text(encoding="utf-8")
    selected = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                and node.name in {"_frame_render_candidate_count", "batch_gpu_frame_tasks"}]
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    namespace = {"GpuFrameBatchTask": lambda frames: SimpleNamespace(frames=frames)}
    exec(compile(ast.fix_missing_locations(module), "<production frame packing>", "exec"), namespace)
    frames = [Frame((("full", tuple(range(index * 3, index * 3 + 3))),)) for index in range(84)]
    array = SimpleNamespace(shape=(2, 2), is_cuda=False)

    def policy_calls(envelope):
        result = []
        for task in namespace["batch_gpu_frame_tasks"](frames, candidate_limit=envelope * 32):
            work = [Work(frame.items[0][1], array, array) for frame in task.frames]
            result.extend(tuple(candidate for item in batch for candidate in item.candidates)
                          for batch in iter_compatible_work_batches(work, candidate_limit=32))
        return result

    baseline, candidate = policy_calls(2), policy_calls(8)
    assert [len(batch) for batch in baseline] == [32, 31] * 4
    assert [len(batch) for batch in candidate] == [32] * 7 + [28]
    assert sorted(candidate_id for batch in baseline for candidate_id in batch) == list(range(252))
    assert sorted(candidate_id for batch in candidate for candidate_id in batch) == list(range(252))
    # An otherwise valid custom policy that uses len(batch) changes its output
    # for candidate 32: no seeds or source candidates were lost or duplicated.
    containing_size = lambda calls, index: next(len(batch) for batch in calls if index in batch)
    assert containing_size(baseline, 32) != containing_size(candidate, 32)
