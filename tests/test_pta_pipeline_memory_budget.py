"""The depth-2 memory guard does not count clean reclaimable file cache as used."""
import pytest

from XTA import pta, workspace
from XTA.pta_scheduler import resolve_pta_pipeline_depth
from tests.test_cgroup_memory_budget import finite, proc_files

MEMINFO = "MemTotal:       8000000 kB\nMemFree:          10000 kB\nMemAvailable:    5000000 kB\n"
CLEAN_CACHE = "anon 250000\nfile 650000\ninactive_file 600000\nfile_dirty 0\nfile_writeback 0\n"


@pytest.mark.parametrize("v2", [True, False])
@pytest.mark.parametrize("stat,expected", [
    (CLEAN_CACHE, 700_000),
    # Dirty or writeback pages are not yet reclaimable.
    (CLEAN_CACHE.replace("file_dirty 0", "file_dirty 150000"), 550_000),
    # Without complete counters the cache stays counted as used.
    ("inactive_file 600000\n", 100_000),
    (None, 100_000),
])
def test_cgroup_budget_excludes_clean_inactive_file_cache(monkeypatch, capsys, v2, stat, expected):
    files, base = proc_files(monkeypatch, v2=v2)
    if stat is not None and not v2:
        stat = stat.replace("inactive_file", "total_inactive_file").replace(
            "file_dirty", "total_dirty").replace("file_writeback", "total_writeback")
    finite(files, base, "/job/step", 1_000_000, 900_000, v2=v2, stat=stat)
    files["/proc/meminfo"] = MEMINFO
    assert pta.available_memory_budget_bytes() == expected
    line = capsys.readouterr().out
    assert "PTA pipeline memory budget: cgroup_headroom_excluding_clean_inactive_file_cache=" in line
    assert "MemAvailable=4.8 GiB" in line


def test_tighter_ancestor_and_meminfo_bound_the_budget(monkeypatch):
    files, base = proc_files(monkeypatch)
    finite(files, base, "/job/step", 1_000_000, 900_000, stat=CLEAN_CACHE)
    finite(files, base, "/job", 950_000, 900_000)
    files["/proc/meminfo"] = MEMINFO
    assert pta.available_memory_budget_bytes() == 50_000
    files["/proc/meminfo"] = "MemAvailable: 10 kB\n"
    assert pta.available_memory_budget_bytes() == 10_240


def test_unknown_sources_keep_previous_fallbacks(monkeypatch, capsys):
    files, _base = proc_files(monkeypatch)
    assert pta.available_memory_budget_bytes() is None
    assert "budget=unavailable" in capsys.readouterr().out
    files["/proc/meminfo"] = MEMINFO
    assert pta.available_memory_budget_bytes() == 5_000_000 * 1024
    files.pop("/proc/self/cgroup")
    workspace._reset_cgroup_layout_cache()  # membership is resolved once per process
    assert pta.available_memory_budget_bytes() == 5_000_000 * 1024


def test_unreadable_usage_under_a_cap_is_conservative(monkeypatch):
    files, base = proc_files(monkeypatch)
    finite(files, base, "/job", 1_000_000, PermissionError())
    files["/proc/meminfo"] = MEMINFO
    assert pta.available_memory_budget_bytes() == 0


def test_page_cache_no_longer_downgrades_depth_two(monkeypatch):
    files, base = proc_files(monkeypatch)
    finite(files, base, "/job/step", 1_000_000, 900_000, stat=CLEAN_CACHE)
    files["/proc/meminfo"] = MEMINFO
    estimates = [240_000, 240_000]
    budget = pta.available_memory_budget_bytes()
    assert resolve_pta_pipeline_depth(requested_depth=2, volume_count=2, resident_estimates=estimates,
                                      available_bytes=budget).effective_depth == 2
    # Counting the cache as used (limit - usage) would have forced depth 1.
    assert resolve_pta_pipeline_depth(requested_depth=2, volume_count=2, resident_estimates=estimates,
                                      available_bytes=100_000).memory_limited
