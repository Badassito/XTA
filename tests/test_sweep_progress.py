import json

from XTA.sweep_progress import FanoutProgress


def test_partial_progress_counts_done_work_across_cutoffs(tmp_path):
    progress = FanoutProgress(tmp_path, (50, 60, 70), total_views=2)
    progress.observe_batch(rendered_frames=9, render_batches=3, forward_calls=3, mask_decodes=27)
    progress.view_ready("view-a", nframes=9)
    progress.view_ready("view-a", nframes=9)
    progress.view_projected("view-a", cutoffs=(50,), geometry_passes=1)
    progress.view_projected("view-a", cutoffs=(60, 70), geometry_passes=1)
    progress.view_published("view-a", 50)
    progress.view_published("view-a", 70)
    path = tmp_path / "fanout_progress.json"
    live = json.loads(path.read_text())
    assert [c["views_published"] for c in live["cutoffs"]] == [1, 0, 1]
    assert [c["native_frames_ready"] for c in live["cutoffs"]] == [9, 9, 9]
    assert live["work"] == {"rendered_frames": 9, "render_batches": 3, "forward_calls": 3,
                            "mask_decodes": 27, "geometry_passes": 2}
    progress.finish("producer_complete", work={"anchors_once": 78})
    final = json.loads(path.read_text())
    assert final["status"] == "producer_complete"
    assert final["work"]["anchors_once"] == 78
    assert final["work"]["geometry_passes"] == 2
