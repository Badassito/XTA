"""Replay identity, disk admission, packet integrity and worker aggregation regressions."""
import multiprocessing as mp
import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import forward_replay as replay
from XTA.geometry import AugJob, ViewInfo, build_affine


def _cache(root, model, *, settings=None, name="test"):
    os.environ["XTA_YOLO_SWEEP_CACHE_DIR"] = str(root)
    return replay.replay_for_model([model], backend=name, settings=settings or {})


def _record_small_worker(root, model, marker):
    cache = _cache(root, model)
    inp = np.array([marker],np.uint8)
    key = cache.input_key(inp)
    cache.store(key,(inp,))
    assert cache.load(key)[0][0] == marker


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    root = tmp_path/"raw"
    model = tmp_path/"test.pt"
    model.write_bytes(b"fixed model")
    monkeypatch.setenv("XTA_YOLO_SWEEP_CACHE_DIR",str(root))
    monkeypatch.setenv("XTA_YOLO_SWEEP_CACHE_GIB","0.001")
    monkeypatch.delenv("XTA_YOLO_SWEEP_WORK_DIR",raising=False)
    monkeypatch.delenv("XTA_YOLO_SWEEP_VERIFY",raising=False)
    return root, model


def test_budget_and_disabled_cache(store_env, monkeypatch):
    root,model=store_env
    monkeypatch.setenv("XTA_YOLO_SWEEP_CACHE_GIB","0")
    assert replay.replay_for_model([model],backend="test",settings={}) is None
    assert not root.exists()
    monkeypatch.setenv("XTA_YOLO_SWEEP_CACHE_GIB","0.001")
    cache=_cache(root,model)
    large=np.zeros(600000,np.uint8)
    k=cache.input_key(np.array([2]))
    assert cache.store(k,(large,))
    assert not cache.store(cache.input_key(np.array([3])),(large,))
    state=replay.replay_cache_summary(root)
    assert state["entries"]==1
    assert state["raw_disk_bytes"]<=int(.001*2**30)
    assert cache.load(k)[0].size==len(large)


def test_identity_and_private_packet_copies(store_env):
    root,model=store_env
    cache=_cache(root,model)
    a=np.arange(120,dtype=np.float32).reshape(12,10)
    key=cache.input_key(a)
    assert key==cache.input_key(a.copy())
    assert key!=cache.input_key(a.reshape(10,12))
    assert key!=cache.input_key(a.astype(np.float16))
    assert key!=cache.input_key(a+1)
    assert key!=_cache(root,model,settings={"precision":"bf16"}).input_key(a)
    model.write_bytes(b"changed model artifact")
    assert key!=_cache(root,model).input_key(a)
    assert cache.load(key) is None
    assert cache.store(key,(a,np.empty((0,3),np.float32)))
    first,empty=cache.load(key)
    first.fill(-1)
    assert empty.shape==(0,3)
    assert np.array_equal(cache.load(key)[0],a)
    assert not list(root.rglob("*.raw"))
    assert replay.replay_cache_summary(root)["by_backend"]["test"]["inline_stored"]==1


def test_opt_in_crc_and_truncation_fall_back(store_env, monkeypatch):
    root,model=store_env
    monkeypatch.setenv("XTA_YOLO_SWEEP_VERIFY","1")
    cache=_cache(root,model)
    a=np.arange(4096,dtype=np.float32)
    key=cache.input_key(a)
    assert cache.store(key,(a,))
    payload=cache._packet_path(key)
    with payload.open("r+b") as f:
        f.seek(-1,os.SEEK_END)
        original=f.read(1)
        f.seek(-1,os.SEEK_END)
        f.write(bytes([original[0]^128]))
    assert cache.load(key) is None
    payload.write_bytes(payload.read_bytes()[:-1])
    assert cache.load(key) is None
    assert replay.replay_cache_summary(root)["by_backend"]["test"]["invalid_packet"]==2


def test_logical_keys_keep_geometry_and_renderer(store_env):
    root,model=store_env
    cache=_cache(root,model)
    view=ViewInfo("Transverse",5,10,12,"clamp")
    aff=build_affine(view.name,12,10,12,0.0,"clamp")
    def key(start=0, render="cpu", native_shape=None):
        job=AugJob("a0",0.,aff)
        task=dict(sweep_source_stamp=["video",42],job=job,view=view,slice_start=start,
                  native_resize={"shape":native_shape,"dtype":"uint8"} if native_shape else None)
        source=SimpleNamespace()
        replay.bind_worker_replay_source(source,task,render)
        return cache.input_key(np.empty((1,1,12,12),np.uint8),source=source,batch_index=0)
    assert key()==key()
    assert key()!=key(start=1)
    assert key()!=key(render="gpu")
    assert key(native_shape=(2,5,5))!=key(native_shape=(3,5,5))


def test_multiple_workers_share_admission_and_flush_statistics(store_env):
    root,model=store_env
    cache=_cache(root,model)
    jobs=[mp.get_context("spawn").Process(target=_record_small_worker,args=(root,model,i)) for i in range(2)]
    for p in jobs:
        p.start()
    for p in jobs:
        p.join(30)
        assert p.exitcode==0
    summary=replay.replay_cache_summary(root)
    assert summary["entries"]==2
    assert summary["by_backend"]["test"]["hit"]==2
    assert summary["by_device"]["cpu"]["test"]["hit"]==2
    assert cache.load(cache.input_key(np.array([1],np.uint8)))[0].item()==1
    with sqlite3.connect(root/"packets.sqlite") as db:
        assert db.execute("SELECT count(*) FROM packets WHERE ready=0").fetchone()[0]==0


def test_sweep_gpu_tie_remains_stealable_and_stable_across_cursors():
    from XTA.tta_scheduler import _sweep_gpu_tiebreak
    task = dict(kind="tile", job_id="a/12", slice_start=8, model_name="Y",
                view=SimpleNamespace(name="Sagittal"), sweep_source_stamp=["witness"])
    positions_a, positions_b = {0: 0, 1: 1, 2: 2}, {0: 2, 1: 0, 2: 1}
    _sweep_gpu_tiebreak(task, [0,1,2], positions_a)
    _sweep_gpu_tiebreak(task, [2,0,1], positions_b)
    preferred = min(positions_a, key=positions_a.get)
    assert min(positions_b, key=positions_b.get) == preferred
    # Only the final key changes. Either a nonzero predicted load or an extra
    # inflight task still breaks the tie in favor of an already-free worker.
    assert min(positions_a, key=lambda w:(int(w==preferred),0,positions_a[w])) != preferred
    assert min(positions_a, key=lambda w:(0,int(w==preferred),positions_a[w])) != preferred
    remaining = {w: rank for rank,w in enumerate(sorted(set(positions_a)-{preferred}))}
    original = remaining.copy()
    _sweep_gpu_tiebreak(task, [0,1,2], remaining)
    assert remaining == original
    for changes in ({"sweep_source_stamp":None},{"augmentation_pass_tasks":[{}]}):
        ordinary = {0:1,1:0,2:2}
        _sweep_gpu_tiebreak({**task, **changes}, [0,1,2], ordinary)
        assert ordinary == {0:1,1:0,2:2}
