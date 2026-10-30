#!/usr/bin/env python3
"""Independent old-source/new-source PTA spherical host-cache qualification.

Fresh CUDA-owner subprocesses execute the same canonical spherical plans, policy
bytes, source identities, task partitions and publication contract. No GPU cache
limit, precision or sampling policy is changed. Requires a separately granted
measurement slot; each child also owns an atomic GPU_LOCK until full retirement.
The synthetic local workload is not a cluster throughput forecast.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def scratch_root(source_root):
    """Frozen source trees live inside Scratch rather than beside it."""
    return next((parent for parent in source_root.parents if parent.name.casefold() == "scratch"),
                source_root.parent / "Scratch")


SCRATCH = scratch_root(ROOT)
POLICY_SHA = "1b2739997905c6ba2905c7e2613dd8c7c676dddfb16b6d518ef94286a481f4c8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shared_helpers():
    path = ROOT / "tools/benchmark_pta_task_envelope.py"
    spec = importlib.util.spec_from_file_location("_pta_cache_benchmark_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counter_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def compare_results(reference, candidate):
    names = ("files", "policy_partitions", "candidate_states", "totals", "workload")
    comparison = {"exact_" + name: reference[name] == candidate[name] for name in names}
    if reference.get("raw_render") is not None and candidate.get("raw_render") is not None:
        comparison["exact_raw_render"] = reference["raw_render"] == candidate["raw_render"]
        comparison["exact_raw_policy"] = reference["raw_policy"] == candidate["raw_policy"]
        comparison["exact_qualification_partitions"] = reference["qualification_partitions"] == candidate["qualification_partitions"]
    if "classification_outputs" in reference and "classification_outputs" in candidate:
        comparison["exact_classification_outputs"] = reference["classification_outputs"] == candidate["classification_outputs"]
    return comparison


def choose_key_groups(views, key_fn, count):
    grouped = defaultdict(list)
    for view in views:
        if view.family == "spherical" and view.shared_view is not None:
            grouped[key_fn(view.shared_view)].append(view)
    selected = list(grouped.items())[:count]
    if len(selected) != count or count < 6:
        raise ValueError("Need at least six distinct canonical direction keys")
    if any(len(group) < 2 for _key, group in selected):
        raise ValueError("Selected geometry must provide at least two radius bands per key")
    return selected


def spherical_fixture_arguments(size):
    return ["--input", "synthetic", "--imgsz", str(size),
        "--enable_tilted", "transverse:15:both", "transverse:30:both", "transverse:45:vertical", "--enable_spherical",
        "transverse", "tilted_transverse", "--spherical_min_radius", "2"]


def make_work(args, augmentation, run_dir):
    import numpy as np
    from XTA import pta, pta_workers, spherical_cuda
    from XTA.pta_config import parse_pta_args
    from XTA.pta_rendering import TileConfig
    from XTA.pta_dataset import OutputCandidate, plan_augmented_versions

    edge, size = args.cube_edge, args.imgsz
    yy, xx = np.indices((edge, edge), dtype=np.int32)
    base = ((13 * xx + 7 * yy + ((xx // 13) ^ (yy // 17))) % 256).astype(np.uint8)
    image = np.empty((edge, edge, edge), np.uint8)
    mask = np.empty_like(image)
    coverage = np.empty_like(image) if args.coverage else None
    for z in range(edge):
        image[z] = base + np.uint8(z % 256)
        mask[z] = ((xx // 47 + yy // 53 + z // 61) % 5 == 0).astype(np.uint8)
        if coverage is not None:
            coverage[z] = ((xx + yy + z) % 29 > 2).astype(np.uint8)
    config = parse_pta_args(spherical_fixture_arguments(size))
    views, _ = pta.compile_v18_pta_views(t_dim=edge, h=edge, w=edge,
                                         config=config, azimuthal_native_raster=size)
    measured_views = [view for view in views if view.shared_view is not None
                      and float(getattr(view.shared_view, "tilt_angle_deg", 0)) != 45.]
    warm_view = next(view for view in views if view.family == "spherical" and view.shared_view is not None
                     and float(getattr(view.shared_view, "tilt_angle_deg", 0)) == 45.)
    groups = choose_key_groups(measured_views, lambda view: spherical_cuda.spherical_direction_key(view, "cuda:0")[1:], args.keys)
    plans, plan_indices = [], {}
    for _key, variants in groups:
        for view in variants:
            aff = pta.build_affine(view.src_w, view.src_h, 0., view.pad_mode, size, shared_view=view.shared_view)
            plan = pta.build_render_plan(view=view, aff=aff, tag=view.name,
                out_dir=run_dir / "plan", stem="spherical-host-cache",
                tile_configs=(TileConfig(args.tile_size, args.tile_size, f"s{args.tile_size}"),),
                save_overlay=False, imgsz=size, label_enabled=True,
                publish_images=False, publish_labels=False)
            plan_indices[view.name] = len(plans)
            plans.append(plan)
    addresses, originals, used_addresses = [], [], set()
    for cycle in range(args.rounds):
        for key_index, (_key, variants) in enumerate(groups):
            view = variants[cycle % len(variants)]
            plan_index = plan_indices[view.name]
            plan = plans[plan_index]
            frame = int(view.num_slices) // 3 + cycle // len(variants)
            if frame >= int(view.num_slices) or (plan_index, frame) in used_addresses:
                raise ValueError("Workload must use distinct native frames, not native-plane cache hits")
            used_addresses.add((plan_index, frame))
            if not plan.tile_layout:
                raise RuntimeError("Canonical plan did not create tiles")
            tile = plan.tile_layout[(key_index + cycle) % len(plan.tile_layout)]
            items = ("full", str(tile.tile_tag))
            addresses.append((plan_index, frame, items))
            for item_key in items:
                originals.append(OutputCandidate(order=len(originals), source_order=len(originals),
                    volume_name="spherical-host-cache", parent_view_tag=plan.tag,
                    output_tag=plan.tag if item_key == "full" else item_key,
                    item_key=item_key, frame_idx=frame, is_tile=item_key != "full",
                    label_enabled=True, foreground=True, channel_kind="gray"))
    versions, _ = plan_augmented_versions(originals, augmentation=augmentation,
        augmentation_ratio=args.ratio, split_active=False, augmented_foregrounds={},
        require_augmented_foregrounds=False)
    by_source = defaultdict(list)
    for candidate in versions:
        by_source[(candidate.parent_view_tag, candidate.frame_idx, candidate.item_key)].append(candidate)
    frames = [pta_workers.FrameRenderTask(index, frame,
        tuple((key, tuple(by_source[(plans[index].tag, frame, key)])) for key in items))
        for index, frame, items in addresses]
    warm_aff = pta.build_affine(warm_view.src_w, warm_view.src_h, 0., warm_view.pad_mode, size, shared_view=warm_view.shared_view)
    warm_plan = pta.build_render_plan(view=warm_view, aff=warm_aff, tag=warm_view.name,
        out_dir=run_dir / "warm-plan", stem="spherical-host-cache-warmup",
        tile_configs=(TileConfig(args.tile_size, args.tile_size, f"s{args.tile_size}"),),
        save_overlay=False, imgsz=size, label_enabled=True, publish_images=False, publish_labels=False)
    warm_index=len(plans);plans.append(warm_plan)
    warm_frame=int(warm_view.num_slices)//3
    warm_items=("full",str(warm_plan.tile_layout[0].tile_tag))
    warm_originals=[OutputCandidate(order=index,source_order=index,volume_name="spherical-host-cache-warmup",
        parent_view_tag=warm_plan.tag,output_tag=warm_plan.tag if key=="full" else key,item_key=key,
        frame_idx=warm_frame,is_tile=key!="full",label_enabled=True,foreground=True,channel_kind="gray")
        for index,key in enumerate(warm_items)]
    warm_versions,_=plan_augmented_versions(warm_originals,augmentation=augmentation,
        augmentation_ratio=args.ratio,split_active=False,augmented_foregrounds={},require_augmented_foregrounds=False)
    warm_frames=[pta_workers.FrameRenderTask(warm_index,warm_frame,tuple((key,tuple(candidate for candidate in warm_versions if candidate.item_key==key)) for key in warm_items))]
    workload = dict(source_shape=list(image.shape), source_bytes=sum(array.nbytes for array in (image, mask, coverage) if array is not None),
        source_sha256=[hashlib.sha256(memoryview(array).cast("B")).hexdigest() for array in (image, mask, coverage) if array is not None],
        direction_keys=[key for key, _variants in groups], bands_per_key=[len(variants) for _key, variants in groups],
        candidate_count=len(versions), frame_count=len(frames),
        addresses=[(plans[index].tag, frame, list(items)) for index, frame, items in addresses])
    workload['unrelated_warmup_key']=spherical_cuda.spherical_direction_key(warm_view.shared_view,"cuda:0")[1:]
    if workload['unrelated_warmup_key'] in workload['direction_keys']:
        raise AssertionError('Warmup key must differ from every measured geometry key')
    return image, mask, coverage, plans, frames, workload, warm_frames


class RssSampler:
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.start = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def host_snapshot(runtime):
    renderer = runtime.get("azimuthal_renderer")
    cache = runtime.get("spherical_host_cache") or getattr(renderer, "_spherical_host_direction_cache", None)
    if cache is None:
        return None
    for name in ("snapshot", "stats", "cache_stats"):
        value = getattr(cache, name, None)
        if callable(value):
            return value()
        if isinstance(value, dict):
            return dict(value)
    raise RuntimeError("Installed host cache has no supported diagnostic snapshot")


def validate_host_snapshot(snapshot, *, cleared=False):
    if snapshot is None:
        return
    if snapshot['live_bytes'] > snapshot['budget_bytes'] or snapshot['peak_live_bytes'] > snapshot['budget_bytes']:
        raise AssertionError('Authoritative host-cache live bytes exceeded the explicit cap')
    # Cached and borrowed overlap: they are deliberately NOT summed.
    if any(snapshot[name] for name in ('building_bytes','borrowed_bytes','scratch_bytes','quarantined_bytes')):
        raise AssertionError('Settled workload retained a builder, lease, scratch buffer or quarantine')
    if cleared and (snapshot['live_bytes'] or snapshot['entries']):
        raise AssertionError('Fenced generation retirement did not clear the host memo')


def run_work(args, runtime, data, destination, *, capture_raw=False):
    import numpy as np
    import torch
    from XTA import pta_workers as workers
    helper = shared_helpers()
    image, mask, coverage, plans, originals, workload = data[:6]
    frames = helper.copy_frames(originals)
    tasks = workers.batch_gpu_frame_tasks(frames, candidate_limit=2 * args.batch_size)
    destination.mkdir(parents=True)
    workers._WORKER_STATIC["out_dir"] = destination
    partitions, raw_policy, raw_render = [], [], {}
    actual_apply = runtime["policy"].apply_batch_many
    actual_render = workers._render_gpu_item_group
    def tensor_record(value):
        if value is None:
            return None
        array = value.detach().to("cpu").numpy() if bool(getattr(value, "is_cuda", False)) else np.asarray(value)
        array = np.ascontiguousarray(array)
        return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(memoryview(array).cast("B")).hexdigest())
    def apply(**kwargs):
        partition = helper.partition_record(kwargs["seeds"])
        partitions.append(partition)
        result = actual_apply(**kwargs)
        if capture_raw:
            raw_policy.append(dict(partition=partition, images=tensor_record(result[0]), masks=tensor_record(result[1])))
        return result
    def render(*values, **kwargs):
        result = actual_render(*values, **kwargs)
        if capture_raw:
            workers._wait_for_gpu_work_ready(runtime, result)
            for item in result:
                raw_render[item.context] = [tensor_record(item.image), tensor_record(item.mask), tensor_record(item.semantic_coverage)]
        return result
    totals, warnings = Counter(), Counter()
    started, cpu_started = time.perf_counter(), time.process_time()
    with mock.patch.object(runtime["policy"], "apply_batch_many", apply), \
         mock.patch.object(workers, "_render_gpu_item_group", render), \
         mock.patch.object(workers, "write_image", side_effect=AssertionError("CPU image encode fallback forbidden")):
        for task in tasks:
            written, flips, counts, _examples, _classification = workers.execute_gpu_frame_batch_task(image, mask, plans, task, coverage)
            totals["written"] += written
            totals["flips"] += sum(flips.values())
            warnings.update(counts)
        torch.cuda.synchronize(args.device)
    result = dict(wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started,
        totals=dict(totals), warnings=dict(warnings), policy_partitions=partitions,
        candidate_states=helper.candidate_states(frames), task_count=len(tasks), workload=workload,
        raw_render=raw_render if capture_raw else None, raw_policy=raw_policy if capture_raw else None)
    if any("fallback" in name for name in warnings):
        raise RuntimeError(f"Unexpected runtime fallback: {dict(warnings)}")
    if warnings["pta_cuda_categorical_items"] != 2 * len(frames):
        raise AssertionError("All full/tile categorical sources must actually render on CUDA")
    result["files"] = helper.output_manifest(destination)
    count = result["totals"]["written"]
    if sum(name.endswith('.jpg') for name in result['files']) != count or sum(name.endswith('.txt') for name in result['files']) != count:
        raise AssertionError("Settled output counts disagree")
    if args.coverage and sum(name.endswith('.png') for name in result['files']) != count:
        raise AssertionError("Semantic coverage output count disagrees")
    return result


def classification_admission_case(args,runtime,data):
    """Classification memo must survive the first intensity source admission."""
    import numpy as np
    import torch
    from XTA import pta_workers as workers,spherical_cuda as sc
    from XTA.pta_gpu_classification import GpuClassificationTask,classify_task
    image,mask,coverage,plans,frames=data[:5]
    frame=frames[0];plan=plans[frame.plan_idx]
    renderer=runtime['azimuthal_renderer']
    if renderer._volume_gpu is not None:
        raise AssertionError('Classification qualification requires a cold intensity source')
    task=GpuClassificationTask(((frame.plan_idx,frame.frame_idx,tuple(key for key,_ in frame.items)),),args.coverage)
    truth,remaining,stats=classify_task(runtime,mask,plans,task,identity='classification-proof-mask',coverage=coverage)
    if remaining or stats['gpu_queries']!=len(frame.items):
        raise AssertionError('Classification proof did not run every full/tile CUDA query')
    host_before=host_snapshot(runtime)
    original=sc._build_host_direction_block
    with mock.patch.object(sc,'_build_host_direction_block',side_effect=original) as build:
        result=workers._gpu_projected_item_image(runtime,image,plan,frame.frame_idx,'full')
        if result is None:raise AssertionError('First intensity source admission fell back')
        rendered,_ready=result
        torch.cuda.synchronize(args.device)
        array=np.ascontiguousarray(rendered.detach().to('cpu').numpy())
        calls=build.call_count
    return dict(host_before_intensity=host_before,host_after_intensity=host_snapshot(runtime),
        intensity_host_build_block_calls=calls,classification_stats=stats,
        outputs=dict(truth=sorted((repr(key),value) for key,value in truth.items()),
                     image_shape=list(array.shape),image_sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest()))


def child(config_path, arm, index):
    values = json.loads(Path(config_path).read_text())
    args = argparse.Namespace(**values)
    for name in ('baseline_root','candidate_root','policy','codec_deps','output'):
        setattr(args,name,Path(getattr(args,name)))
    args.source_root = args.baseline_root if arm == 'baseline' else args.candidate_root
    run_dir=args.output/f'run-{index+1:02d}-{arm}'
    run_dir.mkdir()
    sys.path.insert(0,str(args.source_root))
    sys.path.insert(0,str(args.codec_deps))
    handles=[os.add_dll_directory(str(path)) for path in sorted({p.parent for p in args.codec_deps.rglob('*.dll')})] if os.name=='nt' else []
    for name,folder in (('TEMP','temp'),('TMP','temp'),('CUPY_CACHE_DIR','cupy-cache'),('CUDA_CACHE_PATH','cuda-cache')):
        path=args.output/folder;path.mkdir(exist_ok=True);os.environ[name]=str(path)
    os.environ.update(PTA_GPU_TORCH_COMPILE='0',PTA_GPU_PUBLICATION_PIPELINE='1',PTA_GPU_PUBLICATION_REPORT_SEC='0',
        YOLO_TTA_GPU_SPHERICAL_FP32='0',YOLO_TTA_PTA_GPU_CATEGORICAL='1',
        YOLO_TTA_PTA_SPHERICAL_HOST_CACHE_MIB=str(args.host_mib if arm=='candidate' else 0))
    from tools.gpu_reservation import gpu_reservation
    retired=False
    report=dict(status='running',arm=arm,source_root=str(args.source_root),policy_sha256=digest(args.policy),
                source_sha256={name:digest(args.source_root/name) for name in values['source_files'] if (args.source_root/name).is_file()})
    save(run_dir/'metrics.json',report)
    try:
        with gpu_reservation(SCRATCH/'Temp/GPU_LOCK',timeout=0,task_name='spherical-host-cache-validation',release_when=lambda:retired) as claim:
            report['gpu_claim']=claim
            import torch
            from XTA import pta_workers as workers, spherical_cuda as sc, pta_cuda_shells as shells
            from XTA.pta_augmentation import load_gpu_augmentation_definition
            from XTA.pta_gpu_publication import publication_resources
            from XTA.pta_cuda_masks import retire_gpu_categorical_volume
            from tools.qualify_pta_gpu_masks import heatsoak
            if Path(workers.__file__).resolve()!=args.source_root/'XTA/pta_workers.py':
                raise RuntimeError('Wrong worker source imported')
            if sc._DIRECTION_CACHE_BYTES!=256*1024**2 or shells._SHELL_DIRECTION_CACHE_BYTES!=256*1024**2:
                raise RuntimeError('GPU direction-cache budgets changed')
            if report['policy_sha256']!=POLICY_SHA:
                raise RuntimeError('GPU_sweep bytes differ from the authorized actual-run policy')
            workers.cv2.setNumThreads(1);torch.set_num_threads(1);torch.cuda.set_device(args.device)
            report['hardware']=dict(device=torch.cuda.get_device_name(args.device),torch=torch.__version__,cuda=torch.version.cuda)
            augmentation=load_gpu_augmentation_definition(str(args.policy))
            data=make_work(args,augmentation,run_dir)
            report['workload']=data[5]
            with (run_dir/'worker.log').open('w',encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
                heatsoak(torch,f'cuda:{args.device}',args.heatsoak_seconds)
                workers._WORKER_STATIC=dict(augmentation=augmentation,gpu_batch_size=args.batch_size,gpu_render_threads=1,
                    image_format='jpg',jpeg_encode_backend='nvjpeg',tiff_encode_backend='opencv',out_dir=run_dir,
                    split_active=False,save_images=True,save_labels=True,save_binary=False,save_semantic=args.coverage,
                    jpeg_quality=95,png_compression=1)
                workers._WORKER_GPU_RUNTIME=None;workers._WORKER_GPU_DEVICE_ID=args.device
                runtime=workers._gpu_runtime_for_worker()
                try:
                    if runtime.get('encoder') is None or runtime.get('codec_error'):
                        raise RuntimeError('Native nvJPEG required')
                    resources=publication_resources(runtime,cpu_threads=1)
                    workers._prepare_gpu_volume_task(runtime,identity=('warmup',),generation=0)
                    warm=(*data[:4],data[6],data[5])
                    run_work(args,runtime,warm,run_dir/'warmup')
                    workers._prepare_gpu_volume_task(runtime,identity=('classification-proof',),generation=1)
                    proof=classification_admission_case(args,runtime,data)
                    report['classification_case']=proof
                    report['classification_outputs']=proof['outputs']
                    if arm=='candidate' and not args.qualification_only and proof['intensity_host_build_block_calls']!=0:
                        raise AssertionError('Candidate rebuilt classification-warmed host geometry during source admission')
                    if arm=='baseline' and proof['intensity_host_build_block_calls']==0:
                        raise AssertionError('Baseline did not exercise duplicate first-admission geometry construction')
                    workers._prepare_gpu_volume_task(runtime,identity=('measured',),generation=2)
                    report['host_before']=host_snapshot(runtime)
                    validate_host_snapshot(report['host_before'],cleared=True)
                    before=shells.shell_direction_cache_stats(runtime['azimuthal_renderer'])
                    resources.timings.clear();resources.counts.clear()
                    torch.cuda.reset_peak_memory_stats(args.device)
                    with RssSampler() as rss:
                        measured=run_work(args,runtime,data,run_dir/'outputs')
                    report.update(measured)
                    report.update(peak_rss_bytes=rss.peak,initial_rss_bytes=rss.start,
                        peak_device_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                        peak_device_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                        cache_before=before,cache_after=shells.shell_direction_cache_stats(runtime['azimuthal_renderer']),
                        host_after=host_snapshot(runtime),publication_times=dict(resources.timings),
                        publication_counts=dict(resources.counts),stage_times_overlap=True)
                    report['cache_delta']=counter_delta(before,report['cache_after'])
                    report['host_delta']=counter_delta(report['host_before'] or {},report['host_after'] or {})
                    validate_host_snapshot(report['host_after'])
                    report['gpu_cache_bytes']=dict(intensity=runtime['azimuthal_renderer']._spherical_direction_cache_bytes,categorical=shells._SHELL_DIRECTION_BYTES)
                    if any(size>256*1024**2 for size in report['gpu_cache_bytes'].values()):
                        raise AssertionError('GPU direction cache exceeded unchanged budget')
                    if arm=='candidate' and report['host_after'] is None:
                        raise RuntimeError('Candidate host cache was not installed')
                    if arm=='candidate' and not args.qualification_only and (
                            report['host_delta'].get('hits',0)<=0 or report['host_delta'].get('builds',0)<=0):
                        raise AssertionError('Timed candidate must include compulsory builds and actual memo hits')
                    if index<2:
                        qualified=run_work(args,runtime,data,run_dir/'qualification',capture_raw=True)
                        report['raw_render']=qualified['raw_render'];report['raw_policy']=qualified['raw_policy']
                        report['qualification_partitions']=qualified['policy_partitions']
                        if any(not value for value in compare_results(measured,qualified).values()):
                            raise AssertionError('Instrumented qualification changed measured outputs')
                    import psutil
                    loaded={Path(item.path).resolve() for item in psutil.Process().memory_maps()
                            if item.path and Path(item.path).is_file()}
                    report['codec']=dict(version=str(runtime['nvimgcodec'].__version__),module=str(runtime['nvimgcodec'].__file__),
                        module_sha256=digest(runtime['nvimgcodec'].__file__),loaded_dependency_sha256={str(path):digest(path) for path in sorted(loaded) if path.is_relative_to(args.codec_deps)})
                    report['policy_pointwise_compiled']=bool(getattr(runtime['policy'],'_pointwise_compiled',False))
                    report['environment']={name:os.environ.get(name) for name in ('PTA_GPU_TORCH_COMPILE','YOLO_TTA_GPU_SPHERICAL_FP32','YOLO_TTA_PTA_SPHERICAL_HOST_CACHE_MIB')}
                    workers._prepare_gpu_volume_task(runtime,identity=('retired',),generation=3)
                    report['host_after_generation_retirement']=host_snapshot(runtime)
                    validate_host_snapshot(report['host_after_generation_retirement'],cleared=True)
                finally:
                    resources=runtime.get('publication_resources')
                    if resources is not None:
                        resources.close();resources.finalizer.cancel()
                        if resources.poisoned:raise RuntimeError('Publication was poisoned')
                    retire_gpu_categorical_volume(runtime)
                    runtime['azimuthal_renderer'].retire_volume_array()
                    encoder=runtime.pop('encoder',None)
                    if callable(getattr(encoder,'close',None)):encoder.close()
                    del encoder
                    torch.cuda.synchronize(args.device)
                    runtime.clear();workers._WORKER_GPU_RUNTIME=None
                    gc.collect();torch.cuda.empty_cache()
                torch.cuda.synchronize(args.device)
                retired=True
            report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',retired=retired)
        raise
    finally:
        save(run_dir/'metrics.json',report)
    print(f"{arm}: {report['wall_seconds']:.3f}s, {report['totals']['written']} outputs, cache={report['cache_delta']}",flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-config',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--arm',choices=('baseline','candidate'),help=argparse.SUPPRESS)
    parser.add_argument('--index',type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--candidate-root',type=Path,default=ROOT)
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--codec-deps',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cube-edge',type=int,default=768)
    parser.add_argument('--imgsz',type=int,default=2048)
    parser.add_argument('--tile-size',type=int,default=1024)
    parser.add_argument('--keys',type=int,default=8)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--ratio',type=float,default=3.7)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--host-mib',type=int,default=3072)
    parser.add_argument('--heatsoak-seconds',type=float,default=45.)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--coverage',action='store_true')
    parser.add_argument('--qualification-only',action='store_true',help='One old/new pair, no speedup claim; permits 0/low host cap and zero heatsoak')
    args=parser.parse_args(argv)
    if args.worker_config:
        if args.arm is None or args.index<0:parser.error('Worker requires its arm and nonnegative index')
        return child(args.worker_config,args.arm,args.index)
    if args.baseline_root is None or args.codec_deps is None or args.output is None:
        parser.error('--baseline-root, --codec-deps and --output are required')
    for name in ('baseline_root','candidate_root','codec_deps','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.policy=(args.policy or args.baseline_root/'XTA/examples/external_augmentations/GPU_sweep.py').resolve()
    if args.output.exists() or not args.output.is_relative_to(SCRATCH.resolve()) or args.output==SCRATCH.resolve():
        parser.error('Output must be a fresh task directory under sibling Scratch')
    if not(6<=args.keys<=30 and 2<=args.rounds<=16 and args.repetitions>=1 and 1<=args.batch_size<=32
           and 1<args.ratio<=28 and 0<=args.heatsoak_seconds<=300 and args.host_mib>=0):
        parser.error('Invalid workload, cache or measurement controls')
    if not args.qualification_only and (args.heatsoak_seconds<=0 or args.host_mib<=0):
        parser.error('Timed comparisons require positive heatsoak and an enabled host cache')
    if args.imgsz!=2048 or args.tile_size!=1024 or not(768<=args.cube_edge<=1024):
        parser.error('This controlled fixture requires 2048 output, 1024 tiles and a 768..1024 cube')
    args.output.mkdir(parents=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config['source_files']=['XTA/spherical_cuda.py','XTA/pta_cuda_shells.py','XTA/pta_workers.py',
        'XTA/spherical_host_cache.py','XTA/cuda_backend.py','XTA/pta_gpu_publication.py','XTA/pta_scheduler.py']
    save(args.output/'config.json',config)
    report=dict(status='running',scope=__doc__,config=config,tool_sha256=digest(__file__),
                helper_sha256=digest(ROOT/'tools/benchmark_pta_task_envelope.py'),runs=[])
    save(args.output/'report.json',report)
    try:
        order=('baseline','candidate') if args.qualification_only else shared_helpers().abba_order(args.repetitions)
        for index,arm in enumerate(order):
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker-config',str(args.output/'config.json'),
                            '--arm',arm,'--index',str(index)],check=True)
            result=json.loads((args.output/f'run-{index+1:02d}-{arm}'/'metrics.json').read_text())
            result['comparison']=compare_results(report['runs'][0],result) if report['runs'] else {}
            report['runs'].append(result)
            save(args.output/'report.json',report)
            if not all(result['comparison'].values()):raise AssertionError('Old/new source exact parity failed')
        if not args.qualification_only:
            report['median_wall_seconds']={arm:statistics.median(run['wall_seconds'] for run in report['runs'] if run['arm']==arm) for arm in ('baseline','candidate')}
            report['wall_reduction_fraction']=1-report['median_wall_seconds']['candidate']/report['median_wall_seconds']['baseline']
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save(args.output/'report.json',report)


if __name__=='__main__':
    main()
