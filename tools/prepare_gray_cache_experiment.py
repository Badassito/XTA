"""Prepare separate consecutive real gray8 view samples for offline codec experiments.

CPU only. The full native input is a temporary disk map, with no processing-cube
copy. Samples describe native geometry rather than the production T-upsampled cube.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

GIB = 1024**3
SIZE = 2048
FRAMES = 16
SAMPLE_COUNT = 8


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def consecutive_window(count, fraction, length=FRAMES):
    if count < length or not 0 < fraction < 1:
        raise ValueError('Sample requires a complete consecutive window and an interior fraction')
    center = int(round(fraction*(count-1)))
    start = max(0, min(count-length, center-length//2))
    return tuple(range(start, start+length))


def sample_specs(shape, size=SIZE):
    from XTA import geometry
    from XTA.config import AzimuthalViewRequest, resolve_tilted_view_groups
    from XTA.media import resolve_azimuthal_azimuth_angles
    azimuthal_targets = ('transverse', 'tilted_transverse')
    azimuthal_angles = resolve_azimuthal_azimuth_angles(
        [AzimuthalViewRequest(view=target, azimuth_angle=None) for target in azimuthal_targets],
        diameters=[geometry.azimuthal_target_diameter(target, *shape) for target in azimuthal_targets])
    views = geometry.get_view_infos(*shape, cartesian_views=(),
        tilt_groups=resolve_tilted_view_groups(['transverse:30:both']),
        radial_views=('transverse', 'tilted_transverse'), radial_patch_size=size,
        spherical_views=('transverse', 'tilted_transverse'), spherical_patch_size=size,
        azimuthal_views=azimuthal_targets, azimuthal_azimuth_angles=azimuthal_angles,
        sampling_policy='coverage')

    def choose(family, predicate):
        return next(view for view in views if view.family == family and predicate(view))

    positive_vertical = lambda view: view.tilt_angle_deg == 30 and view.tilt_direction == 'vertical'
    selections = (
        ('spherical-upright', choose('spherical', lambda view:
            not view.spherical_tilted_source and view.spherical_face == 0
            and view.spherical_patch_u == view.spherical_patch_v == 0), .35),
        ('spherical-tilted', choose('spherical', lambda view:
            positive_vertical(view) and view.spherical_face == 4
            and view.spherical_patch_u == view.spherical_patch_v == 0), .70),
        ('radial-upright', choose('radial', lambda view:
            not view.radial_tilted_source and view.radial_patch_index == 1
            and view.radial_height_index == 0), .35),
        ('radial-tilted', choose('radial', lambda view:
            positive_vertical(view) and view.radial_patch_index == 1
            and view.radial_height_index == 0), .70),
        ('azimuthal-upright', choose('azimuthal', lambda view:
            not view.azimuthal_tilted_source), .35),
        ('azimuthal-tilted', choose('azimuthal', positive_vertical), .70),
        ('tilted-cartesian-vertical', choose('tilted', positive_vertical), .35),
        ('tilted-cartesian-horizontal', choose('tilted', lambda view:
            view.tilt_angle_deg == 30 and view.tilt_direction == 'horizontal'), .70),
    )
    return tuple((name, view, fraction, consecutive_window(view.num_slices, fraction))
                 for name, view, fraction in selections)


def histogram_stats(histogram):
    histogram = np.asarray(histogram, np.int64)
    if histogram.shape != (256,) or np.any(histogram < 0) or not histogram.sum():
        raise ValueError('Expected a nonempty gray8 histogram')
    values = np.arange(256, dtype=np.float64)
    count = int(histogram.sum())
    mean = float(values@histogram/count)
    variance = float((values*values)@histogram/count-mean*mean)
    return dict(pixels=count, histogram=histogram.tolist(),
        zero_fraction=float(histogram[0]/count), full_fraction=float(histogram[255]/count),
        mean=mean, std=math.sqrt(max(0., variance)),
        zero_fraction_is_intensity_zero_not_a_segmented_air_measure=True)


def owned_task_path(path, task_root):
    task_root = Path(task_root).resolve(strict=True)
    resolved = Path(path).resolve(strict=True)
    if resolved.parent != task_root or not resolved.name.startswith('native-source-'):
        raise RuntimeError('Refusing to retire a file outside this task or with another role')
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source_path = args.input.resolve(strict=True)
    scratch = (ROOT.parent/'Scratch'/'Experiments').resolve(strict=True)
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(scratch) or destination == scratch:
        parser.error('Write generated samples to a task directory beneath Scratch/Experiments')
    destination.mkdir(parents=True, exist_ok=False)

    import psutil
    from XTA import geometry
    from XTA._deps import cv2
    from XTA.media import decode_video_to_memmap_gray8, ffprobe_info, wait_for_volume_ready
    from XTA.runtime import close_memmap_array_without_flush

    info = ffprobe_info(source_path)
    shape = (int(info['num_frames']), int(info['height']), int(info['width']))
    native_bytes = math.prod(shape)
    corpus_bytes = SAMPLE_COUNT*FRAMES*SIZE*SIZE
    if (not 1 <= native_bytes <= 20*GIB or min(shape) < FRAMES
            or shutil.disk_usage(destination).free < native_bytes+corpus_bytes+8*GIB
            or psutil.virtual_memory().available < 2*GIB):
        raise RuntimeError('Native decode, bounded samples and RAM/disk reserves do not fit')
    specs = sample_specs(shape)
    source_stat = source_path.stat()
    source_hash = file_sha256(source_path)
    native_path = destination/f'native-source-{uuid.uuid4().hex}.gray8.dat'
    if native_path.exists():
        raise FileExistsError(native_path)
    source = None
    report = dict(schema='xta.offline_gray_cache_samples/1', input=dict(path=str(source_path),
        bytes=source_stat.st_size, mtime_ns=source_stat.st_mtime_ns, sha256=source_hash,
        metadata=info), source_native_shape_tyx=list(shape),
        target_processing_shape_tyx=[2911, 3064, 3022],
        geometry_limit='Native full F3 geometry; no production T interpolation to2911. Not a replay of a recorded detector job.',
        sample_shape_tyx=[FRAMES, SIZE, SIZE], samples=[], cpu_only=True,
        requested_sampling_policy='coverage',
        disjoint_windows_are_separate_files=True,
        implementations={name: file_sha256(ROOT/'XTA'/name) for name in
            ('geometry.py', 'media.py', 'cylindrical_geometry.py', 'spherical_geometry.py',
             'sam_canvas_rendering.py', 'sam_integration.py', 'sam_transverse_cache_rendering.py')},
        sample_harness_sha256=file_sha256(__file__))
    report_path = destination/'manifest.json'
    thumbnails = []
    try:
        print(json.dumps(dict(stage='decode', native_shape=shape, native_bytes=native_bytes,
            disk_free=shutil.disk_usage(destination).free, source_sha256=source_hash)), flush=True)
        begin = time.perf_counter()
        source = decode_video_to_memmap_gray8(source_path, native_path, shape[0], shape[2], shape[1],
            overwrite=False, prefer_memory=False, prefer_memfd=False, strict_frame_count=True)
        wait_for_volume_ready(source)
        if not isinstance(source, np.memmap) or source.shape != shape or source.dtype != np.uint8:
            raise RuntimeError('Expected the completed native disk-backed gray8 volume')
        report['native_decode_seconds'] = time.perf_counter()-begin
        report['native_decoded_sha256'] = file_sha256(native_path)
        print(json.dumps(dict(stage='decoded', seconds=report['native_decode_seconds'])), flush=True)

        for name, physical_view, fraction, frames in specs:
            view = geometry.expand_views_into_tta_variants((physical_view,), (0.,))[0]
            job = geometry.build_aug_job_for_variant(view, SIZE, destination)
            path = destination/(name+'.npy')
            target = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(FRAMES, SIZE, SIZE))
            timings = dict(render_seconds=0., write_seconds=0., histogram_seconds=0., flush_seconds=0.)
            histogram = np.zeros(256, np.int64)
            per_frame = []
            started = time.perf_counter()
            image = None
            try:
                # The decoder warms CPU. One untimed same-view frame also establishes
                # geometry plans and page locality before each sixteen-frame sample.
                warm_started = time.perf_counter()
                image = geometry.render_intensity_frame_on_grid(source, view, frames[0],
                    M_src_to_out=job.aff.M_src_to_out, M_out_to_src=job.aff.M_out_to_src,
                    output_height=SIZE, output_width=SIZE)
                warm_seconds = time.perf_counter()-warm_started
                image = None
                for position, frame in enumerate(frames):
                    began = time.perf_counter()
                    image = geometry.render_intensity_frame_on_grid(source, view, frame,
                        M_src_to_out=job.aff.M_src_to_out, M_out_to_src=job.aff.M_out_to_src,
                        output_height=SIZE, output_width=SIZE)
                    elapsed = time.perf_counter()-began
                    timings['render_seconds'] += elapsed
                    if image.shape != (SIZE, SIZE) or image.dtype != np.uint8:
                        raise RuntimeError('Canonical renderer returned a mismatched gray8 sample')
                    began = time.perf_counter()
                    target[position] = image
                    timings['write_seconds'] += time.perf_counter()-began
                    began = time.perf_counter()
                    local_histogram = np.zeros(256, np.int64)
                    for row in range(0, SIZE, 128):
                        local_histogram += np.bincount(image[row:row+128].ravel(), minlength=256)
                    histogram += local_histogram
                    timings['histogram_seconds'] += time.perf_counter()-began
                    per_frame.append(dict(frame_index=frame, render_seconds=elapsed,
                        zero_fraction=float(local_histogram[0]/image.size),
                        full_fraction=float(local_histogram[255]/image.size)))
                    if position == FRAMES//2:
                        thumbnails.append((name, cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)))
                    image = None
                began = time.perf_counter()
                target.flush()
                timings['flush_seconds'] += time.perf_counter()-began
                entry = dict(name=name, path=str(path), family=view.family, sample_role='primary', shape=[FRAMES, SIZE, SIZE],
                    dtype='uint8', view=asdict(view), source_native_shape_tyx=list(shape),
                    source_sha256=source_hash, requested_fraction=fraction, frame_indices=list(frames),
                    frame_axis=('radius' if view.family in {'spherical', 'radial'} else
                        'azimuth' if view.family == 'azimuthal' else 'native_tilted_stack'),
                    affine_native_to_canvas=job.aff.M_src_to_out.tolist(),
                    affine_canvas_to_native=job.aff.M_out_to_src.tolist(),
                    warmup_render_seconds=warm_seconds, timings=timings,
                    preparation_seconds=time.perf_counter()-started,
                    intensity=histogram_stats(histogram), per_frame=per_frame)
                if view.family in {'spherical', 'radial'}:
                    entry['sam_crop'] = sample_sam_crops(source, view, frames, target, job, source_hash)
            finally:
                image = None
                close_memmap_array_without_flush(target)
                target = None
            entry['bytes'] = path.stat().st_size
            entry['sha256'] = file_sha256(path)
            report['samples'].append(entry)
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(dict(stage='sample', name=name, family=view.family,
                frames=list(frames), timings=timings, zero_fraction=entry['intensity']['zero_fraction'],
                std=entry['intensity']['std'], sam_crop=entry.get('sam_crop'))), flush=True)

        from PIL import Image, ImageDraw
        preview = Image.new('L', (4*256, 2*288), color=0)
        draw = ImageDraw.Draw(preview)
        for index, (name, thumbnail) in enumerate(thumbnails):
            x, y = index%4*256, index//4*288
            preview.paste(Image.fromarray(thumbnail), (x, y+24))
            draw.text((x+4, y+4), name, fill=255)
        preview.save(destination/'contact-preview.png')
        report['status'] = 'complete'
        report['total_sample_bytes'] = sum(entry['bytes'] for entry in report['samples'])
        report['limitations'] = ['Local CPU render costs include the native map/page locality.',
            'Native source T differs from the2911 working cube; no T-resize or GPU/SDK execution.',
            'Zero intensity fraction is an air/padding bias indicator, not a segmentation result.']
    finally:
        # Only this invocation's UUID-named native file can be removed, and only
        # after its owning map is retired. Retain every completed sample/evidence.
        close_memmap_array_without_flush(source)
        source = None
        if native_path.exists():
            owned_task_path(native_path, destination).unlink()
        report['native_temporary_retired'] = not native_path.exists()
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    (destination/'samples.json').write_text(json.dumps(report['samples'], indent=2), encoding='utf-8')
    print(json.dumps(dict(stage='done', samples=len(report['samples']),
        total_sample_bytes=report.get('total_sample_bytes'), native_temporary_retired=True,
        disk_free_bytes=shutil.disk_usage(destination).free)), flush=True)


def sample_sam_crops(source, view, frames, full_frames, job, source_identity):
    """Measure the existing SAM CPU crop route, separately from codec timing."""
    from XTA.sam_integration import SamInterpolationContext
    from XTA.runtime import runtime_telemetry
    context = SamInterpolationContext(model_path='unused-offline-cpu', device_ids=(0,),
        temp_dir=ROOT.parent/'Scratch'/'Temp', evidence_root=ROOT.parent/'Scratch'/'Temp',
        source_volume=source, source_identity=source_identity)
    box = (704, 624, 1344, 1424)
    affine, inverse, _ = context._canvas_transform(view, (view.num_slices, SIZE, SIZE))
    if not (np.array_equal(affine, job.aff.M_src_to_out) and np.array_equal(inverse, job.aff.M_out_to_src)):
        context.close()
        return dict(status='different_canonical_affines_not_compared')
    telemetry = runtime_telemetry()
    before = telemetry.snapshot().get('counters', {})
    maximum = changed = delta_sum = pixels = 0
    seconds = 0.
    warm_seconds = 0.
    try:
        began = time.perf_counter()
        warm = context._render_demand_crop(view, frames[0], affine, inverse,
            output_height=640, output_width=800, output_origin_yx=box[:2], output_canvas_width=SIZE)
        warm_seconds = time.perf_counter()-began
        warm = None
        warm_native_pixels = context.native_sampling_pixels
        for position, frame in enumerate(frames):
            began = time.perf_counter()
            crop = context._render_demand_crop(view, frame, affine, inverse,
                output_height=640, output_width=800, output_origin_yx=box[:2], output_canvas_width=SIZE)
            seconds += time.perf_counter()-began
            delta = np.abs(crop.astype(np.int16)-full_frames[position, box[0]:box[2], box[1]:box[3]].astype(np.int16))
            maximum = max(maximum, int(delta.max()))
            changed += int(np.count_nonzero(delta))
            delta_sum += int(delta.sum())
            pixels += int(delta.size)
            crop = delta = None
        after = telemetry.snapshot().get('counters', {})
        return dict(status='measured', method='SamInterpolationContext._render_demand_crop; serial CPU source ROI +canonical remap',
            bbox_yx=list(box), frame_count=len(frames), warmup_seconds=warm_seconds,
            render_seconds=seconds, max_absolute_delta=maximum, mean_absolute_delta=delta_sum/max(1, pixels),
            changed_pixels=changed, compared_pixels=pixels, within_one_gray=maximum <= 1,
            native_sampled_pixels=context.native_sampling_pixels-warm_native_pixels,
            telemetry_includes_warmup={key: value-before.get(key, 0) for key, value in after.items()
                if key.startswith('sam.cpu_images.')})
    finally:
        context.close()


if __name__ == '__main__':
    main()
