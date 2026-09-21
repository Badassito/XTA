"""Align source video to saved reconciliation comparisons and render review overlays."""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from XTA.media import _linear_source_index
from XTA.reconciliation_io import _read_header, _vectors

COLORS = {'retained': (45, 190, 135), 'rejected': (240, 85, 85)}


def sampling_plan(source_shape, target_shape, selected):
    source_t, _, _ = source_shape
    target_t, _, _ = target_shape
    plan = []
    for z in range(target_t):
        position = _linear_source_index(z, target_t, source_t)
        lo = int(math.floor(position))
        hi = min(source_t - 1, lo + 1)
        plan.append((lo, hi, position - lo))
    footprints = {z: (int(math.floor(z * source_t / target_t)),
                      int(math.ceil((z + 1) * source_t / target_t))) for z in selected}
    return plan, footprints


def resize_stream(frames, source_shape, output, selected=(), progress=None):
    """Match low-quality grayscale rendering with at most adjacent resized frames."""
    source_t, source_h, source_w = map(int, source_shape)
    target_t, target_h, target_w = output.shape
    plan, footprints = sampling_plan(source_shape, output.shape, selected)
    completed_at = defaultdict(list)
    references = Counter()
    for z, (lo, hi, alpha) in enumerate(plan):
        completed_at[hi].append((z, lo, hi, alpha))
        references.update({lo, hi})
    windows = defaultdict(list)
    envelopes = {z: np.zeros((target_h, target_w), np.uint8) for z in selected}
    for z, (start, stop) in footprints.items():
        for index in range(start, stop):
            windows[index].append(z)
    cached = {}
    count = 0
    interpolation = cv2.INTER_AREA if target_h <= source_h and target_w <= source_w else cv2.INTER_LINEAR
    for index, frame in enumerate(frames):
        if index >= source_t:
            raise ValueError('Video contains more frames than the run source grid')
        if frame.shape != (source_h, source_w) or frame.dtype != np.uint8:
            raise ValueError('Decoded source frame differs from the run source grid or uint8 type')
        if references[index] or windows[index]:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
            for z in windows[index]:
                np.maximum(envelopes[z], resized, out=envelopes[z])
            if references[index]:
                cached[index] = resized
        for z, lo, hi, alpha in completed_at[index]:
            if lo == hi or alpha <= 1e-7:
                output[z] = cached[lo]
            else:
                output[z] = np.clip(np.rint((1. - alpha) * cached[lo].astype(np.float32)
                    + alpha * cached[hi].astype(np.float32)), 0, 255).astype(np.uint8)
            for key in {lo, hi}:
                references[key] -= 1
                if not references[key]:
                    del cached[key]
        count += 1
        if progress and (count % 100 == 0 or count == source_t):
            progress(count, source_t)
    if count != source_t:
        raise ValueError(f'Video has {count} frames; run source grid requires {source_t}')
    return envelopes


def video_frames(path, shape, log_path):
    """Decode gray8 on CPU, keeping one full-resolution frame in memory."""
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '8', '-i', str(path),
               '-map', '0:v:0', '-an', '-sn', '-pix_fmt', 'gray', '-fps_mode', 'passthrough',
               '-f', 'rawvideo', 'pipe:1']
    size = int(shape[1]) * int(shape[2])
    buffer = bytearray(size)
    with log_path.open('wb') as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            while True:
                offset = 0
                while offset < size:
                    read = process.stdout.readinto(memoryview(buffer)[offset:])
                    if not read:
                        break
                    offset += read
                if not offset:
                    break
                if offset != size:
                    raise ValueError('Decoder returned an incomplete source frame')
                yield np.frombuffer(buffer, np.uint8).reshape(shape[1:])
            if process.wait() != 0:
                raise RuntimeError(f'Video decode failed; see {log_path}')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()


def selected_masks(path, shape, selected, geometry):
    header, offset = _read_header(path)
    if tuple(map(int, header['sizes'].split())) != tuple(reversed(shape)):
        raise ValueError(f'Comparison mask must use the complete target grid: {path}')
    if header.get('type') not in {'uchar', 'uint8', 'unsigned char', 'uint8_t'}:
        raise ValueError(f'Expected a uint8 comparison mask: {path}')
    if header.get('space') != geometry['space']:
        raise ValueError(f'Comparison space differs from report: {path}')
    if header.get('encoding') not in {'gzip', 'gz'}:
        raise ValueError('Comparison masks must use gzip encoding')
    if (not np.allclose(_vectors(header['space directions'], 3, 'space directions'), geometry['directions_xyz'])
            or not np.allclose(_vectors(header['space origin'], 1, 'space origin')[0], geometry['origin_xyz'])):
        raise ValueError(f'Comparison geometry differs from report: {path}')
    # These comparisons use the canonical compact display grid.
    if (not np.array_equal(geometry['directions_xyz'], np.eye(3))
            or not np.array_equal(geometry['origin_xyz'], [0, 0, 0])):
        raise ValueError('Overlay currently requires an identity compact display grid')
    result = {}
    with path.open('rb') as raw:
        raw.seek(offset)
        with gzip.GzipFile(fileobj=raw) as stream:
            for z0 in range(0, shape[0], 8):
                z1 = min(shape[0], z0 + 8)
                size = (z1 - z0) * shape[1] * shape[2]
                data = stream.read(size)
                if len(data) != size:
                    raise ValueError(f'Incomplete comparison mask: {path}')
                slab = np.frombuffer(data, np.uint8).reshape(z1 - z0, *shape[1:])
                if np.any(slab > 1):
                    raise ValueError(f'Comparison mask is not binary: {path}')
                for z in selected:
                    if z0 <= z < z1:
                        result[z] = slab[z - z0].copy()
            if stream.read(1):
                raise ValueError(f'Comparison mask exceeds its grid: {path}')
    return result


def write_gray_nrrd(path, volume):
    header = ('NRRD0005\ntype: uint8\ndimension: 3\nsizes: '
              + ' '.join(map(str, reversed(volume.shape)))
              + '\nspace: left-posterior-superior\nspace directions: (1,0,0) (0,1,0) (0,0,1)'
              + '\nspace origin: (0,0,0)\nkinds: domain domain domain\nencoding: gzip\n\n')
    with path.open('wb') as raw:
        raw.write(header.encode('ascii'))
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6, mtime=0) as stream:
            for z in range(volume.shape[0]):
                stream.write(memoryview(np.ascontiguousarray(volume[z])).cast('B'))


def overlay(source, candidate, retained, alpha=.38):
    if np.any((retained != 0) & (candidate == 0)):
        raise ValueError('Retained mask contains foreground outside the comparison union')
    rgb = np.repeat(source[..., None], 3, axis=2)
    for mask, color in ((retained != 0, COLORS['retained']),
                        ((candidate != 0) & (retained == 0), COLORS['rejected'])):
        rgb[mask] = np.rint((1 - alpha) * rgb[mask] + alpha * np.array(color)).astype(np.uint8)
    return rgb


def png_data(array):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def render_dataset(output_dir, dataset, gray, envelopes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    name = dataset['dataset']
    directory = output_dir / name
    directory.mkdir()
    slices = dataset['previews']['slices']
    selected = [item['z'] for item in slices]
    methods = {m['policy_name']: m for m in dataset['methods'] if m['status'] == 'complete'}
    masks = {method: selected_masks(Path(record['output']), tuple(gray.shape), selected,
                                   dataset['reference_geometry']) for method, record in methods.items()}
    candidate = masks['union']
    records = []
    for item in slices:
        z = item['z']
        expected = item['union_slice_voxels']
        if np.count_nonzero(candidate[z]) != expected:
            raise ValueError('Comparison report and union slice counts differ')
        record = dict(dataset=name, z=z, roi=item['roi_y0_y1_x0_x1'],
                      source=png_data(gray[z]), slab=png_data(envelopes[z]), layers={})
        for method in ('cross_sections', 'largest_island'):
            retained = masks[method][z]
            rendered = overlay(gray[z], candidate[z], retained)
            Image.fromarray(rendered).save(directory / f'z{z}_{method}_full.png')
            rgba = np.zeros((*retained.shape, 4), np.uint8)
            rgba[retained != 0] = (*COLORS['retained'], 255)
            rgba[(candidate[z] != 0) & (retained == 0)] = (*COLORS['rejected'], 255)
            record['layers'][method] = png_data(rgba)
        records.append(record)
        y0, y1, x0, x1 = item['roi_y0_y1_x0_x1']
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='#151920')
        panels = [('Source', np.repeat(gray[z, ..., None], 3, axis=2))]
        panels += [(method, overlay(gray[z], candidate[z], masks[method][z]))
                   for method in ('cross_sections', 'largest_island')]
        for axis, (title, image) in zip(axes, panels):
            axis.imshow(image[y0:y1, x0:x1], origin='upper', interpolation='nearest')
            axis.set_axis_off()
            axis.set_title(title.replace('_', ' '), color='white', fontsize=14)
        fig.suptitle(f'{name} | z={z} | source with reconciliation overlay', color='white', fontsize=15)
        fig.legend(handles=[Patch(color=np.array(COLORS[key])/255, label=key.title())
                            for key in COLORS], loc='lower center', ncol=2, facecolor='#151920', labelcolor='white')
        fig.tight_layout(rect=(0, .055, 1, .93))
        fig.savefig(directory / f'z{z}_comparison.png', dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    return records


def write_viewer(path, records):
    payload = json.dumps(records).replace('</', '<\\/')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Reconciliation source overlays</title>
<style>body{margin:24px;background:#151920;color:#eee;font:16px system-ui}h1{font-size:24px}label{margin-right:20px;display:inline-block;margin-bottom:12px}select,button{font:inherit;background:#29303c;color:white;padding:6px;border:1px solid #687285;border-radius:5px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}canvas{width:100%;image-rendering:pixelated}h2{font-size:17px}.note{color:#bdc7d7;max-width:1100px;line-height:1.5}.kept{color:#2dbe87}.rejected{color:#f05555}</style>
<h1>Source overlays · reconciliation review</h1>
<label>Run / slice <select id="sample"></select></label>
<label>Overlay opacity <input id="alpha" type="range" min="0" max="100" value="38"> <output id="amount">38%</output></label>
<label><input id="crop" type="checkbox" checked>Crop to comparison bounds</label>
<label><input id="retained" type="checkbox" checked><span class="kept">Retained</span></label>
<label><input id="rejected" type="checkbox" checked><span class="rejected">Rejected</span></label>
<label>Source <select id="background"><option value="source">Pipeline grayscale slice</option><option value="slab">Maximum over mask's source-frame footprint</option></select></label>
<div class="grid"><section><h2>Source</h2><canvas id="original"></canvas></section><section><h2>Cross sections</h2><canvas id="cross_sections"></canvas></section><section><h2>Largest island</h2><canvas id="largest_island"></canvas></section></div>
<p class="note" id="position"></p>
<p class="note">The grayscale background follows the pipeline's area resize and endpoint-aligned temporal interpolation. Compact masks pool several source frames. The optional footprint maximum shows that broader source support. No policy decisions or source masks were changed. Opacity 0 shows the source alone.</p>
<script>const records=PAYLOAD, byId=id=>document.getElementById(id), cache=new Map();
function picture(url){if(!cache.has(url)){let im=new Image();im.src=url;cache.set(url,new Promise(resolve=>im.onload=()=>resolve(im)));}return cache.get(url);}
records.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.dataset.slice(-6)+' · z='+r.z;byId('sample').append(o);});
let sequence=0;
async function render(){let token=++sequence,r=records[+byId('sample').value],a=+byId('alpha').value/100;byId('amount').textContent=Math.round(a*100)+'%';
let bg=await picture(r[byId('background').value]), images={};for(let method of ['cross_sections','largest_island'])images[method]=await picture(r.layers[method]);if(token!==sequence)return;
let [y0,y1,x0,x1]=byId('crop').checked?r.roi:[0,bg.height,0,bg.width];
for(let method of ['original','cross_sections','largest_island']){let c=byId(method);c.width=x1-x0;c.height=y1-y0;let ctx=c.getContext('2d');ctx.drawImage(bg,x0,y0,c.width,c.height,0,0,c.width,c.height);
if(method!=='original'){let layer=document.createElement('canvas');layer.width=bg.width;layer.height=bg.height;let lc=layer.getContext('2d');lc.drawImage(images[method],0,0);let d=lc.getImageData(0,0,layer.width,layer.height);for(let p=0;p<d.data.length;p+=4){if((d.data[p]===45&&!byId('retained').checked)||(d.data[p]===240&&!byId('rejected').checked))d.data[p+3]=0;}lc.putImageData(d,0,0);ctx.globalAlpha=a;ctx.drawImage(layer,x0,y0,c.width,c.height,0,0,c.width,c.height);ctx.globalAlpha=1;}}
byId('position').textContent=r.dataset+' | zero-based z='+r.z+' | source '+r.source_position.toFixed(6)+' ('+r.source_seconds.toFixed(3)+' s) | mask source frames '+r.mask_frames[0]+'–'+r.mask_frames[1]+' inclusive';}
for(let id of ['sample','alpha','crop','retained','rejected','background'])byId(id).addEventListener('input',render);render();</script></html>'''
    path.write_text(html.replace('PAYLOAD', payload), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error('Generated overlays belong in Scratch, outside the repository')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'overlays.json').exists() or (output / 'source_scaled.nrrd').exists():
        parser.error('Use a fresh output directory')
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib_cache'))
    cv2.setNumThreads(2)
    report = json.loads(args.comparison.read_text(encoding='utf-8'))
    datasets = report['datasets']
    shape = tuple(datasets[0]['shape_tyx'])
    source_shape = tuple(datasets[0]['geometry_context']['source_shape_tyx'])
    if any(tuple(d['shape_tyx']) != shape or tuple(d['geometry_context']['source_shape_tyx']) != source_shape for d in datasets):
        raise ValueError('All comparisons must share source and output grids')
    selected = sorted({s['z'] for d in datasets for s in d['previews']['slices']})
    before = args.source.stat()
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,pix_fmt,r_frame_rate:stream_tags=rotate', '-of', 'json', str(args.source)]))
    stream = probe['streams'][0]
    if (stream['height'], stream['width']) != source_shape[1:]:
        raise ValueError('Source video dimensions do not match run geometry')
    if int(stream.get('tags', {}).get('rotate', 0)) != 0:
        raise ValueError('Rotated video input requires explicit orientation reconciliation')
    input_paths = {Path(m['output']) for d in datasets for m in d['methods'] if m['status'] == 'complete'}
    mask_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    fps = None
    for dataset in datasets:
        inputs = json.loads(Path(dataset['run_manifest']).read_text(encoding='utf-8'))['inputs']
        if inputs['source']['size_bytes'] != before.st_size or Path(inputs['source']['path']).name != args.source.name:
            raise ValueError('Source identity metadata differs from the original run')
        if tuple(inputs['source_shape_t_y_x']) != source_shape:
            raise ValueError('Original run source dimensions differ')
        if fps is not None and fps != inputs['fps']:
            raise ValueError('Original run frame rates differ')
        fps = float(inputs['fps'])
    rate_n, rate_d = map(int, stream['r_frame_rate'].split('/'))
    if not math.isclose(rate_n / rate_d, fps):
        raise ValueError('Source video frame rate differs from run metadata')
    started = time.monotonic()
    raw = output / 'source_scaled.u8.dat'
    gray = np.memmap(raw, mode='w+', dtype=np.uint8, shape=shape)
    try:
        frames = video_frames(args.source, source_shape, output / 'decode.log')
        try:
            envelopes = resize_stream(frames, source_shape, gray, selected,
                progress=lambda n, total: print(f'Scaling source {n}/{total}', flush=True))
        finally:
            frames.close()
        gray.flush()
        write_gray_nrrd(output / 'source_scaled.nrrd', gray)
        records = []
        for dataset in datasets:
            records.extend(render_dataset(output, dataset, gray, envelopes))
        plan, footprints = sampling_plan(source_shape, shape, selected)
        for record in records:
            z = record['z']
            lo, hi, alpha = plan[z]
            record.update(source_position=lo+alpha, source_seconds=(lo+alpha)/fps,
                          mask_frames=[footprints[z][0], footprints[z][1]-1])
        write_viewer(output / 'review.html', records)
        after = args.source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Source changed while overlays were generated')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in mask_hashes.items()):
            raise RuntimeError('A comparison mask changed during overlay generation')
        receipt = dict(source=str(args.source.resolve()), source_size_bytes=before.st_size,
            source_modified_time_ns=before.st_mtime_ns, source_shape_tyx=source_shape, output_shape_tyx=shape,
            source_fps=fps, source_unchanged=True, comparison_masks_unchanged=True,
            comparison_mask_sha256=mask_hashes, video_probe=probe, comparison=str(args.comparison.resolve()),
            comparison_sha256=hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
            tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            spatial_resize='OpenCV INTER_AREA on each uint8 source frame',
            temporal_resize='endpoint-aligned float32 interpolation followed by rint and uint8',
            intensity_display='original uint8 range, no contrast window or normalization',
            slices=[{k:v for k,v in r.items() if k not in {'source','slab','layers'}} for r in records],
            elapsed_seconds=time.monotonic()-started)
        (output / 'overlays.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k:v for k,v in receipt.items() if k in {'output_shape_tyx','elapsed_seconds','source_unchanged'}}), flush=True)
    finally:
        gray._mmap.close()
        raw.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
