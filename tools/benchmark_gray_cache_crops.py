"""Bounded CPU crop-reader microbenchmark; no whole-pipeline throughput claim.

Use separate invocations with --source-root to alternate unchanged and candidate
trees. Each invocation imports that tree's actual production reader. Results,
fixtures and diagnostics remain in the required output directory outside source.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import sqlite3
import statistics
import sys
import threading
import time
import zlib

def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_identity(root):
    files = {path.relative_to(root).as_posix(): _digest(path)
             for path in sorted((root/'XTA').rglob('*.py'))}
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return dict(scope='XTA/**/*.py raw file bytes', files=files,
                sha256=hashlib.sha256(encoded).hexdigest())


def _requests(pattern, scale):
    if pattern == 'hot_small':
        return [(i % 8, (16+i % 40,20+i % 30,80+i % 40,84+i % 30))
                for i in range(128*scale)]
    if pattern == 'cross_tile':
        return [(i % 16,(240,244,560,564)) for i in range(32*scale)]
    if pattern == 'full_plane':
        return [(i % 16,(0,0,1024,1024)) for i in range(16*scale)]
    raise ValueError(pattern)


def _profile_summary(profile):
    rows = []
    for (filename,line,name), (_primitive,calls,own,cumulative,_callers) in pstats.Stats(profile).stats.items():
        if any(token in name for token in ('execute','fetchone','_check','_decode','read_crop','sha256','stat')):
            rows.append(dict(file=filename,line=line,function=name,calls=calls,
                             own_seconds=own,cumulative_seconds=cumulative))
    return sorted(rows,key=lambda row:row['own_seconds'],reverse=True)[:24]


def _measure(gray, path, spec, data, plan, cache_bytes, *, diagnostic=False):
    import numpy as np
    import psutil
    process = psutil.Process()
    rss_start = process.memory_info().rss
    peak = [rss_start]
    stop = threading.Event()
    def sample():
        while not stop.wait(.01):
            peak[0] = max(peak[0],process.memory_info().rss)
    sampler = threading.Thread(target=sample,daemon=True)
    sampler.start()
    statements = []
    outputs = []
    admission_started = time.perf_counter()
    try:
        with gray.GrayCacheReader(path,spec,max_cache_bytes=cache_bytes) as reader:
            admission = time.perf_counter()-admission_started
            if diagnostic:
                reader._connection.set_trace_callback(statements.append)
            profiler = cProfile.Profile() if diagnostic else None
            if profiler:
                profiler.enable()
            started = time.perf_counter()
            for frame,box in plan:
                outputs.append(reader.read_crop(frame,box))
            elapsed = time.perf_counter()-started
            if profiler:
                profiler.disable()
            # Exact comparison is outside the timed crop region. Retaining <=16 MiB
            # default outputs keeps lifetimes comparable across source invocations.
            for output,(frame,(y0,x0,y1,x1)) in zip(outputs,plan):
                np.testing.assert_array_equal(output,data[frame,y0:y1,x0:x1])
            peak[0] = max(peak[0],process.memory_info().rss)
            row = dict(seconds=elapsed,admission_seconds=admission,requests=len(plan),
                       rss_start=rss_start,sampled_peak_rss=peak[0],
                       cached_bytes_final=reader._cached_bytes,equality=True,
                       diagnostic=diagnostic)
            if diagnostic:
                row['sql_counts'] = dict(total=len(statements),
                    tile_metadata=sum('SELECT h,w,raw_size' in sql for sql in statements),
                    payload=sum('SELECT data FROM chunks' in sql for sql in statements),
                    frame_mapping=sum('SELECT packet,plane' in sql for sql in statements),
                    sealed_version=sum('PRAGMA data_version' in sql for sql in statements))
                row['profile'] = _profile_summary(profiler)
            return row
    finally:
        stop.set()
        sampler.join()


def _summarize(rows):
    groups = {}
    for row in rows:
        if not row['diagnostic']:
            key = row['codec'],row['pattern'],row['cache_bytes']
            groups.setdefault(key,[]).append(row)
    return [dict(codec=key[0],pattern=key[1],cache_bytes=key[2],
                 requests=values[0]['requests'],trials=len(values),
                 median_seconds=statistics.median(row['seconds'] for row in values),
                 min_seconds=min(row['seconds'] for row in values),
                 max_seconds=max(row['seconds'] for row in values),
                 max_sampled_rss=max(row['sampled_peak_rss'] for row in values))
            for key,values in groups.items()]


def _compare(current, prior_path):
    prior = json.loads(prior_path.read_text(encoding='utf-8'))

    def validated_cases(report, label):
        if not isinstance(report, dict) or report.get('schema') != 'xta.gray_crop_benchmark/1':
            raise ValueError(label+' comparison schema is invalid')
        if report.get('source_identity_unchanged') is not True:
            raise ValueError(label+' comparison source integrity is incomplete')
        if not isinstance(report.get('cache_regime'), str) or not report['cache_regime']:
            raise ValueError(label+' comparison cache regime is missing')
        measured = [row for row in report.get('rows', []) if row.get('diagnostic') is False]
        if not measured or any(row.get('equality') is not True for row in measured):
            raise ValueError(label+' comparison measured equality is incomplete')
        index = {}
        for row in report.get('summary', []):
            key = row['codec'],row['pattern'],row['cache_bytes'],row['requests']
            if key in index:
                raise ValueError(label+' comparison has duplicate summary case keys')
            index[key] = row
        measured_keys = {(row['codec'],row['pattern'],row['cache_bytes'],row['requests'])
                         for row in measured}
        if not index or set(index) != measured_keys:
            raise ValueError(label+' comparison summary differs from its measured cases')
        return index

    current_index = validated_cases(current, 'Current')
    prior_index = validated_cases(prior, 'Prior')
    if prior.get('fixture_sha256') != current['fixture_sha256']:
        raise ValueError('Comparison fixture differs')
    if prior['cache_regime'] != current['cache_regime']:
        raise ValueError('Comparison cache regime differs')
    if set(prior_index) != set(current_index):
        raise ValueError('Comparison workload differs')
    rows = []
    for key,row in current_index.items():
        before,after = prior_index[key]['median_seconds'],row['median_seconds']
        rows.append(dict(codec=key[0],pattern=key[1],cache_bytes=key[2],requests=key[3],
                         prior_seconds=before,current_seconds=after,
                         saved_seconds=before-after,saved_fraction=(before-after)/before))
    return dict(prior_path=str(prior_path),prior_result_sha256=_digest(prior_path),
                interpretation='Separate-invocation comparison; caller controls ordering and contention',rows=rows)


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path.cwd(),
                        help='Source checkout containing XTA (default: current directory)')
    parser.add_argument('--output-dir',type=Path,required=True,
                        help='New output directory outside the source checkout; retains fixture SQLite files')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--heat-seconds',type=float,default=10.,
                        help='Global same-matrix CPU heatsoak excluded from measured rows (default: 10)')
    parser.add_argument('--workload-scale',type=int,default=1,help='Request-count multiplier, 1..8 (default: 1)')
    parser.add_argument('--compare',type=Path,help='Optional prior results.json from this tool')
    args = parser.parse_args(argv)
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    if not (source/'XTA'/'gray_cache.py').is_file():
        parser.error('source-root must contain XTA/gray_cache.py')
    if output == source or source in output.parents:
        parser.error('output-dir must be outside source-root')
    # When installed in a checkout's tools/, also keep output out of that checkout
    # even if source-root selects a different baseline checkout.
    tool_checkout = Path(__file__).resolve().parent.parent
    if (tool_checkout/'XTA'/'gray_cache.py').is_file() and (output == tool_checkout or tool_checkout in output.parents):
        parser.error('output-dir must also be outside this tool checkout')
    if args.repeats < 1 or not 0 <= args.heat_seconds < float('inf') or not 1 <= args.workload_scale <= 8:
        parser.error('repeats must be positive; heat finite/nonnegative; workload-scale 1..8')
    if output.exists():
        parser.error('output-dir must be new to preserve previous evidence')
    sys.path.insert(0,str(source))
    import numpy as np
    import psutil
    from XTA import gray_cache as gray
    if Path(gray.__file__).resolve() != (source/'XTA'/'gray_cache.py').resolve():
        raise RuntimeError('Imported gray reader did not come from selected source-root')
    output.mkdir(parents=True)
    spec = dict(source_identity='bounded-synthetic-gray-crop-v1',
                physical_view_record={'name':'bounded_fixture'},native_centers=list(range(16)),
                canvas_shape=[16,1024,1024],capture_contract={'channel':0})
    rng = np.random.default_rng(270)
    data = rng.integers(0,16,size=(16,1024,1024),dtype=np.uint8)
    data += (np.arange(1024,dtype=np.uint16)[None,None,:]//8).astype(np.uint8)
    report = dict(schema='xta.gray_crop_benchmark/1',kind='CPU crop-reader microbenchmark',
        claim_scope='No whole-pipeline throughput claim',source_root=str(source),
        reader_module=str(gray.__file__),package_python_identity=_package_identity(source),
        gray_source_sha256=_digest(Path(gray.__file__)),tool_sha256=_digest(Path(__file__)),
        python=sys.version,python_executable=sys.executable,platform=platform.platform(),
        numpy_version=np.__version__,sqlite_version=sqlite3.sqlite_version,
        zlib_version=zlib.ZLIB_RUNTIME_VERSION,psutil_version=psutil.__version__,
        gpu=False,fixture_bytes=data.nbytes,fixture_sha256=hashlib.sha256(data).hexdigest(),
        cache_regime='Fresh decoded LRU per run; warmed OS page cache; no cold-storage claim',
        rss_scope='10ms sampled process RSS including fixture, retained outputs and reader; not isolated allocation attribution',
        repeats=args.repeats,workload_scale=args.workload_scale,
        heat_requested_seconds=args.heat_seconds,fixtures=[],rows=[])
    cases = []
    for codec in ('raw','zlib'):
        path = output/(codec+'.sqlite')
        with gray.GrayCacheWriter(path,spec,codec,min_free_bytes=0) as writer:
            for first in (0,8):
                if not writer.write_frames(range(first,first+8),data[first:first+8]):
                    raise RuntimeError('Synthetic fixture packet publication declined')
            if not writer.seal():
                raise RuntimeError('Synthetic fixture did not seal')
        report['fixtures'].append(dict(codec=codec,path=str(path),bytes=path.stat().st_size,sha256=_digest(path)))
        for pattern in ('hot_small','cross_tile','full_plane'):
            for cache_bytes in (0,32*1024**2):
                cases.append((path,_requests(pattern,args.workload_scale),
                              dict(codec=codec,pattern=pattern,cache_bytes=cache_bytes)))
    heat_started = time.perf_counter()
    heat_runs = 0
    while time.perf_counter()-heat_started < args.heat_seconds:
        path,plan,common = cases[heat_runs % len(cases)]
        _measure(gray,path,spec,data,plan,common['cache_bytes'])
        heat_runs += 1
    report.update(heat_actual_seconds=time.perf_counter()-heat_started,heat_runs=heat_runs,
                  heat_scope='Global same-matrix heatsoak; excluded from rows')
    for path,plan,common in cases:
        _measure(gray,path,spec,data,plan,common['cache_bytes'])  # Unrecorded page-cache warmup.
        for repeat in range(args.repeats):
            row = _measure(gray,path,spec,data,plan,common['cache_bytes'])
            report['rows'].append(dict(common,repeat=repeat,**row))
        row = _measure(gray,path,spec,data,plan,common['cache_bytes'],diagnostic=True)
        report['rows'].append(dict(common,repeat='diagnostic',**row))
        report['summary'] = _summarize(report['rows'])
        (output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(common),flush=True)
    if _package_identity(source) != report['package_python_identity']:
        raise RuntimeError('Package Python source changed during benchmark; evidence is incomplete')
    report['source_identity_unchanged'] = True
    if args.compare:
        report['comparison'] = _compare(report,args.compare.resolve())
    (output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(str(output/'results.json'),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
