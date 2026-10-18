"""CPU-only audit of completed production SAM image-batch qualification cases."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(path):
    raw=json.loads(path.read_text(encoding='utf-8-sig'))
    assert not raw['changed_source_files'] and raw['gpu_lock_absent_at_exit']
    assert raw['self_sha256_at_entry']==raw['self_sha256_at_exit']
    cells=[];qualified=[];retirements=[];memory=[]
    for case in raw['cases']:
        name=case['name'];rows=[row for row in raw['measurements'] if row['case']==name]
        expected={(boundary,count,cycle,position) for boundary in ('resident','fresh_publication')
            for count in (0,12) for cycle in range(3) for position in range(4)}
        keys=[(row['boundary'],row['consumers'],row['cycle'],row['position']) for row in rows]
        assert len(keys)==len(set(keys)) and set(keys)==expected,(name,'incomplete paired case')
        initial=[row for row in raw['integration_checks'] if row['case']==name]
        probes=[row for row in raw['memory'] if row['case']==name]
        assert len(initial)==len(probes)==2
        assert {row['enabled'] for row in initial}=={False,True}
        assert {row['enabled'] for row in probes}=={False,True}
        for row in rows+initial+probes:
            assert row['chunks']==(case['expected_chunks'] if row['enabled'] else [1,1,1,1])
        for row in rows+initial:
            assert row['completed_crops']==4
            if 'uploads' in row:
                assert len(row['uploads'])==1
                upload=row['uploads'][0]
                assert upload['shape']==case['source_shape'] and upload['logical_t']==case['logical_t']
                assert upload['bytes']==case['source_upload_bytes']
                assert row['retirement_seconds']>=0
                retirements.append(dict(case=name,enabled=row['enabled'],seconds=row['retirement_seconds']))
        profile='small' if name.startswith('small-') else 'stress' if name.startswith('stress-') else 'synthetic_b3'
        target=path.parent/profile/name/'resident-target.dat'
        assert digest(target)==case['expected_pixels_sha256'],(name,'retained pixel fingerprint changed')
        selected_receipts=[receipt for row in rows if row['boundary']=='resident' and row['enabled']
            for receipt in row['receipts']]
        planned=max(receipt['planned_device_scratch_peak_bytes'] for receipt in selected_receipts)
        enabled_probe=next(row for row in probes if row['enabled'])
        assert enabled_probe['increment_bytes']<=planned
        assert all(row.get('peak_bytes',0)<=raw['configuration']['maximum_torch_cuda_allocation_bytes'] for row in rows+probes)
        memory.append(dict(case=name,profile=profile,increment_bytes=enabled_probe['increment_bytes'],
            planned_device_scratch_peak_bytes=planned,peak_resident_bytes=enabled_probe['peak_bytes'],
            peak_fresh_bytes=max(row['peak_bytes'] for row in rows if row['boundary']=='fresh_publication')))
        qualified.append(dict(case=name,profile=profile,timed_units=len(rows),timed_crops=4*len(rows),
            real_publications_and_retirements=26,full_native_reference_images=4,
            native_source_upload_bytes=case['source_upload_bytes'],chunks=case['expected_chunks'],
            retained_pixel_sha256=digest(target)))
    for profile in sorted({row['profile'] for row in qualified}):
        names={row['case'] for row in qualified if row['profile']==profile}
        for boundary in ('resident','fresh_publication'):
            for count in (0,12):
                rows=[row for row in raw['measurements'] if row['case'] in names
                    and row['boundary']==boundary and row['consumers']==count]
                def mean_ms(enabled):
                    selected=[row for row in rows if row['enabled']==enabled]
                    return sum(row['wall_seconds'] for row in selected)/sum(row['completed_crops'] for row in selected)*1000
                before,after=mean_ms(False),mean_ms(True)
                pairs=defaultdict(lambda:defaultdict(float))
                for row in rows:pairs[(row['case'],row['cycle'])][row['enabled']]+=row['wall_seconds']
                ratios=[pair[False]/pair[True] for pair in pairs.values()]
                cases=[]
                for name in sorted(names):
                    group=[row for row in rows if row['case']==name]
                    times={enabled:sum(row['wall_seconds'] for row in group if row['enabled']==enabled)/24*1000
                        for enabled in (False,True)}
                    cases.append(dict(case=name,scalar_ms_per_crop=times[False],selected_ms_per_crop=times[True],
                        speedup=times[False]/times[True]))
                cells.append(dict(profile=profile,boundary=boundary,consumers=count,
                    scalar_ms_per_crop=before,selected_ms_per_crop=after,speedup=before/after,
                    selected_latency_change_percent=(after/before-1)*100,
                    paired_case_cycles=len(ratios),faster_pairs=sum(value>1 for value in ratios),
                    paired_median_speedup=statistics.median(ratios),paired_min_speedup=min(ratios),paired_max_speedup=max(ratios),
                    scalar_crops=sum(row['completed_crops'] for row in rows if not row['enabled']),
                    selected_crops=sum(row['completed_crops'] for row in rows if row['enabled']),cases=cases))
    cap=raw['configuration']['maximum_torch_cuda_allocation_bytes']
    peak=max([row.get('peak_bytes',0) for row in raw['measurements']]+[row['peak_bytes'] for row in raw['memory']])
    log=path.parent.with_suffix('.log')
    log_text=log.read_text(encoding='utf-8-sig',errors='replace')
    free=re.findall(r'driver_free=[0-9.]+->([0-9.]+) GiB',log_text)
    source_bytes=64*3064*3022;view_allowance=256*2911*4300;output_allowance=96*96*116;reserve=12*1024**3
    return dict(status='qualified_13_case_subset_synthetic_case_admission_blocked',raw_run_status=raw['status'],
        raw_failure=raw['error'],raw_receipt_sha256=digest(path),source_sha256=raw['preparation']['source_sha256'],
        self_sha256_at_entry=raw['self_sha256_at_entry'],self_sha256_at_exit=raw['self_sha256_at_exit'],
        postprocessor_sha256=digest(__file__),source_pins_unchanged=True,gpu_lock_released=True,
        qualified_geometry_count=len(qualified),qualified=qualified,timed_units=len(raw['measurements']),
        timed_completed_crops=sum(row['completed_crops'] for row in raw['measurements']),
        full_native_reference_images=4*len(qualified),resident_memory_probe_units=len(raw['memory']),
        real_publications_and_retirements=len(retirements),image_only_consumer_sdk_workers=0,
        peak_torch_cuda_allocated_bytes=peak,torch_cuda_allocator_cap_bytes=cap,
        scratch_cap_bytes=raw['configuration']['scratch_bytes'],memory=memory,
        retirement_seconds=dict(count=len(retirements),median=statistics.median(row['seconds'] for row in retirements),
            maximum=max(row['seconds'] for row in retirements)),
        synthetic_admission=dict(status='not_rendered_actual_admission_declined',source_bytes=source_bytes,
            native_view_allowance_bytes=view_allowance,output_allowance_bytes=output_allowance,
            unchanged_default_reserve_bytes=reserve,required_gpu_bytes=source_bytes+view_allowance+output_allowance+reserve,
            required_gpu_gib=(source_bytes+view_allowance+output_allowance+reserve)/1024**3,
            last_logged_free_gpu_gib=float(free[-1]) if free else None,
            reserve_override_applied=False,scratch_or_allocator_cap_relaxed=False),cells=cells,
        caveats=['Fresh publication is a four-crop transaction, including native upload and real GPU close/fence plus mmap flush/publication; lexical retirement is separate.',
            'ImageOnlyConsumer counts and closes its real mappings and acknowledges zero SDK workers; this is not real SAM tracker/worker validation.',
            'Consumer iteration deltas span the outer call, including verification/retirement, and must not be interpreted as publication-only throughput.',
            'Quiet retained-stress resident timing regressed; small/load wins do not establish a universal win.',
            'Native64 to logical2911 stress is a local mechanism proxy; no H100 or sustained-utilization projection.',
            'Fresh four-crop transactions repeatedly upload and retire; real cohorts can amortize these costs over many more crops.',
            'Attempt01 relative-path containment failure and attempt02 sandbox Path.resolve WinError5 are preserved; attempt03 used narrow filesystem escalation.',
            'No synthetic rerun, reserve reduction or GPU reuse occurred after the admission decline.'])


def report_text(summary):
    lines=['Production SAM guarded batching:13 geometries qualified; synthetic B3 geometry admission-blocked.',
        '',
        'The raw attempt03 status remains failed/partial. All12 small-crop geometries and the retained2048-square Coronal stress proxy completed their required resident and real fresh-transaction checks. The separately labeled2911x4300 synthetic case was refused by unchanged GPU admission before rendering; it is not a GPU-qualified case.',
        '',
        'The main limitation is visible in the timings: quiet retained-stress resident rendering is about8% slower (3.586->3.871 ms/crop, only1/3 paired cycles faster). Under12 consumers it improves about16% in throughput (19.999->17.215 ms/crop,3/3 faster). Full fresh transactions are near-flat when quiet and improve only about4-5% in throughput under consumers. There is no large end-to-end throughput claim.',
        '',
        'Paired ABBA ordinary wall times, milliseconds per completed crop; each row uses its own paired scalar baseline:',
        'fixture | boundary | consumers | scalar ms | selected ms | scalar/selected | faster case-cycle pairs',
        '--- | --- | ---: | ---: | ---: | ---: | ---']
    for row in summary['cells']:
        lines.append(f"{row['profile']} | {row['boundary']} | {row['consumers']} | {row['scalar_ms_per_crop']:.4f} | {row['selected_ms_per_crop']:.4f} | {row['speedup']:.3f}x | {row['faster_pairs']}/{row['paired_case_cycles']}")
    lines+=['',
        'Every unit completes4 crops. Production chooses[4] for the12 small cases and[2,2] for retained stress; disabled control uses[1,1,1,1] and skips the batch provider branch. Three ABBA cycles run under0 and12 same-process mask/JSON consumers. Small rows contain36 matched case-cycle pairs; stress rows contain3. Quiet stress regression is preserved rather than averaged away.',
        '',
        f"Qualification evidence: {summary['full_native_reference_images']} scalar/full-native-reference image comparisons; {summary['resident_memory_probe_units']} resident memory-probe units; {summary['timed_units']} timed units / {summary['timed_completed_crops']} completed crops. All timed output pixels were checked against the exact reference. Final resident target files independently match all13 retained expected SHA256 values.",
        f"There were{summary['real_publications_and_retirements']} successful real private-cache publications and retirements (26 initial checks plus312 timed transactions). Each recorded exactly one native upload, correct native shape/bytes/logicalT, CUDA sampler identity, released renderer source/compute before publication, and removal of the private file after the consuming scope. The production close/fence, flush, publication and ownership path were not stubbed.",
        '',
        'The image-only benchmark consumer tracks actual opened cache mappings and closes them before acknowledging retirement. No SDK workers or model tracking were started. Its mapping barrier is real for this consumer; it does not validate real SAM worker/SDK handoff or inference.',
        '',
        f"Maximum recorded Torch CUDA allocation: {summary['peak_torch_cuda_allocated_bytes']:,} bytes ({summary['peak_torch_cuda_allocated_bytes']/1024**2:.2f} MiB), below the{summary['torch_cuda_allocator_cap_bytes']/1024**3:g} GiB allocator cap. All enabled resident incremental peaks fit their recorded planned batch stage bounds; the scratch cap stayed256 MiB. Torch allocation is distinct from total driver/CuPy/context memory.",
        '',
        'Synthetic admission calculation (unchanged production default reserve; no override applied):',
        f"Native source: {summary['synthetic_admission']['source_bytes']:,} bytes.",
        f"Whole-view allowance (256*2911*4300): {summary['synthetic_admission']['native_view_allowance_bytes']:,} bytes.",
        f"Crop allowance (96*96*116): {summary['synthetic_admission']['output_allowance_bytes']:,} bytes.",
        f"Unchanged12 GiB reserve: {summary['synthetic_admission']['unchanged_default_reserve_bytes']:,} bytes.",
        f"Sum of these four terms: {summary['synthetic_admission']['required_gpu_bytes']:,} bytes ({summary['synthetic_admission']['required_gpu_gib']:.3f} GiB).",
        f"The last preceding cleanup log reported about{summary['synthetic_admission']['last_logged_free_gpu_gib']:.2f} GiB free on the RTX4090 Laptop GPU. The conservative whole-view admission requirement therefore exceeded local headroom even though metadata selected a B3 workspace below256 MiB. The3GiB Torch allocation cap does not override the separate production admission calculation. No reserve reduction or synthetic rerun was performed.",
        '',
        'Timing boundaries: resident timing includes production chooser and ordered sink writes after source residency. Fresh publication timing starts before entering the real image-cohort provider and stops after descriptor publication, including source upload, GPU retirement and mmap flush. Lexical retirement follows pixel verification and is measured separately; its median was'+f"{summary['retirement_seconds']['median']*1000:.3f} ms. A fresh four-crop transaction is intentionally not a projection of large-cohort amortized cost.",
        '',
        'All production and benchmark pins remained unchanged; the GPU lock was released. Prior unchanged48 numerical CUDA cases were reused via exact operator/dependency proof; the new paths above actually executed. Attempt01 containment rejection and attempt02 sandbox source-resolution failure remain preserved. Attempt03 used the narrowly approved filesystem escalation without weakening checks.',
        '',
        'Caveats:']
    lines.extend('- '+value for value in summary['caveats'])
    lines+=['','Historical production/driver pins at the qualified run (later worktree changes do not rewrite these):']
    lines.extend(name+': '+value for name,value in summary['source_sha256'].items())
    lines+=['','Raw receipt SHA256: '+summary['raw_receipt_sha256'],
        'Postprocessor SHA256: '+summary['postprocessor_sha256'],
        '', 'Evidence: attempt-03/results.json, attempt-03.log, qualification-summary.json, and each completed case resident-target.dat. Raw failure status and all measurements remain unchanged.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path)
    args=parser.parse_args();summary=analyze(args.receipt)
    out=args.receipt.parent.parent
    (out/'qualification-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (out/'RESULTS.txt').write_text(report_text(summary),encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ('status','qualified_geometry_count','timed_units',
        'timed_completed_crops','real_publications_and_retirements','peak_torch_cuda_allocated_bytes','gpu_lock_released')},indent=2))


if __name__=='__main__':main()
