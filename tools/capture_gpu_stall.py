"""Read-only Linux GPU-stall evidence. Standard library only; creates no CUDA context.

Run on the compute host: python tools/capture_gpu_stall.py --output-dir EXISTING_JOB_DIR
The requested seconds cover sampling; bounded topology/process snapshots add overhead.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

PROC = Path('/proc')
SYS = Path('/sys')
QUERY_TIMEOUT = 5.0
CADENCE = 1.0
MAX_PIDS = 64
MAX_THREADS = 4096
MAX_CGROUP_DEPTH = 16
READ_LIMIT = 4 * 1024 * 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def observed(function):
    try:
        return {'value': function(), 'error': None}
    except (OSError, ValueError, ET.ParseError) as exc:
        return {'value': None, 'error': f'{type(exc).__name__}: {exc}'}


def read(path, binary=False):
    data = bytearray()
    # Linux sysctl reads can reject a large requested count before reading a tiny value.
    with path.open('rb', buffering=0) as handle:
        while len(data) <= READ_LIMIT:
            chunk = handle.read(min(4096, READ_LIMIT + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    if len(data) > READ_LIMIT:
        raise ValueError(f'{path}: exceeds {READ_LIMIT}-byte read bound')
    return bytes(data) if binary else data.decode('utf-8', errors='replace').strip()


def command(args):
    start = time.monotonic()
    result = {'command': args, 'started_utc': utc(), 'started_monotonic': start}
    try:
        run = subprocess.run(args, capture_output=True, timeout=QUERY_TIMEOUT, check=False)
        data, stderr = run.stdout, run.stderr
        result.update(returncode=run.returncode, error=None if run.returncode == 0 else 'nonzero exit')
    except subprocess.TimeoutExpired as exc:
        data, stderr = exc.stdout or b'', exc.stderr or b''
        result.update(returncode=None, error='query timeout; stdout is partial')
    except OSError as exc:
        data, stderr = b'', b''
        result.update(returncode=None, error=f'{type(exc).__name__}: {exc}')
    result.update(duration_seconds=time.monotonic() - start,
                  stderr=stderr[:4096].decode('utf-8', errors='replace'),
                  stderr_truncated=len(stderr) > 4096)
    return result, data


def parse_gpus(data):
    root = ET.fromstring(data)
    if root.tag != 'nvidia_smi_log':
        raise ValueError('unexpected nvidia-smi XML root')
    gpus, processes, errors = [], [], []
    gpu_elements = root.findall('gpu')
    attached = root.findtext('attached_gpus', '').strip()
    if attached.isdecimal() and int(attached) != len(gpu_elements):
        errors.append('attached GPU count differs from XML inventory')
    if not gpu_elements and attached != '0':
        errors.append('GPU inventory absent; not assumed empty')
    for gpu in gpu_elements:
        identity = {'uuid': gpu.findtext('uuid'), 'pci_bus_id': gpu.findtext('pci/pci_bus_id') or gpu.get('id'),
                    'minor_number': gpu.findtext('minor_number')}
        if any(not value or value.strip().lower() in ('n/a', 'unknown', 'not supported', '[not supported]')
               for value in (identity['uuid'], identity['pci_bus_id'])):
            errors.append('GPU UUID/PCI identity incomplete')
        gpus.append(identity)
        container = gpu.find('processes')
        if container is None or (container.text or '').strip():
            errors.append('GPU process inventory missing or unsupported')
        for process in gpu.findall('processes/process_info'):
            kind = process.findtext('type')
            if kind is None:
                errors.append('process type missing; not assumed to be compute')
            elif 'C' in kind.split('+'):
                pid = process.findtext('pid', '')
                if pid.isdecimal() and int(pid) > 0:
                    processes.append(dict(identity, pid=int(pid), process_type=kind))
                else:
                    errors.append('compute process PID missing or invalid')
            elif kind != 'G':
                errors.append('unrecognized process type; compute membership unknown')
    return {'gpus': gpus, 'compute_processes': processes, 'errors': errors, 'inventory_complete': not errors}


def pci_files(bus):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', bus or '')
    if not match:
        return {'error': 'missing or invalid PCI address'}
    domain, device_bus, device, function = match.groups()
    address = f'{int(domain, 16):04x}:{device_bus}:{device}.{function}'.lower()
    base = SYS / 'bus/pci/devices' / address
    return {name: observed(lambda name=name: read(base / name)) for name in
            ('numa_node', 'local_cpulist', 'current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width')}


def topology(buses):
    result = {'utc': utc(), 'pci': {bus: pci_files(bus) for bus in sorted(buses)}}
    for name, args in [('gpu_topology', ['nvidia-smi', 'topo', '-m']),
                       ('cpu_topology', ['lscpu', '-p=CPU,CORE,SOCKET,NODE'])]:
        metadata, data = command(args)
        result[name] = dict(metadata, stdout=data.decode('utf-8', errors='replace'))
    return result


def process_stat(pid, base=None):
    stat = read((base if base is not None else PROC / str(pid)) / 'stat')
    fields = stat[stat.rfind(')') + 1:].split()
    if stat.split(' ', 1)[0] != str(pid) or ')' not in stat or len(fields) < 20:
        raise ValueError('malformed process stat identity')
    return {'start_ticks': int(fields[19]),
            'last_processor': int(fields[36]) if len(fields) > 36 else None}


def start_ticks(pid):
    return process_stat(pid)['start_ticks']


def status(path):
    fields = dict(line.split(':', 1) for line in read(path / 'status').splitlines() if ':' in line)
    return {key: fields[key].strip() if key in fields else None for key in
            ('Name', 'PPid', 'State', 'Cpus_allowed_list', 'Mems_allowed_list')}


def job_id(pid):
    for field in read(PROC / str(pid) / 'environ', binary=True).split(b'\0'):
        if field.startswith(b'SLURM_JOB_ID='):
            return field.split(b'=', 1)[1].decode('utf-8', errors='replace')
    return None


def numa_bytes(pid):
    totals = {}
    page_size = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096
    for line in read(PROC / str(pid) / 'numa_maps').splitlines():
        size = re.search(r'\bkernelpagesize_kB=(\d+)\b', line)
        unit = int(size.group(1)) * 1024 if size else page_size
        for token in line.split():
            if re.match(r'N\d+=', token):
                node, count = token.split('=', 1)
                totals[node] = totals.get(node, 0) + int(count) * unit
    return totals


def cgroup_cpu(pid):
    entries = [line.split(':', 2) for line in read(PROC / str(pid) / 'cgroup').splitlines()]
    entry = next((x for x in entries if len(x) == 3 and x[0] == '0' and not x[1]), None)
    if entry is None:
        raise ValueError('unified cgroup path unavailable; cpu.max coverage unknown')
    member = PurePosixPath(entry[2])
    if not member.is_absolute() or '..' in member.parts:
        raise ValueError('cgroup path escapes mounted hierarchy')
    mounts = []
    for line in read(PROC / 'self/mountinfo').splitlines():
        before, separator, after = line.partition(' - ')
        fields = before.split()
        if separator and after.split()[:1] == ['cgroup2'] and len(fields) >= 5:
            unescape = lambda text: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), text)
            root, mount = PurePosixPath(unescape(fields[3])), Path(unescape(fields[4]))
            if root.is_absolute() and mount.is_absolute() and member.is_relative_to(root):
                mounts.append((len(root.parts), root, mount))
    if not mounts:
        raise ValueError('no observer cgroup2 mount covers membership; mapping unknown')
    _, root, mount = max(mounts, key=lambda item: item[0])
    base = mount.joinpath(*member.relative_to(root).parts)
    memory_files = ('memory.current', 'memory.max', 'memory.high', 'memory.peak',
                    'memory.events', 'memory.stat', 'memory.pressure',
                    'memory.swap.current', 'memory.swap.max', 'memory.swap.events')
    levels, current = [], base
    for _ in range(MAX_CGROUP_DEPTH):
        levels.append({'directory': str(current),
                       'cpu.stat': observed(lambda: dict(line.split() for line in read(current / 'cpu.stat').splitlines())),
                       'cpu.max': observed(lambda: read(current / 'cpu.max')),
                       **{name: observed(lambda name=name: read(current / name)) for name in memory_files}})
        if current == mount:
            break
        current = current.parent
    return {'path': entry[2], 'mount_root': str(root), 'mountpoint': str(mount),
            'cpu.stat': levels[0]['cpu.stat'], 'cpu.max': levels[0]['cpu.max'],
            **{name: levels[0][name] for name in memory_files},
            'ancestors': levels[1:], 'visible_root_reached': levels[-1]['directory'] == str(mount),
            'omitted_visible_levels': max(0, len(member.relative_to(root).parts) + 1 - len(levels)),
            'above_visible_root': 'unknown; no effective quota inferred'}


def process_snapshot(pid, expected, thread_budget):
    identity = observed(lambda: start_ticks(pid))
    result = {'pid': pid, 'expected_start_ticks': expected, 'start_ticks': identity, 'utc': utc()}
    if identity['value'] != expected or identity['error']:
        result['state'] = 'unavailable' if identity['error'] else 'pid_reused'
        return result
    base = PROC / str(pid)
    result.update(state='observed', status=observed(lambda: status(base)),
                  stat=observed(lambda: process_stat(pid)),
                  slurm_job_id=observed(lambda: job_id(pid)),
                  schedstat=observed(lambda: read(base / 'schedstat')),
                  wchan=observed(lambda: read(base / 'wchan')),
                  numa_mapped_bytes=observed(lambda: numa_bytes(pid)),
                  cgroup_cpu=observed(lambda: cgroup_cpu(pid)))
    tids = observed(lambda: sorted(int(p.name) for p in (base / 'task').iterdir() if p.name.isdecimal()))
    result['thread_inventory'] = tids
    selected = (tids['value'] or [])[:thread_budget[0]]
    result['omitted_threads'] = None if tids['error'] else len(tids['value']) - len(selected)
    thread_budget[0] -= len(selected)
    result['threads'] = []
    for tid in selected:
        path = base / 'task' / str(tid)
        row = {'tid': tid, 'stat': observed(lambda: process_stat(tid, path)),
               'status': observed(lambda: status(path)),
               'schedstat': observed(lambda: read(path / 'schedstat')),
               'wchan': observed(lambda: read(path / 'wchan'))}
        row['after_stat'] = observed(lambda: process_stat(tid, path))
        row['identity_stable'] = row['stat']['error'] is None and row['after_stat']['error'] is None and row['stat']['value']['start_ticks'] == row['after_stat']['value']['start_ticks']
        result['threads'].append(row)
    result['after_start_ticks'] = observed(lambda: start_ticks(pid))
    if result['after_start_ticks']['value'] != expected or result['after_start_ticks']['error']:
        result['state'] = 'changed_or_disappeared_during_snapshot'
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def capture(output_dir, seconds):
    if not str(output_dir).strip():
        raise ValueError('--output-dir must not be empty (set the persistent job path)')
    if not 1 <= seconds <= 120:
        raise ValueError('--seconds must be between 1 and 120')
    output_dir = Path(output_dir).absolute()
    if not output_dir.exists():
        raise FileNotFoundError(f'--output-dir does not exist: {output_dir}')
    if not output_dir.is_dir():
        raise ValueError('--output-dir must be an existing persistent directory')
    folder = Path(tempfile.mkdtemp(prefix='gpu-stall-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'), dir=output_dir))
    manifest = {'schema': 'xta.gpu-stall/1', 'hostname': socket.gethostname(), 'started_utc': utc(),
                'seconds_requested': seconds, 'pid_limit': MAX_PIDS, 'threads_per_boundary_limit': MAX_THREADS,
                'cgroup_levels_per_boundary_limit': MAX_CGROUP_DEPTH,
                'boot_id': observed(lambda: read(PROC / 'sys/kernel/random/boot_id')),
                'coverage': 'Reported compute PIDs only; null/error means unknown. PID/TID start ticks bind observations. wchan=0 is ambiguous, not proof of no waiting. Topology and boundary snapshots add overhead.'}
    tracked, buses, first_pci = {}, set(), {}
    budget = [MAX_THREADS]
    total_start = time.monotonic()
    index = valid = inventories = 0
    try:
        save(folder / 'topology-start.json', topology(buses))
        start = time.monotonic()
        deadline = start + seconds
        with (folder / 'samples.jsonl').open('x', encoding='utf-8') as handle:
            while time.monotonic() < deadline:
                tick = time.monotonic()
                query, data = command(['nvidia-smi', '-q', '-x'])
                filename = f'gpu-{index:04d}' + ('.partial.xml' if query['returncode'] is None else '.xml')
                if data:
                    (folder / filename).write_bytes(data)
                parsed = observed(lambda: parse_gpus(data)) if query['error'] is None else {'value': None, 'error': query['error']}
                sample = {'query': query, 'xml_file': filename if data else None, 'parsed': parsed, 'processes': []}
                pids = set()
                if parsed['value'] is not None:
                    valid += 1
                    inventories += int(parsed['value']['inventory_complete'])
                    for gpu in parsed['value']['gpus']:
                        bus = gpu['pci_bus_id']
                        if bus and bus not in buses:
                            buses.add(bus)
                            first_pci[bus] = {'utc': utc(), 'files': pci_files(bus)}
                    pids = {p['pid'] for p in parsed['value']['compute_processes']}
                omitted = []
                for pid in sorted(pids):
                    identity = observed(lambda: start_ticks(pid))
                    key = (pid, identity['value'])
                    if key not in tracked:
                        if len(tracked) >= MAX_PIDS:
                            omitted.append(pid)
                            continue
                        tracked[key] = process_snapshot(pid, identity['value'], budget)
                    item = {'pid': pid, 'start_ticks': identity,
                            'stat': observed(lambda: process_stat(pid)),
                            'wchan': observed(lambda: read(PROC / str(pid) / 'wchan')) if not identity['error'] else identity}
                    item['after_start_ticks'] = observed(lambda: start_ticks(pid))
                    item['identity_stable'] = not identity['error'] and not item['after_start_ticks']['error'] and identity['value'] == item['after_start_ticks']['value']
                    sample['processes'].append(item)
                sample['omitted_pids'] = omitted
                sample['previous_pids_not_reported'] = sorted({pid for pid, _ in tracked} - pids) if parsed['value'] is not None and parsed['value']['inventory_complete'] else None
                sample['process_coverage'] = 'unknown' if parsed['value'] is None else ('reported' if parsed['value']['inventory_complete'] else 'partial')
                handle.write(json.dumps(sample) + '\n')
                handle.flush()
                index += 1
                time.sleep(max(0, min(CADENCE - (time.monotonic() - tick), deadline - time.monotonic())))
        manifest.update(samples=index, sampling_elapsed_seconds=time.monotonic() - start)
    except KeyboardInterrupt:
        manifest['interrupted'] = True
    finally:
        save(folder / 'process-first-seen.json', list(tracked.values()))
        save(folder / 'pci-first-seen.json', first_pci)
        end_budget = [MAX_THREADS]
        save(folder / 'process-end.json', [process_snapshot(pid, ticks, end_budget) for pid, ticks in tracked])
        save(folder / 'topology-end.json', topology(buses))
        manifest.update(finished_utc=utc(), total_elapsed_seconds=time.monotonic() - total_start)
        manifest.update(gpu_queries_attempted=index, valid_xml_queries=valid, complete_process_inventory_samples=inventories,
                        gpu_query_coverage='failed' if not valid else ('partial' if inventories != index or manifest.get('interrupted') else 'sampled'))
        save(folder / 'manifest.json', manifest)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Existing persistent job directory; creates a unique child folder')
    parser.add_argument('--seconds', type=int, default=15, help='Sampling window, 1..120 seconds (default:15)')
    args = parser.parse_args(argv)
    try:
        folder = capture(args.output_dir, args.seconds)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Capture failed: {exc}\n')
    coverage = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))['gpu_query_coverage']
    if coverage == 'failed':
        parser.exit(1, f'Saved evidence: {folder}; all GPU queries failed. Inspect recorded errors.\n')
    print(f'Saved GPU stall evidence: {folder} (GPU coverage: {coverage}; per-field errors indicate unknown data)')


if __name__ == '__main__':
    main()
