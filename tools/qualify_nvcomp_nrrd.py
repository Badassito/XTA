"""Qualify nvCOMP 5.3 RAW Gzip members for the hybrid NRRD writer.

Preparation only reads an existing NRRD and writes small raw samples. Benchmark
mode owns Scratch/Temp/GPU_LOCK before importing CUDA libraries. All outputs go
to a task-specific Scratch directory; nothing is installed in the project venv.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time
from types import SimpleNamespace
import zlib


SCRATCH = Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch")
LOCK = SCRATCH / "Temp" / "GPU_LOCK"
EXPERIMENT = SCRATCH / "Experiments" / "gpu-utilization-148334" / "nvcomp"
DEFAULT_DEFLATE = EXPERIMENT.parent / "compression-repro" / "deps"
SIZES = (64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20)
_DLL_HANDLES: list[object] = []


def prepare(source: Path, output: Path, scan_mib: int = 256) -> dict:
    """Retain varied real 16-MiB windows without decoding an entire huge NRRD."""
    output.mkdir(parents=True, exist_ok=True)
    cap = max(16, int(scan_mib)) << 20
    window = 16 << 20
    scored: list[tuple[int, int, bytes]] = []
    raw_scanned = 0
    with source.open("rb") as fh:
        header = bytearray()
        while not header.endswith(b"\n\n"):
            value = fh.read(1)
            if not value or len(header) > (1 << 20):
                raise ValueError("Missing or oversized NRRD header")
            header.extend(value)
        if b"encoding: gzip" not in header.lower():
            raise ValueError("Expected a gzip NRRD")
        with gzip.GzipFile(fileobj=fh) as reader:
            index = 0
            while raw_scanned < cap:
                block = reader.read(min(window, cap - raw_scanned))
                if not block:
                    break
                count = len(block) - block.count(0)
                scored.append((count, index, block))
                raw_scanned += len(block)
                index += 1
    if not scored:
        raise ValueError("No NRRD payload bytes found")
    nonzero = sorted((item for item in scored if item[0]), key=lambda item: item[0])
    selected: dict[str, tuple[int, int, bytes]] = {
        "sparse": nonzero[0] if nonzero else scored[0],
        "median": nonzero[len(nonzero) // 2] if nonzero else scored[len(scored) // 2],
        "dense": nonzero[-1] if nonzero else scored[-1],
    }
    samples = []
    for name, (count, index, block) in selected.items():
        path = output / f"{name}.raw"
        path.write_bytes(block)
        samples.append({"name": name, "path": str(path), "bytes": len(block),
                        "nonzero_bytes": count, "source_window": index})
    manifest = {"source": str(source), "scan_cap_mib": int(scan_mib),
                "raw_scanned_bytes": raw_scanned, "samples": samples}
    (output / "samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class GpuLock:
    def __init__(self, path: Path = LOCK) -> None:
        self.path = path
        self.owned = False

    def __enter__(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL gives one owner. An existing lock is never removed implicitly.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"qualify_nvcomp_nrrd pid={os.getpid()} start={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        self.owned = True
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.owned:
            self.path.unlink(missing_ok=True)
            self.owned = False


def load_nvcomp(deps: Path, runtime: Path):
    # The project venv has a regular nvidia package, so extend its search path
    # explicitly for the two task-scoped wheel targets.
    import nvidia

    for target in (runtime / "nvidia", deps / "nvidia"):
        if not target.is_dir():
            raise FileNotFoundError(target)
        nvidia.__path__.insert(0, str(target))
    dll = runtime / "nvidia" / "libnvcomp" / "bin"
    if os.name == "nt":
        _DLL_HANDLES.append(os.add_dll_directory(str(dll)))
    from nvidia import nvcomp

    if tuple(int(x) for x in nvcomp.__version__.split(".")[:2]) < (5, 3):
        raise RuntimeError(f"nvCOMP 5.3+ Gzip compression required, found {nvcomp.__version__}")
    return nvcomp


def load_deflate(deps: Path):
    sys.path.insert(0, str(deps))
    import deflate

    return deflate


def heatsoak(device: int, seconds: float) -> None:
    if seconds <= 0:
        return
    import cupy as cp

    with cp.cuda.Device(device):
        a = cp.random.random((2048, 2048), dtype=cp.float32)
        b = cp.random.random((2048, 2048), dtype=cp.float32)
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            cp.matmul(a, b)
            cp.cuda.get_current_stream().synchronize()


def gpu_encode(nvcomp, codec, payload: bytes, device: int) -> tuple[bytes, dict]:
    import cupy as cp
    import numpy as np

    source = np.frombuffer(payload, dtype=np.uint8)
    with cp.cuda.Device(device):
        start = time.perf_counter()
        on_device = nvcomp.as_array(source).cuda()
        h2d = time.perf_counter() - start  # cuda() synchronizes by default.
        start = time.perf_counter()
        encoded = codec.encode(on_device)
        cp.cuda.runtime.deviceSynchronize()
        encode = time.perf_counter() - start
        start = time.perf_counter()
        member = bytes(encoded.cpu())
        d2h = time.perf_counter() - start
    return member, {"h2d_seconds": h2d, "encode_seconds": encode,
                    "d2h_seconds": d2h, "total_seconds": h2d + encode + d2h}


def gpu_batched_deflate_encode(nvcomp, codec, payload: bytes, device: int,
                               chunk_bytes: int = 64 << 10,
                               validate_members: bool = False) -> tuple[bytes, dict]:
    """Stage one group, encode 64-KiB RAW streams together, frame ordered members."""
    import cupy as cp
    import numpy as np

    host = np.frombuffer(payload, dtype=np.uint8)
    host_view = memoryview(host)
    sizes = [min(chunk_bytes, len(payload) - offset)
             for offset in range(0, len(payload), chunk_bytes)]
    with cp.cuda.Device(device):
        start = time.perf_counter()
        staged = cp.asarray(host)
        cp.cuda.get_current_stream().synchronize()
        h2d = time.perf_counter() - start
        # Every nvcomp.Array keeps its CuPy view alive until the batch finishes.
        start = time.perf_counter()
        views = [nvcomp.as_array(staged[offset:offset + length])
                 for offset, length in ((i * chunk_bytes, size) for i, size in enumerate(sizes))]
        views_seconds = time.perf_counter() - start
        start = time.perf_counter()
        encoded = codec.encode(views)
        cp.cuda.runtime.deviceSynchronize()
        encode = time.perf_counter() - start
        start = time.perf_counter()
        raw_streams = [bytes(item.cpu()) for item in encoded]
        d2h = time.perf_counter() - start
    if len(raw_streams) != len(sizes):
        raise AssertionError("nvCOMP batch returned the wrong number of streams")
    if validate_members:
        for i, (stream, length) in enumerate(zip(raw_streams, sizes)):
            reference = host_view[i * chunk_bytes:i * chunk_bytes + length]
            if zlib.decompress(stream, -15) != reference.tobytes():
                raise AssertionError(f"Batched raw DEFLATE member {i} failed RFC 1951 decode")
    start = time.perf_counter()
    members = [frame_raw_deflate(stream, host_view[i * chunk_bytes:i * chunk_bytes + size])
               for i, (stream, size) in enumerate(zip(raw_streams, sizes))]
    framed = b"".join(members)
    frame = time.perf_counter() - start
    if validate_members:
        for i, (member, length) in enumerate(zip(members, sizes)):
            reference = host_view[i * chunk_bytes:i * chunk_bytes + length]
            if gzip.decompress(member) != reference.tobytes():
                raise AssertionError(f"Batched gzip-framed member {i} failed RFC 1952 decode")
    return framed, {"h2d_seconds": h2d, "views_seconds": views_seconds,
                    "encode_seconds": encode,
                    "d2h_seconds": d2h, "frame_seconds": frame,
                    "total_seconds": h2d + views_seconds + encode + d2h + frame,
                    "physical_members": len(members)}


def _measure(samples: list[float], size: int) -> dict:
    midpoint = statistics.median(samples)
    return {"median_seconds": midpoint, "mib_per_second": size / (1 << 20) / midpoint,
            "samples_seconds": samples}


def canonical_zero_member(length: int) -> bytes:
    raw = zlib.compressobj(1, zlib.DEFLATED, -15)
    data = bytes(length)
    return (b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"
            + raw.compress(data) + raw.flush(zlib.Z_FINISH)
            + struct.pack("<II", zlib.crc32(data), length & 0xFFFFFFFF))


def frame_raw_deflate(raw_deflate: bytes, payload: bytes) -> bytes:
    """Turn a standard RFC 1951 stream into one RFC 1952 gzip member."""
    return (b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"
            + raw_deflate
            + struct.pack("<II", zlib.crc32(payload), len(payload) & 0xFFFFFFFF))


def benchmark(manifest: Path, output: Path, deps: Path, runtime: Path,
              deflate_deps: Path, device: int, repeats: int, heatsoak_seconds: float) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    samples = [(entry["name"], Path(entry["path"]).read_bytes()) for entry in data["samples"]]
    if any(len(raw) < max(SIZES) for _, raw in samples):
        raise ValueError("Every selected real sample must contain at least 16 MiB")
    with GpuLock():
        # Nothing above this point imports a CUDA binding or creates a context.
        nvcomp = load_nvcomp(deps, runtime)
        deflate = load_deflate(deflate_deps)
        heatsoak(device, heatsoak_seconds)
        codec = nvcomp.Codec(algorithm="Gzip", device_id=device,
                             bitstream_kind=nvcomp.BitstreamKind.RAW,
                             algorithm_type=1)
        deflate_codec = nvcomp.Codec(algorithm="Deflate", device_id=device,
                                     bitstream_kind=nvcomp.BitstreamKind.RAW,
                                     algorithm_type=1)
        results = []
        mixed_members: list[bytes] = []
        mixed_raw: list[bytes] = []
        raw_deflate_compatible = True
        raw_deflate_failure = ""
        for name, raw in samples:
            for size in SIZES:
                payload = raw[:size]
                # GPU warmup and a complete standard-gzip proof precede timing.
                gpu_member, _ = gpu_encode(nvcomp, codec, payload, device)
                if gzip.decompress(gpu_member) != payload:
                    raise AssertionError(f"nvCOMP RAW Gzip failed standard decode: {name}/{size}")
                cpu_member = bytes(deflate.gzip_compress(payload, 3))
                if gzip.decompress(cpu_member) != payload:
                    raise AssertionError(f"libdeflate Gzip failed standard decode: {name}/{size}")
                cpu_times = []
                gpu_phases = []
                deflate_phases = []
                deflate_member = b""
                if raw_deflate_compatible:
                    raw_stream, _ = gpu_encode(nvcomp, deflate_codec, payload, device)
                    try:
                        if zlib.decompress(raw_stream, -15) != payload:
                            raise AssertionError("nvCOMP Deflate RAW bytes differ")
                        if gzip.decompress(frame_raw_deflate(raw_stream, payload)) != payload:
                            raise AssertionError("nvCOMP raw Deflate gzip framing failed")
                    except Exception as exc:
                        raw_deflate_compatible = False
                        raw_deflate_failure = f"{type(exc).__name__}: {exc}"
                for _ in range(repeats):
                    start = time.perf_counter()
                    cpu_member = bytes(deflate.gzip_compress(payload, 3))
                    cpu_times.append(time.perf_counter() - start)
                    gpu_member, phases = gpu_encode(nvcomp, codec, payload, device)
                    gpu_phases.append(phases)
                    if raw_deflate_compatible:
                        raw_stream, phases = gpu_encode(nvcomp, deflate_codec, payload, device)
                        start = time.perf_counter()
                        deflate_member = frame_raw_deflate(raw_stream, payload)
                        phases["frame_seconds"] = time.perf_counter() - start
                        phases["total_seconds"] += phases["frame_seconds"]
                        deflate_phases.append(phases)
                if gzip.decompress(gpu_member) != payload:
                    raise AssertionError("Timed GPU member failed standard decode")
                if raw_deflate_compatible and gzip.decompress(deflate_member) != payload:
                    raise AssertionError("Timed GPU raw Deflate member failed standard decode")
                record = {"sample": name, "input_bytes": size,
                          "cpu_encoded_bytes": len(cpu_member),
                          "gpu_gzip_encoded_bytes": len(gpu_member),
                          "cpu": _measure(cpu_times, size),
                          "gpu_gzip": _measure([x["total_seconds"] for x in gpu_phases], size),
                          "gpu_gzip_phases": {key: statistics.median(x[key] for x in gpu_phases)
                                              for key in ("h2d_seconds", "encode_seconds", "d2h_seconds")}}
                if raw_deflate_compatible:
                    record["gpu_raw_deflate_encoded_bytes"] = len(deflate_member)
                    record["gpu_raw_deflate"] = _measure(
                        [x["total_seconds"] for x in deflate_phases], size)
                    record["gpu_raw_deflate_phases"] = {
                        key: statistics.median(x[key] for x in deflate_phases)
                        for key in ("h2d_seconds", "encode_seconds", "d2h_seconds", "frame_seconds")
                    }
                results.append(record)
                if size == 1 << 20:
                    mixed_members.extend((cpu_member, gpu_member))
                    mixed_raw.extend((payload, payload))
                    if raw_deflate_compatible:
                        mixed_members.append(deflate_member)
                        mixed_raw.append(payload)
        zero_size = 64 << 10
        zero = canonical_zero_member(zero_size)
        # Existing writer bypasses the compressor for known zero spans; exercise
        # 128 ordered cached-zero members alongside both actual codecs.
        mixed_members.extend([zero] * 128)
        mixed_raw.extend([bytes(zero_size)] * 128)
        mixed = b"".join(mixed_members)
        expected = b"".join(mixed_raw)
        if gzip.decompress(mixed) != expected:
            raise AssertionError("Mixed CPU/GPU/zero gzip-member sequence failed")
        import nrrd
        import numpy as np

        output.mkdir(parents=True, exist_ok=True)
        nrrd_path = output / "mixed-members.seg.nrrd"
        nrrd_path.write_bytes(
            b"NRRD0005\ntype: unsigned char\ndimension: 3\nsizes: 1 1 "
            + str(len(expected)).encode("ascii") + b"\nencoding: gzip\n\n" + mixed
        )
        # The project's decoder accepts concatenated gzip members and checks
        # their CRCs through GzipFile. Use the same binary-slab path as production.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from XTA.reconciliation_io import FileLayer

        with nrrd_path.open("rb") as fh:
            project_header = nrrd.read_header(fh)
            payload_offset = fh.tell()
        layer = FileLayer("mixed", {}, (len(expected), 1, 1), nrrd_path,
                          project_header, (len(expected), 1, 1), (0, 0, 0),
                          payload_offset, "gzip", False)
        layer._owner = SimpleNamespace(_closed=False, _open_count=0,
                                       max_open=1, _chunk_bytes=1 << 20,
                                       _budget_bytes=len(expected))
        try:
            project_decoded = layer.read_slab(0, len(expected)).reshape(-1)
            if not np.array_equal(project_decoded, np.frombuffer(expected, dtype=np.uint8)):
                raise AssertionError("Project FileLayer decoded different bytes")
        finally:
            layer.close()
        pynrrd_ok = False
        pynrrd_error = ""
        try:
            decoded, _header = nrrd.read(str(nrrd_path))
            pynrrd_ok = bool(np.array_equal(decoded, np.frombuffer(expected, dtype=np.uint8)))
            if not pynrrd_ok:
                pynrrd_error = "pynrrd decoded different bytes"
        except Exception as exc:
            pynrrd_error = f"{type(exc).__name__}: {exc}"
        report = {"nvcomp_version": nvcomp.__version__, "nvcomp_cuda_version": nvcomp.__cuda_version__,
                  "device_index": device, "heatsoak_seconds": heatsoak_seconds,
                  "repeats": repeats, "source": data["source"], "results": results,
                  "raw_deflate_gzip_compatible": raw_deflate_compatible,
                  "raw_deflate_failure": raw_deflate_failure,
                  "mixed_member_count": len(mixed_members),
                  "mixed_raw_bytes": len(expected), "mixed_nrrd": str(nrrd_path),
                  "standard_gzip_roundtrip": True,
                  "project_reader_roundtrip": True,
                  "pynrrd_roundtrip": pynrrd_ok, "pynrrd_error": pynrrd_error}
        (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def benchmark_batched(manifest: Path, output: Path, deps: Path, runtime: Path,
                      deflate_deps: Path, device: int, repeats: int,
                      heatsoak_seconds: float) -> dict:
    """Compare 256-way 64-KiB RAW Deflate with one 16-MiB Gzip member."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    size = 16 << 20
    samples = [(entry["name"], Path(entry["path"]).read_bytes()[:size])
               for entry in data["samples"]]
    if any(len(raw) != size for _, raw in samples):
        raise ValueError("The batched comparison requires 16 MiB per sample")
    with GpuLock():
        nvcomp = load_nvcomp(deps, runtime)
        deflate = load_deflate(deflate_deps)
        heatsoak(device, heatsoak_seconds)
        gzip_codec = nvcomp.Codec(algorithm="Gzip", device_id=device,
                                  bitstream_kind=nvcomp.BitstreamKind.RAW,
                                  algorithm_type=1)
        raw_codec = nvcomp.Codec(algorithm="Deflate", device_id=device,
                                 bitstream_kind=nvcomp.BitstreamKind.RAW,
                                 algorithm_type=1)
        rows = []
        mixed_members = []
        mixed_payload = []
        for name, payload in samples:
            cpu_member = bytes(deflate.gzip_compress(payload, 3))
            gpu_gzip, _ = gpu_encode(nvcomp, gzip_codec, payload, device)
            if gzip.decompress(cpu_member) != payload or gzip.decompress(gpu_gzip) != payload:
                raise AssertionError("Baseline gzip member validation failed")
            row = {"sample": name, "input_bytes": size,
                   "cpu_encoded_bytes": len(cpu_member),
                   "gpu_gzip_encoded_bytes": len(gpu_gzip)}
            cpu_times = []
            gzip_phases = []
            batch_phases = []
            try:
                batch_member, _ = gpu_batched_deflate_encode(
                    nvcomp, raw_codec, payload, device, validate_members=True)
                if gzip.decompress(batch_member) != payload:
                    raise AssertionError("Batched RAW Deflate failed complete standard gzip decode")
                for _ in range(repeats):
                    start = time.perf_counter()
                    cpu_member = bytes(deflate.gzip_compress(payload, 3))
                    cpu_times.append(time.perf_counter() - start)
                    gpu_gzip, phases = gpu_encode(nvcomp, gzip_codec, payload, device)
                    gzip_phases.append(phases)
                    batch_member, phases = gpu_batched_deflate_encode(
                        nvcomp, raw_codec, payload, device)
                    batch_phases.append(phases)
                if gzip.decompress(batch_member) != payload:
                    raise AssertionError("Timed batch failed complete standard gzip decode")
                # A deliberate NumPy-backed CRC/ISIZE check follows the complete
                # 256-member decode; individual streams remain standard gzip.
                iterator = memoryview(np_frombuffer(payload))
                if zlib.crc32(iterator) != zlib.crc32(payload):
                    raise AssertionError("NumPy memoryview CRC differs from bytes")
                row["batched_raw_deflate_64k"] = _measure(
                    [x["total_seconds"] for x in batch_phases], size)
                row["batched_raw_deflate_64k_phases"] = {
                    key: statistics.median(x[key] for x in batch_phases)
                    for key in ("h2d_seconds", "views_seconds", "encode_seconds",
                                "d2h_seconds", "frame_seconds")
                }
                row["batched_raw_deflate_64k_encoded_bytes"] = len(batch_member)
                row["batched_raw_deflate_64k_members"] = 256
                mixed_members.extend((cpu_member, gpu_gzip, batch_member))
                mixed_payload.extend((payload, payload, payload))
                row["batched_valid"] = True
            except Exception as exc:
                row["batched_valid"] = False
                row["batched_error"] = f"{type(exc).__name__}: {exc}"
            if not cpu_times:
                for _ in range(repeats):
                    start = time.perf_counter()
                    deflate.gzip_compress(payload, 3)
                    cpu_times.append(time.perf_counter() - start)
                    _member, phases = gpu_encode(nvcomp, gzip_codec, payload, device)
                    gzip_phases.append(phases)
            row["cpu"] = _measure(cpu_times, size)
            row["gpu_gzip"] = _measure([x["total_seconds"] for x in gzip_phases], size)
            row["gpu_gzip_phases"] = {
                key: statistics.median(x[key] for x in gzip_phases)
                for key in ("h2d_seconds", "encode_seconds", "d2h_seconds")
            }
            rows.append(row)
        zero = canonical_zero_member(64 << 10)
        mixed_members.extend([zero] * 128)
        mixed_payload.extend([bytes(64 << 10)] * 128)
        mixed = b"".join(mixed_members)
        expected = b"".join(mixed_payload)
        full_gzip_ok = gzip.decompress(mixed) == expected
        if not full_gzip_ok:
            raise AssertionError("Batched mixed standard gzip sequence failed")
        output.mkdir(parents=True, exist_ok=True)
        mixed_path = output / "batched-mixed.seg.nrrd"
        mixed_path.write_bytes(
            b"NRRD0005\ntype: unsigned char\ndimension: 3\nsizes: 1 1 "
            + str(len(expected)).encode("ascii") + b"\nencoding: gzip\n\n" + mixed)
        import nrrd
        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from XTA.reconciliation_io import FileLayer

        with mixed_path.open("rb") as fh:
            header = nrrd.read_header(fh)
            payload_offset = fh.tell()
        layer = FileLayer("batched-mixed", {}, (len(expected), 1, 1), mixed_path,
                          header, (len(expected), 1, 1), (0, 0, 0),
                          payload_offset, "gzip", False)
        layer._owner = SimpleNamespace(_closed=False, _open_count=0,
                                       max_open=1, _chunk_bytes=1 << 20,
                                       _budget_bytes=len(expected))
        try:
            decoded = layer.read_slab(0, len(expected)).reshape(-1)
            if not np.array_equal(decoded, np.frombuffer(expected, dtype=np.uint8)):
                raise AssertionError("Project reader decoded different batched bytes")
        finally:
            layer.close()
        physical_members = 128 + sum(258 for row in rows if row["batched_valid"])
        report = {"nvcomp_version": nvcomp.__version__, "device": device,
                  "heatsoak_seconds": heatsoak_seconds, "repeats": repeats,
                  "raw_source": data["source"], "rows": rows,
                  "mixed_member_groups": len(mixed_members),
                  "mixed_physical_gzip_members": physical_members,
                  "mixed_raw_bytes": len(expected), "mixed_standard_gzip_roundtrip": full_gzip_ok,
                  "mixed_project_reader_roundtrip": True,
                  "mixed_nrrd": str(mixed_path)}
        (output / "batched-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def np_frombuffer(payload: bytes):
    import numpy as np

    return np.frombuffer(payload, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, metavar="SOURCE_NRRD")
    parser.add_argument("--benchmark", type=Path, metavar="SAMPLES_JSON")
    parser.add_argument("--benchmark-batched", type=Path, metavar="SAMPLES_JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-mib", type=int, default=256)
    parser.add_argument("--deps", type=Path, default=EXPERIMENT / "deps")
    parser.add_argument("--runtime", type=Path, default=EXPERIMENT / "runtime")
    parser.add_argument("--deflate-deps", type=Path, default=DEFAULT_DEFLATE)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--heatsoak-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if sum(map(bool, (args.prepare, args.benchmark, args.benchmark_batched))) != 1:
        parser.error("Select exactly one of --prepare, --benchmark, or --benchmark-batched")
    if args.repeats < 1 or args.heatsoak_seconds < 0:
        parser.error("--repeats must be positive and --heatsoak-seconds nonnegative")
    if args.prepare:
        result = prepare(args.prepare, args.output, args.scan_mib)
    elif args.benchmark_batched:
        result = benchmark_batched(args.benchmark_batched, args.output, args.deps, args.runtime,
                                   args.deflate_deps, args.device, args.repeats,
                                   args.heatsoak_seconds)
    else:
        result = benchmark(args.benchmark, args.output, args.deps, args.runtime,
                           args.deflate_deps, args.device, args.repeats, args.heatsoak_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
