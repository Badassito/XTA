"""Measure strict nvJPEG lane counts; caller owns Scratch/Temp/GPU_LOCK."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    scratch = ROOT.parent / "Scratch"
    args.output = args.output.resolve()
    if not args.output.is_relative_to(scratch.resolve()) or args.output.exists():
        parser.error("--output must be a fresh path under sibling Scratch")
    if not 32 <= args.size <= 3072 or not 1 <= args.batch <= 64 or args.repetitions < 1:
        parser.error("Require size32..3072, batch1..64, repetitions>=1")
    lock = scratch / "Temp/GPU_LOCK"
    if not lock.is_file() or not lock.read_text(encoding="utf-8").strip():
        parser.error("Caller must own the populated Scratch/Temp/GPU_LOCK")
    args.output.mkdir(parents=True)
    import numpy as np
    import torch
    from nvidia import nvimgcodec
    from XTA import pta_publication as publication, pta_workers as workers
    from tools.qualify_pta_gpu_publication import heatsoak

    report = dict(status="running", scope="Warm native gray JPEG encoding only; quality100; no source rendering, labels or disk publication.",
                  gpu_claim=lock.read_text(encoding="utf-8"), device=torch.cuda.get_device_name(args.device),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  nvimgcodec_version=getattr(nvimgcodec, "__version__", "unknown"),
                  size=args.size, batch=args.batch, repetitions=args.repetitions, results=[])
    destination = args.output / "lanes.json"
    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    stream = torch.cuda.Stream(device=args.device)
    encoder = images = owners = None
    try:
        stop = threading.Event()
        def warm_cpu():
            data = b"PTA codec CPU warmup" * 100_000
            while not stop.is_set():
                hashlib.sha256(data).digest()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(warm_cpu) for _ in range(4)]
            try:
                report["heatsoak"] = heatsoak(torch, f"cuda:{args.device}", 60.)
            finally:
                stop.set()
                for future in futures:
                    future.result()
        report["heatsoak"]["cpu_threads"] = 4
        y, x = np.indices((args.size, args.size), dtype=np.int32)
        host = np.stack([((x * 13 + y * 7 + i * 37) % 256).astype(np.uint8)[..., None]
                         for i in range(args.batch)])
        with torch.cuda.device(args.device), torch.cuda.stream(stream):
            owners = torch.as_tensor(host).to(f"cuda:{args.device}")
            paths = [args.output / f"{index}.jpg" for index in range(args.batch)]
            reference = None
            for lanes in (1, 2, 4, 8, 16):
                encoder = nvimgcodec.Encoder(device_id=args.device, max_num_cpu_threads=lanes,
                    backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)],
                    options=f":num_cuda_streams={lanes}")
                samples = [owners[index] for index in range(args.batch)]
                images = workers._nvjpeg_wrap_samples(nvimgcodec, samples, channel_kind="gray", cuda_stream=int(stream.cuda_stream))
                kwargs = dict(encoder=encoder, images=images, final_paths=paths,
                              params=workers._nvjpeg_encode_params(nvimgcodec, quality=100, channel_kind="gray"),
                              cuda_stream=stream, input_owners=[owners])
                reusable = [nvimgcodec.CodeStream() for _ in range(args.batch)]
                reuse_encoder = SimpleNamespace(encode=lambda *args_, **kwargs_: encoder.encode(
                    *args_, **kwargs_, code_stream_s=reusable))
                for mode, target in (("fresh", encoder), ("reuse", reuse_encoder)):
                    kwargs["encoder"] = target
                    publication._encode_nvjpeg_batch(**kwargs, synchronize_device=lambda: torch.cuda.synchronize(args.device))
                    times = []
                    for _ in range(args.repetitions):
                        started = time.perf_counter()
                        encoded = publication._encode_nvjpeg_batch(
                            **kwargs, synchronize_device=lambda: torch.cuda.synchronize(args.device))
                        times.append(time.perf_counter() - started)
                        if reference is None:
                            reference = encoded.payloads
                        if encoded.payloads != reference:
                            raise AssertionError(f"JPEG bytes changed at lane count {lanes}, mode {mode}")
                    item = dict(lanes=lanes, mode=mode, seconds=times, median_seconds=statistics.median(times),
                                images_per_second=args.batch / statistics.median(times), encoded_bytes=encoded.nbytes)
                    report["results"].append(item)
                    save()
                    print(json.dumps(item), flush=True)
                del kwargs, images, samples, reusable, reuse_encoder, target, encoder
                encoder = images = None
                gc.collect()
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Retain tensors and codec through the exceptional drain as well.
        torch.cuda.synchronize(args.device)
        save()


if __name__ == "__main__":
    main()
