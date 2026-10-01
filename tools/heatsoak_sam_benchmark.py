"""Bounded CPU/GPU heatsoak before local SAM scheduling sanity benchmarks."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=180)
    p.add_argument("--device",type=int,default=0)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("diagnostic",Path(__file__).parent/"diagnose_sam_interpolation.py")
    diagnostic=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    stop=threading.Event()
    counts={"cpu_matmuls":0,"gpu_matmuls":0}
    with diagnostic.gpu_lock(Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"),"sam_improvements_heatsoak"),diagnostic.resource_monitor(args.output/"resources.json",args.device):
        os.environ["OPENBLAS_NUM_THREADS"]="8"
        os.environ["MKL_NUM_THREADS"]="8"
        os.environ["OMP_NUM_THREADS"]="8"
        import numpy as np
        import torch
        import pynvml
        torch.cuda.set_device(args.device)
        def cpu():
            a=np.full((1536,1536),.001,np.float32)
            b=np.full((1536,1536),.002,np.float32)
            out=np.empty_like(a)
            while not stop.is_set():
                np.matmul(a,b,out=out)
                counts["cpu_matmuls"]+=1
        thread=threading.Thread(target=cpu,daemon=True)
        thread.start()
        a=torch.full((8192,8192),.001,device=args.device,dtype=torch.float16)
        b=torch.full_like(a,.002)
        out=torch.empty_like(a)
        handle=pynvml.nvmlDeviceGetHandleByIndex(args.device)
        snapshots=[]
        start=time.perf_counter()
        next_sample=0.
        try:
            while time.perf_counter()-start<args.seconds:
                torch.matmul(a,b,out=out)
                counts["gpu_matmuls"]+=1
                elapsed=time.perf_counter()-start
                if elapsed>=next_sample:
                    torch.cuda.synchronize()
                    row={"elapsed_seconds":elapsed,"gpu_temperature_c":pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU),
                         "gpu_clock_mhz":pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_GRAPHICS),
                         "gpu_power_w":pynvml.nvmlDeviceGetPowerUsage(handle)/1000}
                    snapshots.append(row)
                    print(f"heatsoak {row}",flush=True)
                    next_sample=elapsed+10
        finally:
            stop.set()
            thread.join(timeout=30)
            torch.cuda.synchronize()
        report={"seconds":time.perf_counter()-start,"counts":counts,"thermal_snapshots":snapshots,
                "method":"Concurrent 8-thread float32 CPU matmul and FP16 CUDA matmul; separate process before measured model work",
                "claim":"Local device preparation only; not a SAM benchmark or target-system performance estimate"}
        (args.output/"heatsoak.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
