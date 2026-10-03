"""Exact native window preparation for the post-v25 outer-crop study.

Full-source zero-based frame IDs are authoritative. This tool never opens
annotations, changes image pixels, or loads SAM policy/model modules.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np

def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()

def extract(ffmpeg, source, path, first, last):
    command = [str(ffmpeg), "-v", "error", "-threads", "4", "-i", str(source),
        "-vf", f"select=between(n\\,{first}\\,{last-1})", "-fps_mode", "passthrough",
        "-frames:v", str(last-first), "-pix_fmt", "gray", "-f", "rawvideo", "-y", str(path)]
    subprocess.run(command, check=True)
    return command

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--retained-crop", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", type=Path, default=Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Environment\tools\ffmpeg.exe"))
    p.add_argument("--ffprobe", type=Path, default=Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Environment\tools\ffprobe.exe"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    probe_command = [str(args.ffprobe),"-v","error","-select_streams","v:0","-show_entries",
        "stream=width,height,pix_fmt,avg_frame_rate", "-show_entries","format=duration,size", "-of","json",str(args.input)]
    probe = json.loads(subprocess.check_output(probe_command))
    stream = probe["streams"][0]
    if stream["pix_fmt"] != "gray":
        raise ValueError("Source must already be native gray; no color transformation is authorized")
    height, width = int(stream["height"]), int(stream["width"])
    records = []
    for first,last,middle,crop_frame in ((590,599,594,0),(686,695,690,96)):
        directory = args.output / f"source_{first}_{last}"
        directory.mkdir(exist_ok=True)
        path = directory / "native_images.uint8.dat"
        command = extract(args.ffmpeg,args.input,path,first,last)
        shape = (last-first,height,width)
        if path.stat().st_size != int(np.prod(shape)):
            raise ValueError("Original-source window did not contain every exact native frame")
        images = np.memmap(path,dtype=np.uint8,mode="r",shape=shape)
        comparison = directory / f"retained_crop_frame_{crop_frame}.uint8.dat"
        compare_command = extract(args.ffmpeg,args.retained_crop,comparison,crop_frame,crop_frame+1)
        if comparison.stat().st_size != height*width:
            raise ValueError("Retained crop frame has different native geometry")
        previous = np.memmap(comparison,dtype=np.uint8,mode="r",shape=(height,width))
        middle_image = images[middle-first]
        changed = int(np.count_nonzero(middle_image != previous))
        record = dict(schema="xta.full_source_outer_crop_window/1", source=str(args.input.resolve()),
            source_metadata=probe, source_size_bytes=args.input.stat().st_size,
            source_mtime_ns=args.input.stat().st_mtime_ns, window_full_source_half_open=[first,last],
            source_frame_start=first, shape_tyx=list(shape), image_path=str(path.resolve()),image_sha256=sha(path),
            frame_map=[dict(cache_local=i,full_source_frame=first+i,
                pixel_sha256=hashlib.sha256(images[i].tobytes()).hexdigest()) for i in range(len(images))],
            endpoint_full_source_frames=[first,last-1], middle_full_source_frame=middle,
            retained_crop=str(args.retained_crop.resolve()),retained_crop_frame=crop_frame,
            middle_pixel_match_exact=changed==0,middle_changed_pixels=changed,
            extraction_command=command,retained_comparison_command=compare_command,
            labels_read=False,image_transform="None: already-gray native frames selected by exact zero-based decode index",
            exposure="Seen-source follow-up: retained middle annotations were prior LTA prompts. No independent-patient or pristine heldout claim.")
        (directory/"window.json").write_text(json.dumps(record,indent=2))
        records.append(record)
        print(json.dumps(dict(window=[first,last],shape_tyx=shape,middle_exact=changed==0,changed_pixels=changed)),flush=True)
        del images,previous
    (args.output/"extraction.json").write_text(json.dumps(dict(schema="xta.outer_crop_native_windows/1",windows=records,
        source_probe_command=probe_command,source_gray_native=True,labels_read=False),indent=2))
    if any(not r["middle_pixel_match_exact"] for r in records):
        raise RuntimeError("Original versus retained middle pixels differ; report the discrepancy before comparisons")

if __name__ == "__main__":
    main()
