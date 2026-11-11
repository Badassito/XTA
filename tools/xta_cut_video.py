"""Lossless grayscale MKV half-space cuts for the paired XTA volume utility."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    pixel_format: str
    timing_sha256: str


def _probe(path, *arguments):
    result = subprocess.run(
        ["ffprobe", "-v", "error", *arguments, "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def probe_video(path):
    """Read geometry and exact FFV1 packet timing without decoding pixels.

    FFV1 contains one complete frame in each packet. The packet scan supplies
    both a frame count for MKVs without nb_frames and a timing identity used
    to verify passthrough encoding.
    """
    result = _probe(path, "-show_streams")
    streams = result.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise ValueError("The intensity MKV must contain exactly one video stream.")
    stream = streams[0]
    if stream.get("codec_name") != "ffv1" or stream.get("pix_fmt") != "gray":
        raise ValueError("The intensity MKV must be lossless FFV1 with gray8 pixels.")
    if stream.get("sample_aspect_ratio", "1:1") not in {"1:1", "N/A"}:
        raise ValueError("The intensity MKV must use square pixels.")
    if stream.get("color_range") not in {None, "unknown", "pc"}:
        raise ValueError("The intensity MKV must use full-range grayscale samples.")
    for side_data in stream.get("side_data_list", []):
        if "displaymatrix" in side_data or "rotation" in side_data:
            raise ValueError("The intensity MKV contains display orientation metadata.")
    if "rotate" in stream.get("tags", {}):
        raise ValueError("The intensity MKV contains a display rotation tag.")
    packets = _probe(
        path, "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts,duration",
    ).get("packets", [])
    if not packets:
        raise ValueError("The intensity MKV contains no video frames.")
    time_base = Fraction(stream["time_base"])
    timing = hashlib.sha256()
    previous = None
    for packet in packets:
        if "pts" not in packet:
            raise ValueError("An intensity frame has no presentation timestamp.")
        timestamp = int(packet["pts"]) * time_base
        if previous is not None and timestamp <= previous:
            raise ValueError("Intensity presentation timestamps must increase strictly.")
        duration = int(packet.get("duration", 0)) * time_base
        timing.update(f"{timestamp}|{duration}\n".encode("ascii"))
        previous = timestamp
    frames = len(packets)
    declared = stream.get("nb_frames")
    if declared not in {None, "N/A"} and int(declared) != frames:
        raise ValueError("The intensity frame count differs from its declared count.")
    return VideoInfo(
        width=int(stream["width"]), height=int(stream["height"]),
        frames=frames, pixel_format=stream["pix_fmt"],
        timing_sha256=timing.hexdigest(),
    )


def write_video(source, destination, info, first_removed, mode):
    """Keep columns below first_removed and optionally pad the rest with zero.

    The caller supplies a new staging destination. The written file is probed
    before return, and a failed validation leaves it available for diagnosis.
    """
    if mode not in {"zero", "crop"}:
        raise ValueError("Video cut mode must be zero or crop.")
    if isinstance(first_removed, bool) or not isinstance(first_removed, int):
        raise ValueError("The first removed video column must be an integer.")
    if not 0 < first_removed < info.width:
        raise ValueError("The cut must retain and remove at least one video column.")
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Video destination already exists: {destination}")
    filter_graph = f"crop=w={first_removed}:h=ih:x=0:y=0:exact=1"
    if mode == "zero":
        filter_graph += f",pad=w={info.width}:h=ih:x=0:y=0:color=black"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-nostdin", "-n",
        "-copyts", "-noautorotate", "-i", str(source),
        "-map", "0:v:0", "-vf", filter_graph,
        "-fps_mode", "passthrough", "-c:v", "ffv1", "-level", "3",
        "-pix_fmt", "gray", "-color_range", "pc", "-map_metadata", "0",
        "-f", "matroska", str(destination),
    ]
    subprocess.run(command, check=True)
    written = probe_video(destination)
    expected_width = first_removed if mode == "crop" else info.width
    if (written.width, written.height, written.frames) != (
        expected_width, info.height, info.frames,
    ):
        raise ValueError("Written intensity dimensions or frame count differ from the cut plan.")
    if written.timing_sha256 != info.timing_sha256:
        raise ValueError("Written intensity frame timestamps or durations differ from the source.")
    return {
        "codec": "ffv1", "pixel_format": written.pixel_format,
        "shape_tyx": [written.frames, written.height, written.width],
        "timing_sha256": written.timing_sha256,
        "timing_verified": True, "command": command,
    }
