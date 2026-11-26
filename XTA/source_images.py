"""Lossless source-grid image sequences for TTA output."""

from pathlib import Path

import numpy as np

from ._deps import cv2


def write_source_images_from_pattern(
    volume_gray: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    """Write one grayscale PNG per frame of the selected label grid."""
    # Share naming, staged publication and retry behavior with the label writer.
    from .outputs import (
        _format_frame_path,
        _publish_staged_frame_sequence,
        _staged_frame_sequence,
        choose_slice_parallel_workers,
        parallel_for_indices,
    )

    frames = np.asarray(volume_gray)
    if frames.ndim != 3 or not all(frames.shape) or frames.dtype != np.uint8:
        raise ValueError('Source images require a nonempty uint8 T/Y/X volume')
    pattern_path = Path(pattern_path)
    if pattern_path.suffix.lower() != '.png':
        raise ValueError('Source images require a .png frame pattern')

    total = int(frames.shape[0])
    with _staged_frame_sequence(pattern_path) as stage_pattern:
        def write_frame(index: int) -> None:
            path = _format_frame_path(stage_pattern, index + 1)
            if not cv2.imwrite(str(path), np.ascontiguousarray(frames[index])):
                raise RuntimeError(f'Failed to write source image: {path}')

        parallel_for_indices(
            total,
            write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc='Writing source-grid images',
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern, pattern_path, total, stale_extensions=('.png',),
        )
    return pattern_path.parent
