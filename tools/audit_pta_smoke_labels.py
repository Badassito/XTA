"""Inspect YOLO polygons; write inventories and montages to an explicit external directory.

INPUT_ROOT must contain images/ and labels/. Volume grouping keeps the original
F2/F3 convention (first two filename characters). All pairs enter the inventory.
Up to --montage-limit samples per group enter overview montages; the first such
samples also enter a contrast/detail montage. Repeat --sample-name to choose
examples explicitly instead of relying on the sorted sample order.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

if __package__:
    from .pta_audit_common import external_output, receipt_child, write_json
else:
    from pta_audit_common import external_output, receipt_child, write_json


def load_pair(path: Path, labels: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot decode input image: {path}')
    height, width = image.shape
    label = labels / (path.stem + '.txt')
    polygons = []
    for number, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < 7 or len(tokens) % 2 != 1:
            raise ValueError(f'Expected class plus at least three XY vertices: {label}:{number}')
        xy = np.array(tokens[1:], dtype=float).reshape(-1, 2)
        if not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
            raise ValueError(f'Polygon coordinates must be finite and normalized: {label}:{number}')
        polygons.append(np.rint(xy * [width, height]).astype(np.int32))
    mask = np.zeros_like(image)
    # Fill separately: simultaneous fillPoly can apply even/odd overlap rules.
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 1)
    return image, mask, polygons


def full_overlay(image, polygons):
    import cv2

    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(result, polygons, True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    return result


def _write_image(path, image):
    import cv2

    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write diagnostic image: {path}')


def make_montage(paths, labels, output, filename, font, *, detail=False):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    side, cell_height = (512, 560) if detail else (384, 480)
    canvas = Image.new('RGB', (side * 6, ((len(paths) + 2) // 3) * cell_height), '#222222')
    drawer = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        image, mask, polygons = load_pair(path, labels)
        if detail:
            yy, xx = np.nonzero(mask)
            if len(xx):
                x0, x1 = max(0, int(xx.min()) - 70), min(image.shape[1], int(xx.max()) + 71)
                y0, y1 = max(0, int(yy.min()) - 70), min(image.shape[0], int(yy.max()) + 71)
            else:
                x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
            crop = image[y0:y1, x0:x1]
            positive = crop[crop > 0]
            lo, hi = np.percentile(positive, [1, 99.8]) if positive.size else (0, 0)
            image = np.asarray(np.clip((crop.astype(float) - lo) * 255 / max(1, hi - lo), 0, 255), np.uint8)
            polygons = [polygon - [x0, y0] for polygon in polygons]
            # Retain the original raw contrast crop and its outline separately.
            _write_image(receipt_child(output, path.stem + '_contrast_crop.png'), image)
        overlay = full_overlay(image, polygons)
        suffix = '_contrast_crop_outline.png' if detail else '_outline.png'
        _write_image(receipt_child(output, path.stem + suffix), overlay)
        x, y = (index % 3) * side * 2, (index // 3) * cell_height
        canvas.paste(Image.fromarray(image).convert('RGB').resize((side, side)), (x, y))
        canvas.paste(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).resize((side, side)), (x + side, y))
        short = path.stem.split('_8bit_Y_', 1)[-1]
        drawer.text((x + 4, y + side + 4), path.stem[:2] + ' ' + short[:65], font=font, fill='white')
        drawer.text((x + 4, y + side + 27), f'raw | green outline; {int(mask.sum()):,} px; {len(polygons)} polygons', font=font, fill='white')
    canvas.save(receipt_child(output, filename))


def audit(input_root: Path, output_dir: Path, *, montage_limit: int = 6,
          sample_names: tuple[str, ...] = (), font_path: Path | None = None):
    import cv2
    import numpy as np
    from PIL import ImageFont

    input_root = Path(input_root).expanduser().resolve()
    output = external_output(output_dir, inputs=(input_root,))
    if montage_limit < 1:
        raise ValueError('montage-limit must be positive')
    images, labels = input_root / 'images', input_root / 'labels'
    if not images.is_dir() or not labels.is_dir():
        raise ValueError('input-root must contain images/ and labels/ directories')
    paths = sorted(p for p in images.iterdir() if p.is_file()
                   and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if not paths:
        raise ValueError(f'No supported input images: {images}')
    if len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError('Image stems must be unique, including case')
    missing = set(sample_names) - {p.stem for p in paths}
    if missing:
        raise ValueError(f'Unknown sample names: {sorted(missing)}')
    rows, groups = [], defaultdict(list)
    for path in paths:
        image, mask, polygons = load_pair(path, labels)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
        augmented = bool(re.search(r'_[A-Za-z0-9]{16}$', path.stem))
        rows.append(dict(name=path.stem, volume=path.stem[:2], augmented=augmented,
                         height=image.shape[0], width=image.shape[1],
                         polygon_count=len(polygons), foreground_pixels=int(mask.sum()),
                         components8=count - 1, component_areas_descending=areas,
                         components_le_25=sum(a <= 25 for a in areas),
                         pixels_in_components_le_25=sum(a for a in areas if a <= 25)))
        if not sample_names or path.stem in sample_names:
            groups[(path.stem[:2], augmented)].append(path)
    font = ImageFont.truetype(str(font_path), 17) if font_path else ImageFont.load_default()
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    for index, (_, group) in enumerate(sorted(groups.items()), 1):
        examples = group[:montage_limit]
        selected.extend(examples)
        make_montage(examples, labels, output, f'smoke_group_{index}_montage.png', font)
    make_montage(selected[:montage_limit], labels, output, 'smoke_contrast_detail_montage.png', font, detail=True)
    write_json(output, 'smoke_label_component_inventory.json', rows)
    summary = []
    for volume in sorted({row['volume'] for row in rows}):
        group = [row for row in rows if row['volume'] == volume]
        summary.append(dict(volume=volume, images=len(group),
                            augmented=sum(row['augmented'] for row in group),
                            component_median=float(np.median([row['components8'] for row in group])),
                            small_component_pixels=sum(row['pixels_in_components_le_25'] for row in group)))
    write_json(output, 'smoke_label_summary.json', summary)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--montage-limit', type=int, default=6)
    parser.add_argument('--sample-name', action='append', default=[])
    parser.add_argument('--font', type=Path, help='Optional TrueType font; default is Pillow built-in font')
    args = parser.parse_args(argv)
    rows = audit(args.input_root, args.output_dir, montage_limit=args.montage_limit,
                 sample_names=tuple(args.sample_name), font_path=args.font)
    print(json.dumps({'images': len(rows), 'output_dir': str(args.output_dir.resolve())}))


if __name__ == '__main__':
    main()
