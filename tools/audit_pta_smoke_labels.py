"""Read-only input inspection; generated evidence is confined to this directory."""
from pathlib import Path
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('C:/Users/Bry/Documents/ChatGPT/Scratch/pta-smoke')
OUT = Path(__file__).parent
FONT = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)


def load_pair(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    label = ROOT / 'labels' / (path.stem + '.txt')
    polygons = []
    for line in label.read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 7:
            continue
        xy = np.array(tokens[1:], dtype=float).reshape(-1, 2)
        polygons.append(np.rint(xy * [width, height]).astype(np.int32))
    mask = np.zeros_like(image)
    # Fill individually: simultaneous fillPoly can apply even/odd overlap rules.
    for poly in polygons:
        cv2.fillPoly(mask, [poly], 1)
    return image, mask, polygons


def full_overlay(image, polygons):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(output, polygons, True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    return output


def make_montage(names, filename):
    canvas = Image.new('RGB', (2304, 960), '#222222')
    drawer = ImageDraw.Draw(canvas)
    for index, stem in enumerate(names):
        path = ROOT / 'images' / (stem + '.jpg')
        image, mask, polygons = load_pair(path)
        x, y = (index % 3) * 768, (index // 3) * 480
        raw = Image.fromarray(image).convert('RGB').resize((384, 384))
        overlay = Image.fromarray(cv2.cvtColor(full_overlay(image, polygons), cv2.COLOR_BGR2RGB)).resize((384, 384))
        canvas.paste(raw, (x, y))
        canvas.paste(overlay, (x + 384, y))
        short = stem.split('_8bit_Y_', 1)[-1]
        drawer.text((x + 4, y + 388), stem[:2] + ' ' + short[:65], font=FONT, fill='white')
        drawer.text((x + 4, y + 411), f'raw | green label outline; {int(mask.sum()):,} px; {len(polygons)} polygons', font=FONT, fill='white')
        cv2.imwrite(str(OUT / (stem + '_outline.png')), full_overlay(image, polygons))
    canvas.save(OUT / filename)


rows = []
for path in sorted((ROOT / 'images').glob('*.jpg')):
    image, mask, polygons = load_pair(path)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
    rows.append(dict(name=path.stem, volume=path.stem[:2],
                     augmented=bool(re.search(r'_[A-Za-z0-9]{16}$', path.stem)),
                     height=image.shape[0], width=image.shape[1],
                     polygon_count=len(polygons), foreground_pixels=int(mask.sum()),
                     components8=count-1, component_areas_descending=areas,
                     components_le_25=sum(a <= 25 for a in areas),
                     pixels_in_components_le_25=sum(a for a in areas if a <= 25)))
(OUT / 'smoke_label_component_inventory.json').write_text(json.dumps(rows, indent=2))

f2 = 'F2_30_5_3_2026_8bit_Y_'
f3 = 'F3_10_4_30_2026_8bit_Y_'
make_montage([f2 + name for name in (
    'Transverse_1712', 'Sagittal_0326', 'Coronal_1498',
    'AzimuthalCoronal_0364', 'AzimuthalSagittal_0483', 'Transverse_Azimuthal_2594'
)], 'smoke_F2_unaugmented_montage.png')
make_montage([f3 + name for name in (
    'Transverse_1317', 'Transverse_1545', 'Sagittal_2161',
    'Transverse_Azimuthal_0265', 'TiltedCoronal_Horizontal_p30_0944',
    'TiltedSagittal_Horizontal_p30_1943'
)], 'smoke_F3_unaugmented_montage.png')
make_montage([f2 + name for name in (
    'Coronal_0230_ULaIv3wHcoVPypDo', 'Sagittal_0099_unG6DwJk6wsOeObR',
    'Transverse_2789_DjztdatWfDHxLt3o',
    'AzimuthalCoronal_2836_poDWCcwkkSmddMKM',
    'AzimuthalTiltedSagittal_Vertical_p30_2612_x5iKu4z9q6CXND0Y',
    'AzimuthalTiltedSagittal_Horizontal_p30_0878_d45Yb1fJQ7gxnl5S'
)], 'smoke_F2_augmented_montage.png')
for volume in ('F2', 'F3'):
    group = [r for r in rows if r['volume'] == volume]
    print(volume, len(group), 'augmented', sum(r['augmented'] for r in group),
          'component median', float(np.median([r['components8'] for r in group])),
          'small component pixels', sum(r['pixels_in_components_le_25'] for r in group))
print('montages and inventory saved to', OUT)

details = [f2 + 'Coronal_1498', f2 + 'AzimuthalCoronal_0364',
           f2 + 'AzimuthalSagittal_0483', f3 + 'Transverse_1317',
           f3 + 'Transverse_Azimuthal_0265',
           f2 + 'AzimuthalCoronal_2836_poDWCcwkkSmddMKM']
detail_canvas = Image.new('RGB', (3072, 1120), '#222222')
detail_drawer = ImageDraw.Draw(detail_canvas)
for index, stem in enumerate(details):
    image, mask, polygons = load_pair(ROOT / 'images' / (stem + '.jpg'))
    yy, xx = np.nonzero(mask)
    x0, x1 = max(0, int(xx.min())-70), min(image.shape[1], int(xx.max())+71)
    y0, y1 = max(0, int(yy.min())-70), min(image.shape[0], int(yy.max())+71)
    crop = image[y0:y1, x0:x1]
    lo, hi = np.percentile(crop[crop > 0], [1, 99.8])
    enhanced = np.asarray(np.clip((crop.astype(float)-lo) * 255 / max(1, hi-lo), 0, 255), np.uint8)
    local_polygons = [p - [x0, y0] for p in polygons]
    raw = Image.fromarray(enhanced).convert('RGB').resize((512, 512))
    overlay = Image.fromarray(cv2.cvtColor(full_overlay(enhanced, local_polygons), cv2.COLOR_BGR2RGB)).resize((512, 512))
    x, y = (index % 3)*1024, (index//3)*560
    detail_canvas.paste(raw, (x, y))
    detail_canvas.paste(overlay, (x+512, y))
    detail_drawer.text((x+4, y+515), stem[:2] + ' ' + stem.split('_8bit_Y_', 1)[-1], font=FONT, fill='white')
    cv2.imwrite(str(OUT / (stem + '_contrast_crop.png')), cv2.cvtColor(np.asarray(Image.fromarray(enhanced).convert('RGB')), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / (stem + '_contrast_crop_outline.png')), full_overlay(enhanced, local_polygons))
detail_canvas.save(OUT / 'smoke_contrast_detail_montage.png')
