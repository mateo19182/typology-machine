"""Tighten alphabet candidates to the letter's ink and reject clipped or low-resolution tiles.

Reads a candidate JSON from build_alphabet.py, re-cuts each letter from its sign crop with even
padding around the detected glyph, and re-reads the tight tile with OCR. Writes a refined JSON.
"""
import argparse
import base64
import io
import json
from pathlib import Path


def glyph_box(gray, core, np, cv2):
    """Ink bounds of the glyph inside `core` (left, right columns), plus the nearest neighbour ink.

    Threshold and polarity come from the OCR box alone, where ink is the minority. Components overlapping
    the middle half of the OCR box form the glyph; other substantial components are neighbours.
    Returns None when the glyph reaches the window edge or is too small to be a letter.
    """
    h, w = gray.shape
    inner = gray[core[2]:core[3], core[0]:core[1]]
    level, inner_mask = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = np.where(gray > level, 255, 0).astype(np.uint8)
    if (inner_mask > 0).mean() > .5:
        mask = 255 - mask  # Letters cover less of their own box than the background does.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lo, hi = core[0] + (core[1] - core[0]) * .25, core[0] + (core[1] - core[0]) * .75
    glyph = []
    for i, (x, y, cw, ch, area) in enumerate(stats[1:count], 1):
        if area < (core[3] - core[2]) * (core[1] - core[0]) * .004 or ch < (core[3] - core[2]) * .08:
            continue  # Specks and grain.
        if cw > (core[1] - core[0]) * 1.6 or ch > (core[3] - core[2]) * 1.6:
            continue  # Bands, frames and backgrounds: larger than any one letter here.
        if core[2] <= y + ch / 2 <= core[3] and x < hi and x + cw > lo:
            glyph.append(i)
    if not glyph:
        return None
    rows = stats[glyph]
    rows = stats[glyph]
    left, top = rows[:, 0].min(), rows[:, 1].min()
    right, bottom = (rows[:, 0] + rows[:, 2]).max(), (rows[:, 1] + rows[:, 3]).max()
    if left <= 0 or right >= w or top <= 0 or bottom >= h or bottom - top < (core[3] - core[2]) * .35 or right - left < 3:
        return None
    width = core[1] - core[0]
    if left < core[0] - width * .3 or right > core[1] + width * .3:
        return None  # Ink spills well past the OCR box: likely joined to a neighbouring letter.
    # Any other ink level with the glyph, including large shapes, limits the side padding.
    other = (mask[top:bottom] > 0) & ~np.isin(labels[top:bottom], glyph)
    columns = np.flatnonzero(other.sum(axis=0) > max(1, (bottom - top) * .05))
    before = columns[columns < left].max() + 1 if (columns < left).any() else 0
    after = columns[columns >= right].min() if (columns >= right).any() else w
    return left, top, right, bottom, before, after


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('candidates', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--site', type=Path, default=Path('data/coruna/site'))
    ap.add_argument('--min-height', type=int, default=56, help='minimum glyph height in source pixels')
    args = ap.parse_args()
    import cv2
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR
    cv2.setNumThreads(2)
    model = Path(__file__).resolve().parent.parent / 'data/models/ocr/latin_PP-OCRv5_rec_mobile.onnx'
    engine = RapidOCR(rec_model_path=str(model), intra_op_num_threads=2, inter_op_num_threads=1)
    data = json.loads(args.candidates.read_text())
    kept, rejected, images = [], {}, {}
    for item in data['letters']:
        path = args.site / item['crop'].lstrip('/')
        if item['crop'] not in images:
            images[item['crop']] = np.asarray(Image.open(path).convert('RGB')) if path.is_file() else None
        source = images[item['crop']]
        reason = None
        if source is None:
            reason = 'missing_crop'
        else:
            H, W = source.shape[:2]
            x0, y0 = round(item['rect'][0] * W), round(item['rect'][1] * H)
            x1, y1 = round((item['rect'][0] + item['rect'][2]) * W), round((item['rect'][1] + item['rect'][3]) * H)
            margin = round((y1 - y0) * .4)
            wx0, wx1 = max(0, x0 - margin), min(W, x1 + margin)
            wy0, wy1 = max(0, y0 - margin), min(H, y1 + margin)
            window = source[wy0:wy1, wx0:wx1]
            core = (x0 - wx0, x1 - wx0, y0 - wy0, y1 - wy0)
            box = glyph_box(cv2.cvtColor(window, cv2.COLOR_RGB2GRAY), core, np, cv2) if window.size else None
            if box is None:
                reason = 'clipped_or_empty'
            elif box[3] - box[1] < args.min_height:
                reason = 'low_resolution'
            elif min(box[0] - box[4], box[5] - box[2]) < max(2, (box[3] - box[1]) * .04):
                reason = 'touching_neighbour'  # No rectangle separates it from the next letter.
            else:
                l, t, r, b, before, after = box
                pad = round((b - t) * .12)
                # Even padding, but never past halfway to a neighbouring letter.
                pad_x = max(1, min(pad, (l - before) // 2, (after - r) // 2))
                l, r = l - pad_x + wx0, r + pad_x + wx0
                t, b = max(0, wy0 + t - pad), min(H, wy0 + b + pad)
                cut = source[t:b, l:r]
                recognized, _ = engine(cut, use_det=False, use_cls=False)
                if not recognized or recognized[0][0].strip().upper() != item['letter'] or recognized[0][1] < .9:
                    reason = 'tight_recognition'
                else:
                    image = Image.fromarray(cut)
                    image.thumbnail((150, 120), Image.LANCZOS)
                    out = io.BytesIO(); image.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out, format='PNG', optimize=True)
                    item = {**item, 'image': 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode(),
                            'rect': [l / W, t / H, (r - l) / W, (b - t) / H], 'pixel_height': int(box[3] - box[1]),
                            'tight_confidence': float(recognized[0][1])}
                    kept.append(item)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
    data = {**data, 'letters': kept}
    data['selection'] = {**data.get('selection', {}), 'min_glyph_height': args.min_height, 'refine_rejected': rejected}
    args.output.write_text(json.dumps(data, ensure_ascii=False))
    print(f'Kept {len(kept)} of {len(kept) + sum(rejected.values())}; rejected {rejected}')


if __name__ == '__main__':
    main()
