"""CPU-only text-region experiment, independent of SAM; no transcription required.

Run: python explore/street_text_experiment.py --run data/coruna/apple/multicategory/765844a02f75
Outputs a frozen sample and per-view resumable records under comparison_share/street-text.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import pillow_heif
from rapidocr_onnxruntime import RapidOCR

pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def bounds(poly):
    a = np.asarray(poly)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def overlap(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def merge(rows):
    # Merge repeat detections from overlapping tiles, never across street views.
    kept = []
    for row in sorted(rows, key=lambda r: -area(r['box'])):
        match = next((p for p in kept if overlap(p['box'], row['box']) / max(1, area(row['box'])) >= .8
                      and overlap(p['box'], row['box']) / max(1, area(p['box'])) >= .5), None)
        if match:
            match['passes'] = sorted(set(match['passes'] + row['passes']))
        else:
            kept.append(row)
    return sorted(kept, key=lambda r: (r['box'][1], r['box'][0]))


def positions(length, size=1536, overlap_px=256):
    return sorted(set(list(range(0, max(1, length-size+1), size-overlap_px)) + [max(0, length-size)]))


def sample(run, count):
    paths = sorted((run/'views').glob('*/result.json'))
    random.Random(20261012).shuffle(paths)
    # Random half plus targeted variety; one view per panorama throughout.
    chosen, used = [], set()
    quotas = {'random': count//2, 'graffiti': count//10, 'street_name_plate': count//10,
              'store_sign': count//10, 'other': count//10, 'empty': count - count//2 - 4*(count//10)}
    for group, quota in quotas.items():
        if quota == 0:
            continue
        found = 0
        for path in paths:
            r = json.loads(path.read_text())
            pano = r['metadata']['id']
            if pano in used:
                continue
            cats = {p['category_proposal'] for p in r['predictions']}
            fits = group == 'random' or group in cats or (group == 'empty' and not cats) or (group == 'other' and bool(cats) and not cats.intersection({'graffiti', 'store_sign', 'street_name_plate'}))
            if not fits:
                continue
            chosen.append({'group': group, 'result': r})
            used.add(pano)
            found += 1
            if found >= quota:
                break
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT/'data/coruna/comparison_share/street-text')
    p.add_argument('--count', type=int, default=50)
    a = p.parse_args()
    if a.count < 10:
        p.error('--count must be at least 10 for the stratified sample')
    a.out.mkdir(parents=True, exist_ok=True)
    frozen = a.out/'sample.json'
    if not frozen.exists():
        save(frozen, {'run': str(a.run.resolve()), 'samples': sample(a.run, a.count)})
    manifest = json.loads(frozen.read_text())
    if len(manifest['samples']) != a.count:
        raise ValueError('Frozen sample size differs from --count; use a new output directory')
    if manifest['run'] != str(a.run.resolve()):
        raise ValueError('Output belongs to a different run')
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1, det_use_cuda=False,
                      cls_use_cuda=False, rec_use_cuda=False)
    package = Path(__import__('rapidocr_onnxruntime').__file__).parent
    models = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in package.rglob('*det*.onnx')}
    config = {'package': importlib.metadata.version('rapidocr_onnxruntime'), 'detector_sha256': models,
              'box_threshold': .5, 'tile_size': 1536, 'tile_overlap': 256, 'recognition': False,
              'comparison': 'text axis-aligned box >=50% inside SAM proposal box; geometric overlap only'}
    config_path = a.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Detector configuration changed; use a new output directory')
    save(config_path, config)
    shutil.copyfile(package/'config.yaml', a.out/'detector-defaults.yaml')
    records = []
    start = time.monotonic()
    for index, item in enumerate(manifest['samples']):
        result = item['result']
        view = result['view_id']
        folder = a.out/view
        folder.mkdir(exist_ok=True)
        record_path = folder/'regions.json'
        if record_path.exists():
            records.append(json.loads(record_path.read_text()))
            continue
        if result['source'] != 'apple':
            raise ValueError('This pilot expects original Apple perspective images')
        with Image.open(result['source_path']) as src:
            im = src.convert('RGB')
        if list(im.size) != result['image_size']:
            raise ValueError('Source dimensions differ from SAM geometry')
        raw = []
        windows = [('whole', 0, 0, im.width, im.height)]
        windows += [('tiles', x, y, min(x+1536, im.width), min(y+1536, im.height))
                    for y in positions(im.height) for x in positions(im.width)]
        for mode, x, y, right, bottom in windows:
            # ndarray input is BGR in RapidOCR. Recognition and orientation are disabled.
            arr = np.asarray(im.crop((x, y, right, bottom)))[:, :, ::-1].copy()
            boxes, _ = engine(arr, use_det=True, use_cls=False, use_rec=False)
            for poly in boxes or []:
                poly = (np.asarray(poly) + [x, y]).tolist()
                raw.append({'polygon': poly, 'box': bounds(poly), 'passes': [mode]})
        regions = merge(raw)
        for i, row in enumerate(regions):
            row['id'] = f'{view}:{i}'
            matches = [q for q in result['predictions'] if overlap(row['box'], q['box']) / max(1, area(row['box'])) >= .5]
            row['sam_matches'] = [{'id': q['id'], 'category': q['category_proposal']} for q in matches]
            row['coverage'] = ('ocr_category' if any(q['category_proposal'] in {'store_sign', 'street_name_plate'} for q in matches)
                               else 'other_sam' if matches else 'outside_sam')
            box = row['box']
            crop = [max(0, int(box[0])-8), max(0, int(box[1])-8), min(im.width, int(box[2])+9), min(im.height, int(box[3])+9)]
            row['crop_box'] = crop
            row['crop'] = f'{view}/{i}.jpg'
            im.crop(crop).save(folder/f'{i}.jpg', quality=94)
        im.save(folder/'original.jpg', quality=90)
        im.thumbnail((1600, 1600))
        im.save(folder/'context.jpg', quality=88)
        record = {'view_id': view, 'group': item['group'], 'image_size': result['image_size'],
                  'metadata': result['metadata'], 'regions': regions, 'raw_detections': len(raw),
                  'sam_boxes': [{'box': q['box'], 'category': q['category_proposal']} for q in result['predictions']]}
        save(record_path, record)
        records.append(record)
        print(f'{index+1}/{len(manifest["samples"])}: {len(regions)} regions, {time.monotonic()-start:.1f}s', flush=True)
    counts = {k: sum(r['coverage'] == k for scene in records for r in scene['regions']) for k in ['outside_sam', 'other_sam', 'ocr_category']}
    report = {'sample_id': hashlib.sha256(frozen.read_bytes()).hexdigest()[:12], 'config': config,
              'counts': counts, 'scenes': records, 'total_regions': sum(counts.values()),
              'tile_only': sum(r['passes'] == ['tiles'] for s in records for r in s['regions'])}
    save(a.out/'review.json', report)
    shutil.copyfile(Path(__file__).with_suffix('.html'), a.out/'index.html')
    print(json.dumps({k: v for k, v in report.items() if k not in {'scenes', 'config'}}))


if __name__ == '__main__':
    main()
