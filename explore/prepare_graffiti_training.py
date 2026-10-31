"""Prepare a bounded, provenance-preserving graffiti LoRA dataset from published crops.

No detection runs. Selection is deliberately conservative but remains machine selected;
contact sheets support human review. Camera spacing and image hashes reduce repeated
works without pretending to prove that all artworks are distinct.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path.home() / '.local/state/typology-machine/query'
CAPTION = 'a photograph of c0runa graffiti, painted street art on a wall'


def distance(a, b):
    return math.hypot((a['la']-b['la'])*111320, (a['lo']-b['lo'])*111320*math.cos(math.radians(a['la'])))


def perceptual_hash(image):
    # Low-frequency DCT hash, independent of optional imagehash/scipy packages.
    pixels = np.asarray(image.convert('L').resize((32, 32)), dtype=float)
    x = np.arange(32)
    matrix = np.cos(np.pi / 32 * (x[None, :] + .5) * x[:, None])
    low = (matrix @ pixels @ matrix.T)[:8, :8].flatten()[1:]
    median = np.median(low)
    return sum(int(value > median) << bit for bit, value in enumerate(low))


def prepare(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new path to preserve existing data and splits')
    meta = json.loads((args.state / 'current.json').read_text())
    db = sqlite3.connect(f"file:{args.state / (meta['query_snapshot'] + '.sqlite3')}?mode=ro", uri=True)
    rows = db.execute('''SELECT r.payload,o.group_key FROM observations o JOIN records r ON r.id=o.id
        WHERE o.category='graffiti' AND o.score>=? ORDER BY o.score DESC,o.quality DESC,o.id''', (args.confidence,)).fetchall()
    db.close()
    selected, hashes, groups, panoramas = [], [], set(), set()
    original_train, original_validation, inherited_exclusions = [], [], []
    parent_summary = None
    if getattr(args, 'expand_from', None):
        from publish_huggingface import validate_reviewed_dataset
        parent_summary, parent_rows = validate_reviewed_dataset(args.expand_from)
        if args.train < len(parent_rows['train']) or args.validation < len(parent_rows['validation']):
            raise ValueError('Expanded split targets must retain every reviewed source image')
        for split, records in parent_rows.items():
            for row in records:
                item = dict(row)
                path = args.expand_from / split / row['file_name']
                with Image.open(path) as image:
                    item['_image'] = image.convert('RGB').copy()
                item['_original_path'] = path
                selected.append(item)
                (original_train if split == 'train' else original_validation).append(item)
                hashes.append(int(item['phash'], 16)); groups.add(item['group_id']); panoramas.add(item['panorama_id'])
        exclusion_path = args.expand_from / 'review-exclusions.json'
        if exclusion_path.exists():
            inherited_exclusions = json.loads(exclusion_path.read_text())
    inherited_ids = {item['id'] for item in selected}
    excluded_ids = {item['id'] for item in inherited_exclusions}
    rejections = {}
    def reject(reason):
        rejections[reason] = rejections.get(reason, 0) + 1
    target = args.train + args.validation
    for payload, group in rows:
        item = json.loads(payload)
        if item['id'] in inherited_ids or item['id'] in excluded_ids:
            reject('previously_reviewed'); continue
        panorama = item['source'] + ':' + item['view_id'].rsplit('_', 1)[0]
        if group in groups or panorama in panoramas:
            reject('same_group_or_panorama'); continue
        if item.get('width', 0) < 256 or item.get('height', 0) < 256:
            reject('small_crop'); continue
        if any(distance(item, other) < args.spacing for other in selected):
            reject('nearby_camera'); continue
        path = (args.site / item['crop'].lstrip('/')).resolve()
        try:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
                if min(image.size) < 256 or max(image.size)/min(image.size) > 3:
                    reject('size_or_aspect'); continue
                # Flat/near-blank and severely soft crops should not dominate training.
                gray = np.asarray(image.convert('L').resize((256, 256)), dtype=float)
                sharpness = float(np.var(gray[1:-1, :-2]+gray[1:-1, 2:]+gray[:-2, 1:-1]+gray[2:, 1:-1]-4*gray[1:-1, 1:-1]))
                if float(gray.std()) < 20 or sharpness < 35:
                    reject('low_detail'); continue
                phash = perceptual_hash(image)
                if any((phash ^ other).bit_count() <= 12 for other in hashes):
                    reject('perceptual_duplicate'); continue
                item.update(group_id=group, panorama_id=panorama, phash=f'{phash:016x}', sharpness=round(sharpness, 2), original_size=list(image.size))
                # Save max 1024px, retaining full crop and aspect ratio. Training owns resizing.
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                item['_image'] = image
        except (OSError, ValueError):
            reject('unreadable_crop'); continue
        selected.append(item); hashes.append(phash); groups.add(group); panoramas.add(panorama)
        if len(selected) >= target * 3:
            break
    # Hold out geographically separated camera locations, then exclude nearby train crops.
    validation, train = list(original_validation), list(original_train)
    candidates = [item for item in selected if item['id'] not in inherited_ids]
    for item in candidates:
        if (len(validation) < args.validation and all(distance(item, other) >= 150 for other in validation)
                and all(distance(item, other) >= 100 for other in original_train)):
            validation.append(item)
    validation_ids = {item['id'] for item in validation}
    for item in candidates:
        if item['id'] in validation_ids:
            continue
        if len(train) >= args.train:
            break
        if any(distance(item, other) < 100 for other in validation):
            reject('validation_distance_buffer'); continue
        train.append(item)
        if len(train) >= args.train:
            break
    counts = {}
    for split, items in [('train', train), ('validation', validation)]:
        directory = output / split; directory.mkdir(parents=True, exist_ok=True)
        lines = []
        for number, item in enumerate(items):
            filename = item.get('file_name') or hashlib.sha256(item['id'].encode()).hexdigest()[:20] + '.jpg'
            if '_original_path' in item:
                shutil.copyfile(item['_original_path'], directory / filename)
            else:
                item['_image'].save(directory / filename, quality=93)
            row = {k: v for k, v in item.items() if not k.startswith('_')}
            row.update(file_name=filename, text=item.get('text', CAPTION))
            lines.append(json.dumps(row, ensure_ascii=False))
        (directory / 'metadata.jsonl').write_text('\n'.join(lines)+'\n')
        # Review every image in numbered pages, not only the first few attractive samples.
        for start in range(0, len(items), 64):
            batch = items[start:start+64]
            sheet = Image.new('RGB', (8*160, math.ceil(len(batch)/8)*180), '#e8e5dc')
            draw = ImageDraw.Draw(sheet)
            for offset, item in enumerate(batch):
                thumb = ImageOps.contain(item['_image'], (152, 148))
                x, y = offset % 8 * 160, offset // 8 * 180
                sheet.paste(thumb, (x+(160-thumb.width)//2, y))
                draw.text((x+4, y+150), f'{start+offset:03d} {item["score"]:.2f} {item["source"]}', fill='black')
            sheet.save(output / f'review-{split}-{start//64+1}.jpg', quality=88)
        counts[split] = len(items)
    summary = dict(counts, snapshot=meta['query_snapshot'], minimum_confidence=args.confidence, minimum_crop_edge=256,
        minimum_camera_spacing_metres=args.spacing, validation_buffer_metres=100, caption=CAPTION,
        rejections=rejections, reviewed=False,
        limitations=['Machine-selected graffiti proposals can include wrong categories, people or signs.',
                    'Camera coordinates, existing groups and perceptual hashes reduce duplication but cannot guarantee distinct artworks.',
                    'All accepted crops need contact-sheet review before training.'])
    if parent_summary:
        summary['expanded_from'] = {'snapshot': parent_summary['snapshot'],
            'manifest_sha256': parent_summary['manifest_sha256'],
            'train': len(original_train), 'validation': len(original_validation)}
        summary['inherited_reviewed_ids'] = sorted(inherited_ids)
        (output / 'review-exclusions.json').write_text(json.dumps(inherited_exclusions, indent=2)+'\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'inherited_reviewed_ids'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--site', type=Path, default=ROOT/'data/coruna/site')
    parser.add_argument('--output', type=Path, default=Path('/mnt/data/projects/typology-machine/graffiti-ai/dataset'))
    parser.add_argument('--confidence', type=float, default=.90)
    parser.add_argument('--spacing', type=float, default=20)
    parser.add_argument('--train', type=int, default=256)
    parser.add_argument('--validation', type=int, default=32)
    parser.add_argument('--expand-from', type=Path, help='Preserve all reviewed source images and their split assignments')
    arguments = parser.parse_args()
    if not 1 <= arguments.train <= 4096 or not 1 <= arguments.validation <= 256:
        parser.error('Train must be 1–4096 and validation 1–256')
    prepare(arguments)
