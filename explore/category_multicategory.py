"""Freeze a small category-only review set from the published SAM snapshot."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
from PIL import Image
from category_eval import BASE, CATEGORIES, INSTRUCTION, ROOT, write_json


def prepare(site, output):
    destination = output / 'sample.json'
    if destination.exists():
        return json.loads(destination.read_text())
    manifest = json.loads((site / 'layers/manifest.json').read_text())
    rng = random.Random(20261012)
    selected, panoramas = [], set()
    focus = {'graffiti', 'street_name_plate', 'store_sign'}
    for layer in sorted(manifest['categories'], key=lambda c: (c['id'] not in focus, c['id'])):
        rows = [f['properties'] for f in json.loads((site / layer['url']).read_text())['features'] if not f['properties'].get('duplicate_of')]
        rng.shuffle(rows)
        # Half the focus cases deliberately challenge verification near SAM's threshold.
        count = 24 if layer['id'] in focus else 3
        challenging = sorted(rows, key=lambda r: r['score'])[:max(count * 10, 100)]
        rng.shuffle(challenging)
        pool = challenging[:count // 2] + rows if layer['id'] in focus else rows
        picked = 0
        for row in pool:
            pano = row['view_id'].rsplit('_', 1)[0]
            if pano in panoramas:
                continue
            original = (site / row['crop'].lstrip('/')).resolve()
            if not original.is_file():
                continue
            panoramas.add(pano)
            selected.append({'id': row['id'], 'source': row['source'], 'pano': pano, 'date': row['date'],
                             'path': str(original), 'sampling_hint': layer['id'], 'detector_prompt': row['prompt'],
                             'detector_score': row['score'], 'view_id': row['view_id'], 'run_id': manifest['run_id'],
                             'context_url': os.environ.get('TYPOLOGY_PUBLIC_URL', 'http://127.0.0.1:8790').rstrip('/') + row['raw']})
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise ValueError(f'Insufficient distinct panoramas for {layer["id"]}')
    rng.shuffle(selected)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(selected):
        original = Path(item['path'])
        item['sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
        item['image'] = f'images/{index}.jpg'
        with Image.open(original) as image:
            image = image.convert('RGB')
            image.thumbnail((768, 768))
            image.save(output / item['image'], quality=95)
        item['evaluation_sha256'] = hashlib.sha256((output / item['image']).read_bytes()).hexdigest()
    categories = {**CATEGORIES, 'door': 'An entrance door or doorway, rather than lettering painted on it.',
                  'shutter': 'A roller shutter covering an entrance or window; choose graffiti only when the paint itself is the primary subject.'}
    sample = {'id': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:16],
              'categories': categories, 'instruction': INSTRUCTION, 'items': selected,
              'source_snapshot': manifest['snapshot_id'], 'run_id': manifest['run_id'],
              'sampling': dict(Counter(r['sampling_hint'] for r in selected)),
              'note': '96 frozen crops from new SAM multicategory results: 24 each graffiti, street names and shop signs, plus 3 per other category. Half of each focus bucket comes from low SAM scores. One crop per panorama. Detector proposals are sampling strata, never reference labels. Human labels start empty; no attributes.'}
    write_json(destination, sample)
    return sample


def summarize(output):
    sample = json.loads((output / 'sample.json').read_text())
    items = {r['id']: r for r in sample['items']}
    with sqlite3.connect(BASE / 'reviews.sqlite3') as db:
        labels = dict(db.execute('SELECT item_id,category FROM labels WHERE sample_id=?', (sample['id'],)))
    summaries = []
    for run in sorted((output / 'runs').glob('*/run.json')):
        config = json.loads(run.read_text())
        if config['sample_id'] != sample['id']:
            continue
        predictions = {r['id']: r for r in map(json.loads, (run.parent / 'predictions.jsonl').read_text().splitlines())}
        reviewed = [key for key in predictions if key in labels]
        agreement = Counter()
        disagreements = []
        for key, prediction in predictions.items():
            proposed = items[key]['sampling_hint']
            if prediction['category'] == proposed:
                agreement[proposed] += 1
            else:
                disagreements.append({'id': key, 'proposal': proposed, 'prediction': prediction['category'], 'confidence': prediction['confidence']})
        summary = {'sample_id': sample['id'], 'run_id': run.parent.name, 'predictions': len(predictions),
                   'human_reviewed': len(reviewed), 'human_accuracy': sum(predictions[k]['category'] == labels[k] for k in reviewed) / len(reviewed) if reviewed else None,
                   'proposal_agreement': dict(agreement), 'sampling': sample['sampling'], 'disagreements': disagreements,
                   'note': 'Agreement with SAM is not accuracy. Only explicit human labels are reference labels. No automatic category overrides applied.'}
        write_json(run.parent / 'summary.json', summary)
        summaries.append(summary)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT / 'data/coruna/site')
    parser.add_argument('--out', type=Path, default=BASE / 'multicategory')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.out), indent=2))
    else:
        sample = prepare(args.site, args.out)
        print(sample['id'], len(sample['items']), sample['sampling'])
