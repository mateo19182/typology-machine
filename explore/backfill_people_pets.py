"""Backfill only person/pet prompts into a separate, resumable Apple run."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import multicategory_detect as sam

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/coruna/apple/multicategory/765844a02f75'


def prepare(source):
    inventory = json.loads((source / 'person_pet_backfill.json').read_text())
    views = sorted(inventory['views'])
    if not views or len(views) != len(set(views)):
        raise ValueError('Backfill inventory must contain unique views')
    for view in views:
        if Path(view).name != view or view in ('.', '..'):
            raise ValueError('Invalid view ID')
    original = json.loads((source / 'config.json').read_text())
    effective = sam.runtime_config(source, original)
    config = {**original, 'prompts': {k: effective['prompts'][k] for k in ('person', 'pet')},
              'legacy_store_sign': 'disabled', 'experiment': 'people-pets-backfill-v1',
              'source_run': source.name,
              'views_sha256': hashlib.sha256(json.dumps(views).encode()).hexdigest()}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out = source.parent / run_id
    return views, config, out


def pending_images(source, views, out, stop):
    for view in views:
        if stop():
            break
        if (out / 'views' / view / 'result.json').exists():
            continue
        result = json.loads((source / 'views' / view / 'result.json').read_text())
        if result['source'] != 'apple' or result['view_id'] != view:
            raise ValueError('Source identity mismatch')
        if {'person', 'pet'} & set(result.get('attempted_categories', [])):
            raise ValueError('Inventory includes an already attempted category')
        path = ROOT / result['source_path']
        with Image.open(path) as image:
            image = image.convert('RGB')
        if list(image.size) != result['image_size']:
            raise ValueError('Source dimensions changed')
        yield view, result['metadata'], path, image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=SOURCE)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--max-new-views', type=int)
    args = parser.parse_args()
    views, config, out = prepare(args.source_run)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.backfill.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Existing backfill config differs')
        sam.atomic_json(path, config)
        sam.atomic_json(out / 'selection.json', {'source_run': str(args.source_run), 'views': views})
        print(out, flush=True)
        if args.prepare_only:
            return
        sam.loaded_views = lambda base, options, output, index, stop: sam.prefetch(
            pending_images(args.source_run, views, output, stop), lambda entry: entry)
        options = SimpleNamespace(source='apple', workers=1, torch_threads=2,
                                  threshold=config['threshold'], cache_samples=0,
                                  max_new_views=args.max_new_views)
        sam.run_worker(args.source_run.parents[2], options, out, config, 0)
        completed = sum((out / 'views' / v / 'result.json').exists() for v in views)
        sam.atomic_json(out / 'summary.json', {'planned_views': len(views), 'completed_views': completed})
        if completed != len(views) and not args.max_new_views:
            raise RuntimeError('Backfill interrupted; resume to finish missing views')


if __name__ == '__main__':
    main()
