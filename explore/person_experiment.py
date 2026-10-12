"""Person-only SAM3 pilot on the frozen street-text sample; no classifier or OCR.

Reuses production inference/crop/mask writing with one GPU worker. Completed
views are immutable checkpoints; repeat the same command to resume.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import multicategory_detect as sam

ROOT = Path(__file__).resolve().parents[1]


def prepare(sample_path, source_run):
    frozen = json.loads(sample_path.read_text())
    if Path(frozen['run']).resolve() != source_run.resolve():
        raise ValueError('Frozen sample belongs to a different source run')
    samples = frozen['samples']
    keys = [item['result']['view_id'] for item in samples]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate views in frozen sample')
    if not samples or any(item['result']['source'] != 'apple' for item in samples):
        raise ValueError('Pilot requires existing Apple source views')
    config = json.loads((source_run/'config.json').read_text())
    config.update(prompts={'person': ['person']}, legacy_store_sign='disabled',
                  experiment='person-pilot-v1', sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest())
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    return samples, config, source_run.parent/run_id


def pending_images(samples, out, should_stop):
    for item in samples:
        if should_stop():
            break
        result = item['result']
        key = result['view_id']
        if (out/'views'/key/'result.json').exists():
            continue
        path = Path(result['source_path'])
        with Image.open(path) as source:
            image = source.convert('RGB')
        if list(image.size) != result['image_size']:
            raise ValueError(f'Source dimensions changed for {key}')
        yield key, result['metadata'], path, image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=ROOT/'data/coruna/comparison_share/street-text/sample.json')
    parser.add_argument('--source-run', type=Path, default=ROOT/'data/coruna/apple/multicategory/765844a02f75')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    samples, config, out = prepare(args.sample, args.source_run)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out/'.pilot.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out/'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Existing run config mismatch')
        sam.atomic_json(path, config)
        sam.atomic_json(out/'sample.json', {'selection': 'Unchanged 50-view street-text sample: half random, half stratified by prior object categories; one view per panorama. Not selected for people.',
                                          'source_run': str(args.source_run.resolve()), 'views': [r['result']['view_id'] for r in samples]})
        print(out, flush=True)
        if args.prepare_only:
            return
        # Overrides are isolated to this standalone process. Never add legacy
        # shop detections or run the production category prompt list here.
        sam.PROMPTS = config['prompts']
        sam.legacy_results = lambda *_: (set(), {})
        sam.loaded_views = lambda base, options, output, index, stop: sam.prefetch(
            pending_images(samples, output, stop), lambda entry: entry)
        options = SimpleNamespace(source='apple', workers=1, torch_threads=2,
                                  threshold=config['threshold'], cache_samples=0, max_new_views=None)
        sam.run_worker(args.source_run.parents[2], options, out, config, 0)
        rows = [json.loads(p.read_text()) for p in (out/'views').glob('*/result.json')]
        summary = {'views': len(rows), 'views_with_people': sum(bool(r['predictions']) for r in rows),
                   'person_crops': sum(len(r['predictions']) for r in rows), 'planned_views': len(samples)}
        sam.atomic_json(out/'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
