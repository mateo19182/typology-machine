"""Snapshot completed SAM category proposals for the static map.

Usage: .venv/bin/python explore/export_layers.py coruna --run data/coruna/apple/multicategory/765844a02f75
Add --extra-run PATH for a pilot containing additional categories.
The legacy data.json and images are untouched. Crop assets are individual file symlinks;
all detections, including same-view duplicate flags, remain in the snapshot.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import shutil
import time
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlencode
from streetlevel import lookaround
from map_groups import load_groups

ROOT = Path(__file__).resolve().parent.parent
STYLES = {
    'store_sign': ('Shop signs', '#e8a435'),
    'street_name_plate': ('Street names', '#42b4dd'),
    'graffiti': ('Graffiti', '#ed6a9e'),
    'traffic_sign': ('Traffic signs', '#eb6557'),
    'street_light': ('Street lights', '#e7c954'),
    'bin': ('Bins', '#64b77a'),
    'bench': ('Benches', '#aa89d8'),
    'manhole_cover': ('Manhole covers', '#9ca5b4'),
    'junction_box': ('Electrical cabinets', '#b18b65'),
    'door': ('Doors', '#6d9eeb'),
    'shutter': ('Shutters', '#51b9ae'),
    'person': ('People', '#f08545'),
}


def asset_url(crop, city_base, out):
    """Expose only known crop directories, never arbitrary filesystem paths."""
    path = Path(crop)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    relative = path.relative_to(city_base.resolve())
    parts = relative.parts
    if path.suffix.lower() != '.jpg':
        raise ValueError(f'Unsupported crop asset: {relative}')
    if len(parts) >= 4 and parts[1:3] == ('store_sign', 'crops'):
        mount = Path(parts[0]) / 'legacy-crops'
        suffix = Path(*parts[3:])
    elif len(parts) >= 6 and parts[1] == 'multicategory' and parts[3] == 'views':
        mount = Path(parts[0]) / parts[2]
        suffix = Path(*parts[4:])
    else:
        raise ValueError(f'Crop outside approved directories: {relative}')
    link = out / 'proposal-assets' / mount / suffix
    if not link.is_symlink():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(path)
    elif link.resolve() != path:
        raise ValueError(f'Existing asset link points elsewhere: {link}')
    return '/proposal-assets/' + (mount / suffix).as_posix()


def feature(result, prediction, crop_url, run_id):
    metadata = result['metadata']
    lat, lon = float(metadata['lat']), float(metadata['lon'])
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('Invalid camera coordinates')
    width, height = result['image_size']
    box = prediction['box']
    crop_box = prediction.get('crop_box', box)
    heading = math.degrees(float(metadata.get('heading', 0)))
    heading += {0: 0, 1: 90, 2: 180, 3: -90}.get(metadata.get('face'), 0)
    provider_url = lookaround.build_permalink(lat, lon, heading=heading % 360, radians=False)
    if result['source'] == 'gsv':
        provider_url = 'https://www.google.com/maps/@?' + urlencode({'api': 1, 'map_action': 'pano', 'pano': metadata['id']})
    props = {
        'id': prediction['id'], 'category': prediction['category_proposal'],
        'score': round(prediction['score'], 5), 'prompt': prediction['prompt'],
        'crop': crop_url, 'width': round(crop_box[2] - crop_box[0]), 'height': round(crop_box[3] - crop_box[1]),
        'date': metadata.get('date', ''), 'source': result['source'], 'view_id': result['view_id'],
        'duplicate_of': prediction.get('duplicate_of'), 'location_basis': 'camera',
        'status': 'unverified_proposal',
        'raw': f"/proposal-context/{result['source']}/{run_id}/{result['view_id']}.jpg",
        'bx': [round(max(0, min(1, v / (width if i % 2 == 0 else height))), 6) for i, v in enumerate(box)],
        'provider_url': provider_url,
    }
    return {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(lon, 7), round(lat, 7)]}, 'properties': props}


def export(city_base, run, out, if_changed=False, extra_runs=()):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.layer-export.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return export_locked(city_base, run, out, if_changed, extra_runs)


def read_ocr(run):
    path = run / 'ocr.jsonl'
    data = path.read_bytes() if path.exists() else b''
    rows = {}
    for line in data.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            continue  # The worker may be appending its next checkpoint.
        row = json.loads(line)
        if not row.get('error'):
            rows[row['id']] = row
    return rows, hashlib.sha256(data).hexdigest()


def export_locked(city_base, run, out, if_changed, extra_runs=()):
    inputs, prompts, group_sources = [], {}, []
    # Supplemental runs add distinct layers; reject overlaps rather than silently
    # double-counting observations or choosing one run's model configuration.
    for folder in (run, *extra_runs):
        config = json.loads((folder / 'config.json').read_text())
        overlap = prompts.keys() & config['prompts'].keys()
        if overlap:
            raise ValueError(f'Categories appear in multiple runs: {sorted(overlap)}')
        prompts.update(config['prompts'])
        # Freeze lists before reading; newly completed views enter the next snapshot.
        results = sorted((folder / 'views').glob('*/result.json'))
        ocr, ocr_sha = read_ocr(folder)
        groups, sources = load_groups(city_base, folder.name, config['source'])
        for source in sources:
            if source not in group_sources:
                group_sources.append(source)
        stats = [(p.parent.name, p.stat().st_size, p.stat().st_mtime_ns) for p in results]
        inputs.append({'run': folder, 'config': config, 'results': results,
                       'ocr': ocr, 'ocr_sha': ocr_sha, 'groups': groups, 'stats': stats})
    signature = hashlib.sha256(json.dumps({'schema': 4, 'groups': group_sources, 'runs': [
        {'run': str(item['run'].resolve()), 'config': item['config'],
         'results': item['stats'], 'ocr': item['ocr_sha']} for item in inputs
    ]}, sort_keys=True).encode()).hexdigest()
    config = inputs[0]['config']
    current = out / 'layers' / 'manifest.json'
    if if_changed and current.exists():
        existing = json.loads(current.read_text())
        if existing.get('input_signature') == signature:
            return existing
    snapshot_at = datetime.now(timezone.utc).isoformat()
    snapshot = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = out / 'layers' / snapshot
    destination.mkdir(parents=True)
    streams = {category: (destination / f'{category}.geojson').open('w') for category in prompts}
    counts, flagged = Counter(), Counter()
    object_ids = {category: set() for category in prompts}
    fingerprint = hashlib.sha256()
    try:
        for stream in streams.values():
            stream.write('{"type":"FeatureCollection","features":[')
        for data in inputs:
            folder, groups, ocr = data['run'], data['groups'], data['ocr']
            fingerprint.update(str(folder.resolve()).encode() + b'\0')
            for index, path in enumerate(data['results']):
                raw = path.read_bytes()
                fingerprint.update(path.parent.name.encode() + b'\0' + raw)
                result = json.loads(raw)
                for prediction in result['predictions']:
                    category = prediction['category_proposal']
                    item = feature(result, prediction, asset_url(prediction['crop'], city_base, out), folder.name)
                    group = groups.get(prediction['id'])
                    if group:
                        if group['group_category'] != category:
                            raise ValueError('Duplicate group crosses category boundary')
                        item['properties'].update(duplicate_group=group['duplicate_group'], group_rep=group['group_rep'])
                    if not prediction.get('duplicate_of'):
                        object_ids[category].add(group['duplicate_group'] if group else prediction['id'])
                    text = ocr.get(prediction['id'])
                    if text:
                        item['properties'].update(text=text.get('text', ''), ocr_model=text.get('model'),
                                                  ocr_confidence=text.get('confidence'))
                    stream = streams[category]
                    if counts[category]:
                        stream.write(',')
                    json.dump(item, stream, ensure_ascii=False, separators=(',', ':'))
                    counts[category] += 1
                    flagged[category] += bool(prediction.get('duplicate_of'))
                if index and index % 10000 == 0:
                    print(f"{index}/{len(data['results'])} views; {sum(counts.values())} detections", flush=True)
        for stream in streams.values():
            stream.write(']}')
    except Exception:
        for stream in streams.values():
            stream.close()
        shutil.rmtree(destination)
        raise
    finally:
        for stream in streams.values():
            stream.close()
    manifest = {
        'schema_version': 4, 'group_sources': group_sources, 'grouping': 'partial_duplicate_proposals', 'input_signature': signature,
        'published_at': datetime.now(timezone.utc).isoformat(),
        'latest_detection_at': datetime.fromtimestamp(max((x[2] for data in inputs for x in data['stats']), default=0) / 1e9, timezone.utc).isoformat(),
        'ocr_records': sum(len(data['ocr']) for data in inputs), 'refresh_interval_seconds': 900,
        'city': city_base.name, 'snapshot_at': snapshot_at, 'snapshot_id': snapshot,
        'run_id': run.name, 'source': config['source'], 'completed_views': len(inputs[0]['results']),
        'runs': [{'run_id': data['run'].name, 'source': data['config']['source'],
                  'completed_views': len(data['results']), 'model': data['config']['model'],
                  'threshold': data['config']['threshold']} for data in inputs],
        'status': 'unverified_proposals', 'location_basis': 'camera', 'deduplicated': False,
        'total_detections': sum(counts.values()), 'results_sha256': fingerprint.hexdigest(),
        'model': config['model'], 'threshold': config['threshold'],
        'categories': [{'id': category, 'label': STYLES.get(category, (category, '#aaa'))[0],
                        'color': STYLES.get(category, (category, '#aaa'))[1], 'count': counts[category],
                        'same_view_duplicate_flags': flagged[category], 'object_count': len(object_ids[category]), 'prompts': category_prompts,
                        'run_id': data['run'].name, 'completed_views': len(data['results']),
                        'pilot': data['config'].get('experiment'),
                        'url': f'layers/{snapshot}/{category}.geojson'} for data in inputs for category, category_prompts in data['config']['prompts'].items()],
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    temp = out / 'layers' / f'.manifest-{snapshot}.tmp'
    temp.write_text(json.dumps(manifest, indent=2))
    os.replace(temp, out / 'layers' / 'manifest.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('city', nargs='?', default='coruna')
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--extra-run', action='append', type=Path, default=[],
                        help='Supplement with a run containing additional categories (repeatable)')
    parser.add_argument('--if-changed', action='store_true', help='Skip publishing when detections and OCR are unchanged')
    parser.add_argument('--prune-hours', type=int, help='Retain at least 8 snapshots and all snapshots newer than this many hours')
    args = parser.parse_args()
    base = ROOT / 'data' / args.city
    out = args.out or base / 'site'
    manifest = export(base, args.run.resolve(), out, args.if_changed, [p.resolve() for p in args.extra_run])
    if args.prune_hours is not None:
        if args.prune_hours < 24:
            parser.error('Snapshot retention must be at least 24 hours')
        snapshots = sorted((out / 'layers').glob('*/manifest.json'), reverse=True)
        for path in snapshots[8:]:
            if path.parent.name != manifest['snapshot_id'] and time.time() - path.stat().st_mtime > args.prune_hours * 3600:
                shutil.rmtree(path.parent)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
