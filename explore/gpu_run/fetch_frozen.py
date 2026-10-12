"""Fetch frozen Google inputs directly, accepting only byte-identical panoramas."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import requests
from streetlevel import streetview

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    inventory = ROOT / 'data/coruna/gsv'
    entries = json.loads((inventory / 'gpu_manifest.json').read_text())['panoramas']
    missing = []
    for entry in entries:
        path = inventory / 'panos' / (entry['metadata']['id'] + '.jpg')
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
            continue
        missing.append(entry)
    if args.limit:
        missing = missing[:args.limit]
    print(f'{len(missing)} frozen images need fetching', flush=True)
    original_get = requests.Session.get
    def timed_get(self, *a, **kw):
        kw.setdefault('timeout', 30)
        return original_get(self, *a, **kw)
    requests.Session.get = timed_get

    def fetch(entry):
        pano_id = entry['metadata']['id']
        path = inventory / 'panos' / (pano_id + '.jpg')
        temporary = path.with_suffix('.fetch.jpg')
        error = None
        for attempt in range(3):
            try:
                pano = streetview.find_panorama_by_id(pano_id, download_depth=False)
                if pano is None:
                    raise ValueError('Panorama unavailable')
                image = streetview.get_panorama(pano, zoom=4)
                image.save(temporary, quality=88)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('Checksum differs from frozen source')
                temporary.replace(path)
                return pano_id, None
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:200]
                if 'Checksum differs' in error:
                    break
                time.sleep(2)
            finally:
                temporary.unlink(missing_ok=True)
        return pano_id, error

    started = time.monotonic()
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, entry) for entry in missing]
        for count, future in enumerate(as_completed(futures), 1):
            pano_id, error = future.result()
            if error:
                failed.append({'id': pano_id, 'error': error})
                temporary = ROOT / 'logs/fetch-failures.tmp'
                temporary.write_text(json.dumps(failed, indent=2))
                temporary.replace(ROOT / 'logs/fetch-failures.json')
            if count % 50 == 0 or count == len(missing):
                print(f'{count}/{len(missing)} fetched; {len(failed)} need upload; {time.monotonic()-started:.1f}s', flush=True)
    (ROOT / 'logs/fetch-failures.json').write_text(json.dumps(failed, indent=2))
    print('Direct fetch finished', flush=True)


if __name__ == '__main__':
    main()
