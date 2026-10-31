"""Upload only published JPEG crops to R2, checkpointing successful uploads.

No originals, masks, credentials, or review artifacts are eligible. Inventory
comes from a frozen published manifest, never a recursive bucket sync.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'data/coruna/site'
STATE = Path.home() / '.local/state/typology-machine/r2'
PREFIX = 'crops/v1'


def crop_path(site, url):
    if not isinstance(url, str) or not re.fullmatch(r'/?(?:img|proposal-assets)/[A-Za-z0-9_./-]+\.jpg', url):
        raise ValueError('Not a public crop URL')
    if any(part in ('.', '..') for part in url.split('/')):
        raise ValueError('Invalid crop path')
    relative = url.lstrip('/')
    path = site / relative
    resolved = path.resolve()
    city = site.parent.resolve()
    # Proposal assets are symlinks into the city's detection folders.
    if not resolved.is_relative_to(city) or not resolved.is_file():
        raise ValueError('Crop missing or outside the city dataset')
    return relative, path


def inventory(site, manifest=None, legacy=None):
    manifest = manifest or json.loads((site / 'layers/manifest.json').read_text())
    for category in manifest['categories']:
        path = (site / category['url']).resolve()
        if not path.is_relative_to((site / 'layers').resolve()):
            raise ValueError('Layer path outside published layers')
        for feature in json.loads(path.read_text())['features']:
            yield feature['properties']['crop']
    for sign in (legacy or json.loads((site / 'data.json').read_text()))['signs']:
        yield 'img/' + sign['id'] + '.jpg'


def open_database(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS uploads (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, sha256 TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS inventory (path TEXT PRIMARY KEY)')
    return db


def upload(client, site, bucket, url, previous=None):
    relative, path = crop_path(site, url)
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    if previous and signature == previous[:2]:
        return relative, *signature, previous[2], False
    data = path.read_bytes()
    after = path.stat()
    if signature != (after.st_size, after.st_mtime_ns):
        raise ValueError('Crop changed while reading')
    sha = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data, usedforsecurity=False).digest()
    client.put_object(Bucket=bucket, Key=PREFIX + '/' + relative, Body=data,
                      ContentType='image/jpeg', CacheControl='public, max-age=86400',
                      ContentMD5=base64.b64encode(md5).decode(), Metadata={'sha256': sha})
    return relative, *signature, sha, True


def verify_listing(client, bucket, db):
    """Compare checkpoints with the bucket listing; forget rows R2 lacks so the next run re-uploads them."""
    remote = {}
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=PREFIX + '/'):
        for item in page.get('Contents', []):
            remote[item['Key'][len(PREFIX) + 1:]] = item['Size']
    rows = db.execute('SELECT i.path,u.size FROM inventory i LEFT JOIN uploads u ON i.path=u.path').fetchall()
    stale = [path for path, size in rows if size is None or remote.get(path) != size]
    db.executemany('DELETE FROM uploads WHERE path=?', [(path,) for path in stale])
    db.commit()
    return {'listed': len(remote), 'inventory': len(rows), 'stale': len(stale), 'stale_sample': stale[:20]}


def verify_public(site, sample, public):
    import requests
    if urlsplit(public).scheme != 'https':
        raise ValueError('Public crop origin must use HTTPS')
    for relative in sample:
        _, path = crop_path(site, relative)
        response = requests.get(public.rstrip('/') + '/' + PREFIX + '/' + relative, timeout=30)
        response.raise_for_status()
        if hashlib.sha256(response.content).digest() != hashlib.sha256(path.read_bytes()).digest():
            raise ValueError('Public R2 crop differs from local crop')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=SITE)
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--env-file', type=Path, default=Path.home() / '.config/typology-machine/r2.env')
    parser.add_argument('--account')
    parser.add_argument('--bucket')
    parser.add_argument('--public-url')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, help='Upload a bounded sample, without activating remote delivery')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--verify-listing', action='store_true', help='Check every crop against the bucket listing by size, then exit')
    parser.add_argument('--activate', action='store_true', help='Enable remote delivery only after a complete upload and public checks')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    args.account = args.account or os.environ.get('R2_ACCOUNT_ID')
    args.bucket = args.bucket or os.environ.get('R2_BUCKET', 'typology-crops')
    args.public_url = args.public_url or os.environ.get('R2_PUBLIC_URL')
    if not args.inventory_only and (not args.account or not args.public_url):
        parser.error('Set R2_ACCOUNT_ID and R2_PUBLIC_URL in --env-file, or pass --account and --public-url')
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Workers and limit must be positive')
    if args.activate and (args.limit or args.inventory_only):
        parser.error('Activation requires an unlimited upload')
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'account': args.account, 'bucket': args.bucket, 'prefix': PREFIX}
        identity_path = args.state / 'destination.json'
        if args.account and identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Use a separate checkpoint directory for a different destination')
        if args.account:
            identity_path.write_text(json.dumps(identity))
        with open_database(args.state / 'uploads.sqlite3') as db:
            manifest_bytes = (args.site / 'layers/manifest.json').read_bytes()
            legacy_bytes = (args.site / 'data.json').read_bytes()
            signature = hashlib.sha256(manifest_bytes + legacy_bytes).hexdigest()
            cached_path = args.state / 'inventory.json'
            cached = json.loads(cached_path.read_text()) if cached_path.exists() else {}
            cached_count = db.execute('SELECT count(*) FROM inventory').fetchone()[0]
            if cached.get('signature') == signature and cached_count == cached.get('crops'):
                total_bytes = cached['bytes']
            else:
                db.execute('DELETE FROM inventory')
                total_bytes = 0
                for url in inventory(args.site, json.loads(manifest_bytes), json.loads(legacy_bytes)):
                    relative, path = crop_path(args.site, url)
                    cursor = db.execute('INSERT OR IGNORE INTO inventory VALUES (?)', (relative,))
                    if cursor.rowcount:
                        total_bytes += path.stat().st_size
                db.commit()
            count = db.execute('SELECT count(*) FROM inventory').fetchone()[0]
            cached_path.write_text(json.dumps({'signature': signature, 'crops': count, 'bytes': total_bytes}))
            report = {'crops': count, 'bytes': total_bytes, 'bucket': args.bucket,
                      'public_url': args.public_url, 'started_at': time.time(), 'uploaded': 0, 'checked': 0}
            report_path = args.state / 'status.json'
            report_path.write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            if args.inventory_only:
                return
            import boto3
            from botocore.config import Config
            # Never fall back to credentials for an unrelated AWS account.
            if not all(os.environ.get(key) for key in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')):
                raise RuntimeError('Save bucket-scoped R2 credentials in the configured env file')
            client = boto3.client('s3', endpoint_url=f'https://{args.account}.r2.cloudflarestorage.com',
                                  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], region_name='auto',
                                  config=Config(max_pool_connections=args.workers, retries={'mode': 'standard', 'max_attempts': 5}))
            client.head_bucket(Bucket=args.bucket)
            if args.verify_listing:
                report['listing'] = verify_listing(client, args.bucket, db)
                report['finished_at'] = time.time()
                report_path.write_text(json.dumps(report, indent=2))
                print(json.dumps(report['listing']), flush=True)
                return
            rows = db.execute('SELECT i.path,u.size,u.mtime_ns,u.sha256 FROM inventory i LEFT JOIN uploads u ON i.path=u.path ORDER BY i.path')
            sample = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                while batch := rows.fetchmany(min(64, args.limit - report['checked']) if args.limit else 64):
                    futures = [pool.submit(upload, client, args.site, args.bucket, row[0], tuple(row[1:]) if row[1] is not None else None) for row in batch]
                    for future in futures:
                        relative, size, mtime, sha, changed = future.result()
                        db.execute('INSERT OR REPLACE INTO uploads VALUES (?,?,?,?)', (relative, size, mtime, sha))
                        report['uploaded'] += int(changed)
                        report['checked'] += 1
                        if len(sample) < 3:
                            sample.append(relative)
                    db.commit()
                    report_path.write_text(json.dumps(report, indent=2))
                    print(f"{report['checked']}/{count} checked; {report['uploaded']} uploaded", flush=True)
                    if args.limit and report['checked'] >= args.limit:
                        break
            if args.activate:
                if report['checked'] != count:
                    raise RuntimeError('Incomplete crop upload')
                migration = Path.home() / '.local/state/typology-machine/storage-migration.json'
                while migration.exists() and json.loads(migration.read_text()).get('stage') not in ('complete', 'failed'):
                    print('Uploads complete; waiting for storage migration before activating delivery', flush=True)
                    time.sleep(30)
                verify_public(args.site, sample, args.public_url)
                config = {'base_url': args.public_url.rstrip('/') + '/' + PREFIX, 'published_at': time.time(), 'crops': count}
                temporary = args.site / '.crop-storage.json.tmp'
                temporary.write_text(json.dumps(config))
                temporary.replace(args.site / 'crop-storage.json')
                report['activated'] = True
            report['finished_at'] = time.time()
            report_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
