"""Read-only checks for running transfers, OCR and the published viewer."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home() / '.local/state/typology-machine'
REPORT = STATE / 'background-status.json'
DATA_DISK = os.environ.get('TYPOLOGY_DATA_DISK', '/mnt/data')


def read_json(path):
    # Some existing workers rewrite their status files in place.
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.1)


def tail(path, size=4096):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-size))
        return stream.read().decode(errors='replace')


def check():
    now = time.time()
    previous = read_json(REPORT)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'alerts': [], 'progress': {}}
    alerts = report['alerts']
    units = ['typology-' + name for name in (
        'site.service', 'object-api.service', 'proposal-context.service', 'storage-migration.service',
        'r2-crops.service', 'ocr.service', 'layer-refresh.service', 'followup.service',
        'backfill.service', 'dedupe-followup.service', 'expanded-dedupe.service', 'expanded-dedupe.timer', 'layer-refresh.timer', 'followup.timer', 'ocr.timer', 'r2-crops.timer')]
    result = subprocess.run(['systemctl', '--user', 'show', *units, '-p', 'Id', '-p', 'ActiveState', '-p', 'Result'],
                            check=True, capture_output=True, text=True, timeout=20)
    services = {}
    for block in result.stdout.strip().split('\n\n'):
        row = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
        services[row.pop('Id')] = row
    report['services'] = services
    for name, row in services.items():
        if name == 'typology-expanded-dedupe.service' and row.get('Result') == 'exec-condition':
            continue  # Waiting for verified storage migration completion.
        if row.get('ActiveState') == 'failed' or row.get('Result', 'success') != 'success':
            alerts.append(name + ': ' + str(row))
    for name in ('site', 'object-api'):
        if services['typology-' + name + '.service']['ActiveState'] != 'active':
            alerts.append(name + ' is not running')
    migration = read_json(STATE / 'storage-migration.json')
    r2 = read_json(STATE / 'r2/status.json')
    report.update(migration=migration, r2=r2)
    moving = migration.get('stage') not in (None, 'complete', 'failed')
    if migration.get('stage') == 'failed':
        alerts.append('Storage migration failed; inspect storage-migration.json before changing either copy')
    if not os.path.ismount(DATA_DISK):
        alerts.append(f'{DATA_DISK} is not mounted')
    for path, minimum in (('/', 15), (DATA_DISK, 30)):
        free = round(shutil.disk_usage(path).free / 1024**3, 1)
        report.setdefault('free_gib', {})[path] = free
        if free < minimum:
            alerts.append(f'{path}: only {free} GiB free')
    for name in ('layer-refresh', 'followup', 'ocr', 'r2-crops'):
        if moving and name in ('layer-refresh', 'followup', 'ocr'):
            continue  # Migration deliberately pauses writers and publication.
        if services['typology-' + name + '.timer']['ActiveState'] != 'active':
            alerts.append(name + ' timer is not active')

    def progress(name, value, running, grace=1800):
        old = previous.get('progress', {}).get(name, {})
        advanced = now if old.get('value') != value else old.get('advanced_at', now)
        report['progress'][name] = {'value': value, 'advanced_at': advanced}
        if running and now-advanced > grace:
            alerts.append(f'{name}: no recorded progress for {int((now-advanced)/60)} minutes')

    log = tail(STATE / 'storage-migration.log')
    transferred = re.findall(r'([\d,]+)\s+\d+%\s+', log)
    report['migration_last_progress'] = log.split('\r')[-1].strip()
    progress('migration', [migration.get('stage'), transferred[-1] if transferred else log[-200:]], moving, 3600)
    if moving and services['typology-storage-migration.service']['ActiveState'] != 'active':
        alerts.append('Migration is unfinished but its worker is not running')
    progress('r2', [r2.get('started_at'), r2.get('checked'), r2.get('activated')],
             services['typology-r2-crops.service']['ActiveState'] in ('active', 'activating')
             and r2.get('checked', 0) < r2.get('crops', 0))
    plan = read_json(ROOT / 'data/coruna/followup_plan.json')
    if plan.get('apple_run'):
        ocr = ROOT / 'data/coruna/apple/multicategory' / plan['apple_run'] / 'ocr.jsonl'
        progress('ocr', ocr.stat().st_size if ocr.exists() else 0,
                 services['typology-ocr.service']['ActiveState'] in ('active', 'activating'))
    report['expanded_dedupe'] = {source: read_json(ROOT / f'data/coruna/{source}/multicategory_dedupe/expanded_20261013/summary.json') for source in ('apple', 'gsv')}
    report['crop_delivery'] = read_json(ROOT / 'data/coruna/site/crop-storage.json')
    urls = ['http://127.0.0.1:8790/api/meta']
    if public := os.environ.get('TYPOLOGY_PUBLIC_URL'):
        urls.append(public.rstrip('/') + '/api/meta')
    for url in dict.fromkeys(urls):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'typology-background-watch/1.0'}), timeout=15) as response:
                metadata = json.load(response)
                if not metadata.get('query_snapshot'):
                    raise ValueError('Missing query snapshot')
            report.setdefault('viewer', {})[url] = metadata['query_snapshot']
        except Exception as error:
            alerts.append(f'{url}: {error}')
    STATE.mkdir(parents=True, exist_ok=True)
    temp = REPORT.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(REPORT)
    print(json.dumps(report, indent=2), flush=True)
    return bool(alerts)


if __name__ == '__main__':
    raise SystemExit(check())
