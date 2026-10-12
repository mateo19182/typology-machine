"""Check ongoing runs and perform the queued, CPU-safe pipeline handoffs."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/coruna'
PLAN = BASE / 'followup_plan.json'


def service(name):
    result = subprocess.run(['systemctl', '--user', 'show', name, '-p', 'ActiveState', '-p', 'Result'],
                            check=True, capture_output=True, text=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def completed(run):
    return sum(1 for _ in (run / 'views').glob('*/result.json'))


def validate_assets(run):
    config = json.loads((run / 'config.json').read_text())
    count = 0
    for path in (run / 'views').glob('*/result.json'):
        result = json.loads(path.read_text())
        if set(result.get('attempted_categories', [])) != set(config['prompts']):
            raise ValueError('Incomplete attempted categories')
        if not (ROOT / result['source_path']).is_file():
            raise ValueError('Missing original panorama')
        for prediction in result['predictions']:
            for field in ('crop', 'mask'):
                asset = (ROOT / prediction[field]).resolve()
                if not asset.is_relative_to(run.resolve()) or not asset.is_file():
                    raise ValueError('Missing or invalid prediction asset')
        count += 1
    if not count:
        raise ValueError('No completed views')
    return count


def tick(plan, apply=False):
    apple = BASE / 'apple/multicategory' / plan['apple_run']
    google = BASE / 'gsv/multicategory' / plan['google_run']
    backfill = BASE / 'apple/multicategory' / plan['backfill_run']
    rental = json.loads((BASE / 'vast_run.json').read_text())
    primary = service('typology-multicategory.service')
    status = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'apple': {'completed': completed(apple), 'planned': plan['apple_views'], **primary},
              'google': {'copied': completed(google), 'planned': plan['google_views'], 'rental_phase': rental['phase']},
              'backfill': {'completed': completed(backfill), **service('typology-backfill.service')},
              'dedupe': service('typology-dedupe-followup.service'),
              'free_gib': round(shutil.disk_usage(BASE).free / 1024**3, 1)}
    if apply:
        if (not plan.get('backfill_started') and primary == {'Result': 'success', 'ActiveState': 'inactive'}
                and status['apple']['completed'] == plan['apple_views'] and status['free_gib'] > 25):
            subprocess.run(['systemctl', '--user', 'start', '--no-block', 'typology-backfill.service'], check=True)
            plan['backfill_started'] = status['checked_at']
        # The watcher marks destroyed only after its final checksum transfer and asset checks.
        if not plan.get('google_published') and rental['phase'] == 'destroyed':
            if rental['run_id'] != plan['google_run']:
                raise ValueError('Rental run differs from queued Google run')
            count = validate_assets(google)
            if count != rental['completed_views']:
                raise ValueError('Google count differs from final verified transfer')
            dropin = Path.home() / '.config/systemd/user/typology-layer-refresh.service.d/followup.conf'
            dropin.parent.mkdir(parents=True, exist_ok=True)
            command = (f'{ROOT}/.venv/bin/python -u explore/export_layers.py coruna --run {apple} '
                       f'--extra-run {BASE}/apple/multicategory/8b6f53a1131e '
                       f'--extra-run {backfill} --extra-run {google} --if-changed --prune-hours 24')
            dropin.write_text('[Service]\nExecStart=\nExecStart=' + command + '\n')
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            subprocess.run(['systemctl', '--user', 'start', '--no-block', 'typology-layer-refresh.service'], check=True)
            plan['google_published'] = {'queued_at': status['checked_at'], 'views': count}
        temporary = PLAN.with_suffix('.tmp')
        temporary.write_text(json.dumps(plan, indent=2))
        temporary.replace(PLAN)
        (BASE / 'pipeline_status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not args.apply:
        tick(json.loads(PLAN.read_text()))
        return
    with (BASE / '.followup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tick(json.loads(PLAN.read_text()), args.apply)


if __name__ == '__main__':
    main()
