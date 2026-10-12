"""Keep this rental bounded, retrieve completed outputs, and end billing after verification."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'data/coruna/vast_run.json'
KEYS = Path.home() / '.config/typology-machine'
SESSION = requests.Session()
SESSION.headers['Authorization'] = 'Bearer ' + (KEYS / 'vast-api-key').read_text().strip()


def api(method, instance_id, **kwargs):
    response = SESSION.request(method, f'https://console.vast.ai/api/v0/instances/{instance_id}/', timeout=30, **kwargs)
    response.raise_for_status()
    data = response.json()
    if not data.get('success', False):
        raise RuntimeError(f'Instance API action failed: {data.get("error")}')
    return data


def ssh_args(state):
    return ['ssh', '-i', str(KEYS / 'vast-ssh'), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-o', 'StrictHostKeyChecking=accept-new', '-o', f'UserKnownHostsFile={KEYS}/known_hosts',
            '-p', str(state['ssh_port'])]


def remote(state, command, timeout=90):
    return subprocess.run(ssh_args(state) + ['root@' + state['ssh_host'], command],
                          text=True, capture_output=True, timeout=timeout, check=True).stdout


def sync(state, final=False):
    destination = ROOT / 'data/coruna/gsv/multicategory' / state['run_id']
    destination.mkdir(parents=True, exist_ok=True)
    source = f"root@{state['ssh_host']}:/workspace/typology/data/coruna/gsv/multicategory/{state['run_id']}/"
    args = ['rsync', '-a', '--partial', '--timeout=90', '-e', shlex.join(ssh_args(state))]
    if final:
        args.append('--checksum')
    subprocess.run(args + [source, str(destination) + '/'], timeout=1800, check=True)
    if final:
        check = subprocess.run(args + ['--dry-run', '--itemize-changes', source, str(destination) + '/'],
                               timeout=1800, check=True, capture_output=True, text=True)
        if check.stdout.strip():
            raise RuntimeError('Remote outputs differ after final copy')
        config = json.loads((destination / 'config.json').read_text())
        frozen = ROOT / 'data/coruna/gpu_google_current/data/coruna/gsv/multicategory' / state['run_id'] / 'config.json'
        if config != json.loads(frozen.read_text()):
            raise RuntimeError('Retrieved run configuration differs from the frozen run')
        count = 0
        for path in (destination / 'views').glob('*/result.json'):
            record = json.loads(path.read_text())
            if set(record.get('attempted_categories', [])) != set(config['prompts']):
                raise RuntimeError('Completed view did not attempt every configured category')
            if not (ROOT / record['source_path']).is_file():
                raise RuntimeError('Local source panorama missing')
            for prediction in record['predictions']:
                for field in ('crop', 'mask'):
                    asset = (ROOT / prediction[field]).resolve()
                    if not asset.is_relative_to(destination) or not asset.is_file():
                        raise RuntimeError('Missing or invalid output asset')
            count += 1
        print(f'Final copy verified: {count} completed views', flush=True)
        return count
    return sum(1 for _ in (destination / 'views').glob('*/result.json'))


def upload_failed_inputs(state):
    report = remote(state, 'cat /workspace/typology/logs/fetch-failures.json 2>/dev/null || echo "[]"')
    failed = json.loads(report)
    record = ROOT / 'data/coruna/vast_uploaded_inputs.json'
    uploaded = set(json.loads(record.read_text())) if record.exists() else set()
    manifest = json.loads((ROOT / 'data/coruna/gpu_google_current/data/coruna/gsv/gpu_manifest.json').read_text())
    allowed = {entry['metadata']['id'] for entry in manifest['panoramas']}
    needed = sorted({entry['id'] for entry in failed} - uploaded)
    if not set(needed) <= allowed:
        raise ValueError('Fetch report contains an input outside the frozen inventory')
    if needed:
        subprocess.run(['rsync', '-a', '--checksum', '--timeout=90', '--files-from=-',
                        '-e', shlex.join(ssh_args(state)), str(ROOT / 'data/coruna/gsv/panos') + '/',
                        f"root@{state['ssh_host']}:/workspace/typology/data/coruna/gsv/panos/"],
                       input=''.join(pano + '.jpg\n' for pano in needed), text=True, check=True, timeout=900)
        uploaded.update(needed)
        record.write_text(json.dumps(sorted(uploaded)))
        print(f'Uploaded {len(needed)} frozen inputs unavailable by direct fetch', flush=True)


def main():
    failures = 0
    last_count = None
    last_progress = time.monotonic()
    while True:
        state = json.loads(STATE.read_text())
        if state['phase'] in ('destroyed', 'stopped'):
            return
        deadline = state['deadline'] if state['phase'] == 'running' else min(state['deadline'], state['created_at'] + 3600)
        expired = time.time() >= deadline
        try:
            if state.get('direct_fetch') and not expired:
                upload_failed_inputs(state)
            if state['phase'] == 'running':
                if expired:
                    remote(state, "cd /workspace/typology; kill -TERM -- -$(cat logs/run.pid) 2>/dev/null || true; sleep 15", timeout=45)
                status = remote(state, 'cd /workspace/typology; if test -f logs/run.exit; then cat logs/run.exit; elif kill -0 "$(cat logs/run.pid)" 2>/dev/null; then echo running; else echo crashed; fi').strip()
                if status != 'running' or expired:
                    if expired and status == 'running':
                        remote(state, "cd /workspace/typology; kill -KILL -- -$(cat logs/run.pid) 2>/dev/null || true")
                    count = sync(state, final=True)
                    # Logs are copied before destruction too.
                    subprocess.run(['rsync', '-a', '-e', shlex.join(ssh_args(state)),
                        f"root@{state['ssh_host']}:/workspace/typology/logs/", str(ROOT / 'data/coruna/vast_logs') + '/'], check=True, timeout=120)
                    api('DELETE', state['instance_id'])
                    state.update(phase='destroyed', completed_views=count, finished_at=time.time(), exit_status=status)
                    STATE.write_text(json.dumps(state, indent=2))
                    print('Rental destroyed; compute and storage billing ended', flush=True)
                    return
                count = sync(state)
                if count != last_count:
                    last_count, last_progress = count, time.monotonic()
                    print(f'Synced {count} completed views', flush=True)
                elif time.monotonic() - last_progress > 900:
                    state['deadline'] = min(state['deadline'], time.time())
                    state['reason'] = 'No completed views for 15 minutes'
                    STATE.write_text(json.dumps(state, indent=2))
                    print('No progress for 15 minutes; ending the run on next check', flush=True)
                failures = 0
            elif expired:
                api('PUT', state['instance_id'], json={'state': 'stopped'})
                state.update(phase='stopped', reason='setup deadline')
                STATE.write_text(json.dumps(state, indent=2))
                print('Deadline reached: compute stopped; storage retained', flush=True)
                return
        except Exception as error:
            failures += 1
            print(f'Watch attempt failed: {type(error).__name__}: {error}', flush=True)
            if expired or failures >= 5:
                try:
                    api('PUT', state['instance_id'], json={'state': 'stopped'})
                    state.update(phase='stopped', reason='retrieval failed; outputs retained')
                    STATE.write_text(json.dumps(state, indent=2))
                    print('Compute stopped; remote outputs retained for recovery', flush=True)
                    return
                except Exception as stop_error:
                    print(f'Stop failed: {type(stop_error).__name__}; retrying', flush=True)
        time.sleep(120)


if __name__ == '__main__':
    if '--hard-stop' in sys.argv:
        state = json.loads(STATE.read_text())
        if state['phase'] not in ('destroyed', 'stopped'):
            api('PUT', state['instance_id'], json={'state': 'stopped'})
            state.update(phase='stopped', reason='independent rental deadline; outputs retained')
            STATE.write_text(json.dumps(state, indent=2))
            print('Independent deadline stopped compute; storage retained', flush=True)
    else:
        main()
