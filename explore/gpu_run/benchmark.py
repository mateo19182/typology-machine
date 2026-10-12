"""Choose worker concurrency from completed Google views on the rented GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    plan = json.loads(Path('run-plan.json').read_text())
    run = Path('data/coruna/gsv/multicategory') / plan['run_id']
    Path('logs').mkdir(exist_ok=True)
    measurements = []
    for workers in (1, 2, 4):
        before = set((run / 'views').glob('*/result.json'))
        command = [sys.executable, '-u', 'explore/multicategory_detect.py', 'coruna',
                   '--source', 'gsv', '--thin', '0', '--pano-manifest', 'data/coruna/gsv/gpu_manifest.json',
                   '--resume-run', str(run), '--no-legacy-reuse', '--include-people-pets',
                   '--workers', str(workers), '--torch-threads', '1', '--max-new-views', '32',
                   '--input-wait-seconds', '300']
        start = time.time()
        with open(f'logs/benchmark-{workers}.log', 'w') as log:
            try:
                result = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30', '600'] + command,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}, timeout=650)
            except subprocess.TimeoutExpired:
                raise RuntimeError('Benchmark did not terminate within its time cap')
        paths = set((run / 'views').glob('*/result.json')) - before
        if result.returncode or len(paths) != workers * 32:
            print(f'{workers} workers failed or incomplete; see benchmark log', flush=True)
            break
        records = [(p.stat().st_mtime, json.loads(p.read_text())) for p in paths]
        records.sort(key=lambda pair: pair[0])
        # Exclude model startup and each worker first-view warmup from throughput.
        warmup = workers * 4
        duration = records[-1][0] - records[warmup - 1][0]
        rate = (len(records) - warmup) / duration
        row = {'workers': workers, 'torch_threads': 1, 'views': len(paths),
               'wall_seconds': time.time() - start, 'steady_views_per_second': rate}
        measurements.append(row)
        print(json.dumps(row), flush=True)
    if not measurements:
        raise RuntimeError('No successful GPU benchmark')
    best = max(measurements, key=lambda m: m['steady_views_per_second'])
    # Prefer fewer processes when the measured difference is negligible.
    chosen = next(m for m in measurements if m['steady_views_per_second'] >= best['steady_views_per_second'] * .97)
    Path('logs/benchmark.json').write_text(json.dumps({'measurements': measurements, 'chosen': chosen}, indent=2))
    print(f"Selected {chosen['workers']} workers", flush=True)


if __name__ == '__main__':
    main()
