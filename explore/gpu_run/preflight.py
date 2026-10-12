"""Check a staged GPU run's inputs, pinned configuration and optional CUDA readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checksums', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--available-inputs', action='store_true',
                        help='check staged inputs while the checksum-verified fetch finishes')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'run-plan.json').read_text())
    manifest_path = ROOT / 'data/coruna/gsv/gpu_manifest.json'
    if digest(manifest_path) != plan['manifest_sha256']:
        raise ValueError('Source manifest changed')
    for name, checksum in plan['code_sha256'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Prepared code changed: {name}')
    manifest = json.loads(manifest_path.read_text())
    pending = 0
    for entry in manifest['panoramas']:
        path = ROOT / 'data/coruna/gsv/panos' / f"{entry['metadata']['id']}.jpg"
        if args.available_inputs and not path.exists():
            pending += 1
            continue
        if path.stat().st_size != entry['bytes']:
            raise ValueError(f'Image size mismatch: {path.name}')
        if args.checksums and digest(path) != entry['sha256']:
            raise ValueError(f'Image checksum mismatch: {path.name}')
    run = ROOT / 'data/coruna/gsv/multicategory' / plan['run_id']
    config = json.loads((run / 'config.json').read_text())
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12] != run.name:
        raise ValueError('Run configuration identity changed')
    if config['revision'] != plan['revision'] or len(config['prompts']) != 13:
        raise ValueError('Model or category configuration differs')
    if not config.get('portable_paths') or config['legacy_store_sign'] != 'disabled':
        raise ValueError('Run depends on nonportable legacy assets')
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise ValueError('Need at least 30 GiB free for output and model download')
    if args.gpu:
        import torch
        import transformers
        from transformers import Sam3Model, Sam3Processor
        import py360convert
        if sys.version_info < (3, 12):
            raise ValueError('Use Python 3.12 or newer')
        if transformers.__version__ != config['transformers']:
            raise ValueError('Transformers differs from the pinned run')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA with bfloat16 support is required')
        print('GPU:', torch.cuda.get_device_name(), 'CUDA:', torch.version.cuda)
    done = sum(1 for _ in (run / 'views').glob('*/result.json'))
    print(f"READY: {plan['panoramas']} panoramas, {plan['views']} views, 13 categories; {done} completed")
    if pending:
        print(f'{pending} inputs still pending; inference requires the fetch to stay ahead')


if __name__ == '__main__':
    main()
