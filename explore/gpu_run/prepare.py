"""Freeze downloaded current Google panoramas into a portable GPU transfer directory."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/coruna/gpu_google_current')
    args = parser.parse_args()
    out = args.out.resolve()
    inventory = out / 'data/coruna/gsv'
    inventory.mkdir(parents=True, exist_ok=True)
    manifest_path = inventory / 'gpu_manifest.json'
    # Freeze once: later downloads belong to a separate transfer/run inventory.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        from PIL import Image
        rows = {}
        for line in (ROOT / 'data/coruna/gsv/panos.jsonl').read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            if row.get('is_current'):
                rows[row['id']] = row
        entries = []
        for index, row in enumerate(sorted(rows.values(), key=lambda r: r['id']), 1):
            path = ROOT / 'data/coruna/gsv/panos' / f"{row['id']}.jpg"
            with Image.open(path) as image:
                image.verify()
            entries.append({'metadata': row, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
            if index % 500 == 0:
                print(f'Indexed {index}/{len(rows)} panoramas', flush=True)
        manifest = {'schema_version': 1, 'source': 'gsv', 'panoramas': entries}
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, sort_keys=True))
        temporary.replace(manifest_path)
    for entry in manifest['panoramas']:
        source = ROOT / 'data/coruna/gsv/panos' / f"{entry['metadata']['id']}.jpg"
        target = inventory / 'panos' / source.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            os.link(source, target)  # Same filesystem: no second 21 GB image copy.
    (inventory / 'panos.jsonl').write_text(''.join(json.dumps(e['metadata']) + '\n' for e in manifest['panoramas']))
    files = ['explore/multicategory_detect.py', 'explore/apple_detect.py', 'explore/gsv_detect.py']
    files += [str(p.relative_to(ROOT)) for p in (ROOT / 'explore/gpu_run').glob('*') if p.is_file()]
    for name in files:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    dependencies = ('torch', 'torchvision', 'transformers', 'numpy', 'pillow', 'pillow-heif',
                    'streetlevel', 'requests', 'py360convert', 'opencv-python-headless', 'scipy',
                    'huggingface-hub', 'safetensors', 'accelerate')
    (out / 'requirements-gpu.txt').write_text(''.join(f'{n}=={importlib.metadata.version(n)}\n' for n in dependencies))
    apple = json.loads((ROOT / 'data/coruna/apple/multicategory/765844a02f75/config.json').read_text())
    subprocess.run([sys.executable, str(out / 'explore/multicategory_detect.py'), 'coruna',
                    '--source', 'gsv', '--thin', '0', '--pano-manifest', str(manifest_path),
                    '--model-revision', apple['revision'], '--no-legacy-reuse',
                    '--include-people-pets', '--prepare-only'], cwd=out, check=True)
    runs = list((out / 'data/coruna/gsv/multicategory').glob('*/config.json'))
    if len(runs) != 1:
        raise ValueError('Expected exactly one prepared run in this bundle')
    run = runs[0].parent
    plan = {'run_id': run.name, 'panoramas': len(manifest['panoramas']),
            'views': len(manifest['panoramas']) * 8,
            'image_bytes': sum(e['bytes'] for e in manifest['panoramas']),
            'revision': apple['revision'], 'categories': 13,
            'manifest_sha256': sha256(manifest_path),
            'code_sha256': {name: sha256(out / name) for name in files}}
    (out / 'run-plan.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
