"""Export archive metadata and reviewed training runs, then upload explicit packages to HF."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home() / '.local/state/typology-machine/query'
HOME = Path(os.environ.get('TYPOLOGY_GRAFFITI_HOME', '/mnt/data/projects/typology-machine/graffiti-ai'))
PUBLIC_URL = 'https://typology.m19182.dev'
MODEL_TYPES = {
    'stable-diffusion-v1-5/stable-diffusion-v1-5': ('StableDiffusionPipeline', 'creativeml-openrail-m'),
    'black-forest-labs/FLUX.2-klein-base-4B': ('Flux2KleinPipeline', 'apache-2.0'),
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_file(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or unsafe package file: {name}')
    return path


@contextmanager
def package_directory(output):
    output = output.resolve()
    if output.exists():
        raise ValueError(f'Output already exists; preserve it and choose a new path: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        yield temporary
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def finish_package(directory, kind, **provenance):
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path != directory / 'package.json':
            files[path.relative_to(directory).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    write_json(directory / 'package.json', {
        'schema_version': 1, 'kind': kind,
        'created_at': datetime.now(timezone.utc).isoformat(),
        **provenance, 'files': files,
    })


def verify_package(directory):
    manifest = read_json(directory / 'package.json')
    if manifest['kind'] not in ('archive', 'graffiti', 'model') or not manifest['files']:
        raise ValueError('Invalid or empty package')
    expected = set(manifest['files']) | {'package.json'}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Package contains missing or unlisted files; rebuild it before uploading')
    for name, metadata in manifest['files'].items():
        path = safe_file(directory, name)
        if path.stat().st_size != metadata['bytes'] or sha256(path) != metadata['sha256']:
            raise ValueError(f'Package checksum mismatch: {name}')
    return manifest


def validate_reviewed_dataset(dataset):
    from PIL import Image
    summary = read_json(dataset / 'summary.json')
    if summary.get('reviewed') is not True:
        raise ValueError('Review the source dataset before exporting or training')
    rows = {}
    for split in ('train', 'validation'):
        directory = dataset / split
        metadata = directory / 'metadata.jsonl'
        expected_hash = summary.get('manifest_sha256', {}).get(split)
        if not expected_hash or sha256(metadata) != expected_hash:
            raise ValueError(f'Missing or changed reviewed {split} manifest hash')
        records = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
        if not records or len(records) != summary[split]:
            raise ValueError(f'Incorrect {split} count')
        names, identities = set(), set()
        for row in records:
            name = row['file_name']
            if Path(name).name != name or name in names or row['id'] in identities or not row.get('text'):
                raise ValueError(f'Duplicate, unsafe, or uncaptioned {split} image')
            names.add(name); identities.add(row['id'])
            with Image.open(safe_file(directory, name)) as image:
                image.verify()
        if {p.name for p in directory.iterdir() if p.is_file()} != names | {'metadata.jsonl'}:
            raise ValueError(f'Unlisted files in {split}; preserve the reviewed image inventory')
        rows[split] = records
    for field in ('id', 'group_id', 'panorama_id'):
        if {r[field] for r in rows['train']} & {r[field] for r in rows['validation']}:
            raise ValueError(f'Train/validation overlap in {field}')
    from prepare_graffiti_training import distance
    for train in rows['train']:
        for validation in rows['validation']:
            if distance(train, validation) < summary['validation_buffer_metres']:
                raise ValueError('Train/validation camera buffer violated')
            if (int(train['phash'], 16) ^ int(validation['phash'], 16)).bit_count() <= 12:
                raise ValueError('Cross-split perceptual duplicate')
    return summary, rows


def export_graffiti(dataset, output):
    summary, rows = validate_reviewed_dataset(dataset)
    with package_directory(output) as directory:
        for split in ('train', 'validation'):
            target = directory / split
            target.mkdir()
            shutil.copyfile(dataset / split / 'metadata.jsonl', target / 'metadata.jsonl')
            for row in rows[split]:
                shutil.copyfile(safe_file(dataset / split, row['file_name']), target / row['file_name'])
        shutil.copyfile(dataset / 'summary.json', directory / 'summary.json')
        exclusions = dataset / 'review-exclusions.json'
        if exclusions.exists():
            shutil.copyfile(exclusions, directory / exclusions.name)
        review = dataset / 'review.json'
        if review.exists():
            shutil.copyfile(review, directory / review.name)
        (directory / 'README.md').write_text(f'''---
language: [en]
task_categories: [text-to-image]
tags: [typology-machine, graffiti, coruna]
pretty_name: A Coruña reviewed graffiti crops
---

# A Coruña reviewed graffiti crops

The exact image dataset used by the Typology Machine graffiti LoRA experiment.
It contains {summary['train']} training images and {summary['validation']} validation images.
The original JPEG bytes, captions, provenance, reviewed manifests and split assignments are preserved.

## Loading

```python
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="path/to/downloaded/repository")
```

The Hub also supports `load_dataset("YOUR_NAMESPACE/YOUR_REPOSITORY", token=True)`.
`image` contains the crop and `text` is its training caption. Other columns retain
source IDs, provider links, camera coordinates, capture dates, detection confidence,
group and panorama IDs, crop geometry and review metadata.

## Selection and splits

Query snapshot: `{summary['snapshot']}`. Minimum detector confidence: {summary['minimum_confidence']}.
The preparation script rejects small, flat, soft and repeated crops, then buffers
training cameras at least {summary['validation_buffer_metres']} metres from validation cameras.
{summary.get('review_method', 'The source dataset has been marked reviewed.')}
Manifest and image checksums are in `summary.json` and `package.json`.
Caption strategy: {summary.get('caption_strategy', 'shared generic caption')}.
{('Descriptions were generated per image; uncertain lettering is retained separately and excluded from training captions.' if summary.get('caption_strategy') else 'All images share a generic caption. More descriptive, individually reviewed captions would help a future FLUX run.')}

## Limitations and rights

Apple Look Around and Google Street View supply the source imagery. The underlying
graffiti can have separate artist rights. Redistribution permission remains unresolved;
this package makes no grant of rights to the imagery. Keep this repository private
until those rights are established. The project's MIT code license does not license these images.
Review does not prove distinct artworks or artist-independent evaluation.
Labels, camera positions and OCR can be wrong. Validation is geographically buffered,
but it is from the same city. Excluded images and original full street photographs are absent.

Project: https://github.com/mateo19182/typology-machine
''')
        finish_package(directory, 'graffiti', query_snapshot=summary['snapshot'],
                       counts={s: len(r) for s, r in rows.items()}, rights_status='unresolved')
    return verify_package(output)


def crop_url(path, public_url, crop_base):
    if not path:
        return None
    if not re.fullmatch(r'/?(?:img|proposal-assets)/[A-Za-z0-9_./-]+\.jpg', path):
        return None
    if any(p in ('.', '..') for p in path.split('/')):
        return None
    return (crop_base or public_url).rstrip('/') + '/' + quote(path.lstrip('/'), safe='/')


def archive_record(row, group, public_url, crop_base):
    def number(value):
        return float(value) if value is not None and math.isfinite(float(value)) else None
    def string(value):
        return str(value) if value is not None else None
    return {
        'id': str(row['id']), 'category': str(row['category']), 'source': str(row['source']),
        'run_id': string(row.get('run_id')), 'view_id': string(row.get('view_id')),
        'group_id': str(group), 'status': string(row.get('status')),
        'detection_score': number(row.get('score')), 'prompt': string(row.get('prompt')),
        'capture_date': string(row.get('date', row.get('d'))),
        'longitude': number(row.get('lo')), 'latitude': number(row.get('la')),
        'location_basis': string(row.get('location_basis')),
        'barrio': string(row.get('barrio')), 'barrio_id': string(row.get('barrio_id')),
        'source_url': string(row.get('provider_url', row.get('u'))),
        'crop_path': string(row.get('crop')),
        'crop_url': crop_url(row.get('crop'), public_url, crop_base),
        'crop_width': row.get('width', row.get('w')), 'crop_height': row.get('height', row.get('h')),
        'bbox': [float(v) for v in row['bx']] if row.get('bx') else None,
        'ocr_text': string(row.get('text')), 'ocr_confidence': number(row.get('ocr_confidence')),
        'ocr_model': string(row.get('ocr_model')),
    }


def export_archive(state, site, output, public_url=PUBLIC_URL, shard_rows=50000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if shard_rows < 1:
        raise ValueError('shard_rows must be positive')
    meta = read_json(state / 'current.json')
    snapshot = meta['query_snapshot']
    if not re.fullmatch(r'[a-f0-9]+', snapshot):
        raise ValueError('Invalid query snapshot')
    storage = site / 'crop-storage.json'
    crop_base = read_json(storage).get('base_url') if storage.exists() else None
    example = archive_record({'id': '', 'category': '', 'source': ''}, '', public_url, crop_base)
    schema = pa.schema([(key, pa.float64() if key in ('detection_score', 'longitude', 'latitude', 'ocr_confidence')
                        else pa.int64() if key in ('crop_width', 'crop_height')
                        else pa.list_(pa.float64()) if key == 'bbox' else pa.string()) for key in example])
    counts, sources, shards = Counter(), Counter(), Counter()
    uri = (state / f'{snapshot}.sqlite3').resolve().as_uri() + '?mode=ro'
    with package_directory(output) as directory, closing(sqlite3.connect(uri, uri=True)) as db:
        total = db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        pending, category = [], None
        def flush():
            if not pending:
                return
            path = directory / 'data' / category / f'{shards[category]:05d}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(pending, schema=schema), path, compression='zstd')
            shards[category] += 1
            pending.clear()
        cursor = db.execute('''SELECT r.payload,o.group_key,o.category FROM observations o
            JOIN records r ON r.id=o.id ORDER BY o.category,o.id''')
        for payload, group, next_category in cursor:
            if next_category != category:
                flush(); category = next_category
                if not re.fullmatch(r'[a-z_]+', category):
                    raise ValueError('Invalid category directory')
            record = archive_record(json.loads(payload), group, public_url, crop_base)
            pending.append(record)
            counts[category] += 1; sources[record['source']] += 1
            if len(pending) >= shard_rows:
                flush()
                print(json.dumps({'category': category, 'exported': sum(counts.values()), 'total': total}), flush=True)
        flush()
        if sum(counts.values()) != total:
            raise ValueError('Missing joined archive records')
        write_json(directory / 'snapshot.json', meta)
        write_json(directory / 'summary.json', {'observations': total, 'categories': dict(counts),
                   'sources': dict(sources), 'query_snapshot': snapshot, 'embedded_images': False,
                   'crop_base_url': crop_base, 'rights_status': 'unresolved'})
        import yaml
        configs = [{'config_name': 'all', 'default': True,
                    'data_files': [{'split': 'observations', 'path': 'data/*/*.parquet'}]}]
        configs += [{'config_name': name, 'data_files': [{'split': 'observations',
                    'path': f'data/{name}/*.parquet'}]} for name in sorted(counts)]
        frontmatter = yaml.safe_dump({'language': ['en', 'es', 'gl'],
            'tags': ['typology-machine', 'coruna', 'street-imagery', 'detections', 'ocr'],
            'pretty_name': 'A Coruña typology observations', 'configs': configs}, sort_keys=False)
        (directory / 'README.md').write_text(f'''---
{frontmatter}---

# A Coruña typology observations

{total:,} indexed observations across {len(counts)} categories from query snapshot `{snapshot}`.
These are detector proposals with OCR where available. They are not verified labels.
The export freezes one existing query snapshot and does not rerun detection or OCR.

```python
from datasets import load_dataset
records = load_dataset("YOUR_NAMESPACE/YOUR_REPOSITORY", "graffiti", split="observations", token=True)
```

The `all` configuration includes every category. Category configurations select
the same rows without duplicating files. The `observations` split is an inventory,
not a training/validation split. Build reviewed, grouped splits before model training.

## Contents

Parquet rows contain IDs, categories, detector scores, prompts, run/view/group IDs,
capture dates, camera coordinates, barrios, normalized view bounding boxes when
available, provider links, crop dimensions and OCR text/confidence/model.
`crop_url` links to the existing crop deployment. The image bytes and full street
photographs are not embedded. Links can change or disappear and do not make this
a self-contained image dataset. `snapshot.json` records detector revisions and
run provenance. `package.json` records every exported file's SHA-256 checksum.

## Limitations and rights

Apple Look Around and Google Street View supply the imagery. Provider and artwork
rights remain separate from the project's MIT code license. Redistribution rights
are unresolved. Keep the repository private until those rights are established.
This dataset grants no rights to the linked images.
Coordinates usually identify the camera. Cross-view grouping is partial and can
be wrong. Repeated views and wrong labels remain. Detector scores are not calibrated
accuracy estimates. OCR is experimental. People and pets are object proposals;
group IDs do not establish individual identity.

Project: https://github.com/mateo19182/typology-machine
''')
        finish_package(directory, 'archive', query_snapshot=snapshot, observations=total,
                       categories=dict(counts), rights_status='unresolved')
    return verify_package(output)


def export_model(home, output, dataset_repo, dataset_revision=None):
    summary, _ = validate_reviewed_dataset(home / 'dataset')
    training = read_json(home / 'training.json')
    if training.get('stage') != 'trained':
        raise ValueError('Only completed training runs can be exported')
    model = training['model']
    if model not in MODEL_TYPES:
        raise ValueError('Add the base model pipeline and license to MODEL_TYPES first')
    if training.get('train_count') != summary['train'] or training.get('validation_count') != summary['validation']:
        raise ValueError('Training counts do not match the reviewed dataset')
    pipeline, license_name = MODEL_TYPES[model]
    with package_directory(output) as directory:
        weights = safe_file(home / 'lora', 'pytorch_lora_weights.safetensors')
        shutil.copyfile(weights, directory / weights.name)
        # Training files are explicit allowlisted artifacts; never upload the entire home.
        allowed = ('stage', 'model', 'revision', 'steps', 'step', 'seed', 'train_count',
                   'validation_count', 'validation_loss_before', 'validation_loss_after',
                   'rank', 'resolution', 'loss', 'started', 'finished')
        exported_training = {k: training[k] for k in allowed if k in training}
        if training.get('recipe'):
            exported_training['recipe'] = {k: training['recipe'][k] for k in
                ('diffusers_revision', 'trainer', 'quantization', 'evaluation', 'dataset_manifest_sha256',
                 'caption_strategy', 'text_max_sequence_length', 'caption_dropout')
                if k in training['recipe']}
            command = training['recipe'].get('command', [])
            numeric_flags = {'learning_rate': float, 'train_batch_size': int,
                'gradient_accumulation_steps': int, 'lr_warmup_steps': int,
                'checkpointing_steps': int, 'checkpoints_total_limit': int,
                'max_sequence_length': int, 'caption_dropout': float}
            parameters = {}
            for name, convert in numeric_flags.items():
                flag = '--' + name
                if flag in command:
                    parameters[name] = convert(command[command.index(flag) + 1])
            for name in ('mixed_precision', 'lr_scheduler'):
                flag = '--' + name
                if flag in command:
                    parameters[name] = command[command.index(flag) + 1]
            for name in ('gradient_checkpointing', 'cache_latents', 'offload',
                         'use_8bit_adam', 'use_aspect_ratio_buckets', 'skip_final_inference'):
                parameters[name] = '--' + name in command
            exported_training['recipe']['parameters'] = parameters
        recorded_hashes = training.get('dataset_manifest_sha256')
        if recorded_hashes and recorded_hashes != summary['manifest_sha256']:
            raise ValueError('Reviewed dataset differs from the recorded training manifests')
        write_json(directory / 'training.json', exported_training)
        provenance = {'dataset_repo': dataset_repo, 'dataset_revision': dataset_revision,
                      'dataset_snapshot': summary['snapshot'], 'manifest_sha256': summary['manifest_sha256'],
                      'model': model, 'model_revision': training['revision']}
        write_json(directory / 'provenance.json', provenance)
        for name in ('base-sample.png', 'lora-sample.png', 'evaluation-comparison.jpg', 'evaluation.json', 'metrics.json'):
            if (home / name).is_file():
                shutil.copyfile(home / name, directory / name)
        if (home / 'evaluation.json').is_file():
            report = read_json(home / 'evaluation.json')
            for sample in report['samples']:
                name = sample['file_name']
                if not re.fullmatch(r'[1-9][0-9]*-(?:base|lora)\.png', name):
                    raise ValueError('Unsafe evaluation image name')
                source = safe_file(home / 'evaluation', name)
                (directory / 'evaluation').mkdir(exist_ok=True)
                shutil.copyfile(source, directory / 'evaluation' / name)
        before, after = training.get('validation_loss_before'), training.get('validation_loss_after')
        evaluation = (f'Held-out denoising loss changed from {before:.6f} to {after:.6f}. '
                      'This does not establish better visual quality.' if before is not None and after is not None
                      else 'No held-out quality metric is reported for this run.')
        variant = ', variant="fp16"' if pipeline == 'StableDiffusionPipeline' else ''
        inference_dtype = 'float16' if pipeline == 'StableDiffusionPipeline' else 'bfloat16'
        per_image = training.get('recipe', {}).get('caption_strategy') == 'per-image captions'
        prompt_advice = ('Use ordinary descriptive prompts. This run has no special trigger word; unverified lettering '
                         'was excluded from its individual training captions.' if per_image
                         else 'Use the trigger `c0runa graffiti` in prompts.')
        prompt_example = ('blue graffiti lettering on a concrete wall' if per_image
                          else 'a photograph of c0runa graffiti, blue lettering on a concrete wall')
        example_options = (f", height=512, width=512, num_inference_steps=30, guidance_scale=4.0, "
                           f"max_sequence_length={training.get('recipe', {}).get('text_max_sequence_length', 512)}"
                           if pipeline == 'Flux2KleinPipeline' else '')
        (directory / 'README.md').write_text(f'''---
license: {license_name}
base_model: {model}
base_model_relation: adapter
library_name: diffusers
pipeline_tag: text-to-image
datasets: [{dataset_repo}]
tags: [lora, typology-machine, coruna, graffiti]
---

# A Coruña graffiti LoRA

A small experimental adapter trained on {summary['train']} reviewed graffiti crops,
with {summary['validation']} held out. The base model is `{model}` at revision
`{training['revision']}`. This repository contains adapter weights, not the base model.
{prompt_advice}

## Training and evaluation

{training['steps']} optimization steps; rank {training['rank']}; {training['resolution']}px
training inputs; seed {training['seed']}. `training.json` records the completed run.
`provenance.json` links to the exact dataset manifests and Hub commit when available.
{evaluation}
The original SD 1.5 experiment evaluated at most 16 of the 29 held-out images.
The validation crops are from the same city; they do not prove artist-independent generalization.
Any included base/adapter samples are illustrations, not a quality benchmark.

## Loading

```python
import torch
from diffusers import {pipeline}
pipe = {pipeline}.from_pretrained(
    "{model}", revision="{training['revision']}",
    torch_dtype=torch.{inference_dtype}{variant},
)
pipe.load_lora_weights("YOUR_NAMESPACE/YOUR_REPOSITORY", token=True)
pipe.enable_model_cpu_offload()
image = pipe("{prompt_example}"{example_options}).images[0]
```

## Limitations and rights

Spelling, local style and novelty are unreliable. Review of the training crops
does not establish image redistribution or artwork rights. Those rights remain
unresolved; the linked image dataset stays private. Follow the base model's license
and usage terms. The original SD 1.5 serving pipeline retains its safety checker.

Project: https://github.com/mateo19182/typology-machine
''')
        finish_package(directory, 'model', **provenance)
    return verify_package(output)


def upload_package(directory, repo_id, public=False):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    manifest = verify_package(directory)
    api = HfApi()
    repo_type = 'model' if manifest['kind'] == 'model' else 'dataset'
    api.create_repo(repo_id, repo_type=repo_type, private=not public, exist_ok=True)
    info = api.repo_info(repo_id, repo_type=repo_type)
    if info.private != (not public):
        raise ValueError('Existing repository visibility differs; change it explicitly before uploading')
    existing = set(api.list_repo_files(repo_id, repo_type=repo_type)) - {'.gitattributes'}
    files = set(manifest['files']) | {'package.json'}
    if existing:
        try:
            previous_path = hf_hub_download(repo_id, 'package.json', repo_type=repo_type, revision=info.sha)
        except RepositoryNotFoundError:
            raise
        except Exception as error:
            raise ValueError('Existing repository has no readable Typology package; refusing to replace it') from error
        previous = read_json(Path(previous_path))
        if previous.get('kind') != manifest['kind']:
            raise ValueError('Existing repository has a different package kind')
        owned = set(previous['files']) | {'package.json'}
        if existing - owned - files:
            raise ValueError('Existing repository has files outside the previous package')
        # A streamed upload can leave new files committed before its final manifest.
        # Resume only additions that match this prepared package's content hashes.
        partial = existing - owned
        if partial:
            metadata = {f.rfilename: f for f in api.repo_info(repo_id, repo_type=repo_type,
                        revision=info.sha, files_metadata=True).siblings}
            for name in partial:
                item = metadata[name]
                local = manifest['files'].get(name)
                if local is None or item.size != local['bytes']:
                    raise ValueError(f'Unowned remote file differs from prepared package: {name}')
                if item.lfs:
                    matches = item.lfs.sha256 == local['sha256']
                else:
                    remote_path = hf_hub_download(repo_id, name, repo_type=repo_type, revision=info.sha)
                    matches = sha256(Path(remote_path)) == local['sha256']
                if not matches:
                    raise ValueError(f'Unowned remote file differs from prepared package: {name}')
    # The Hub's Xet uploader adapts commit sizes and resumes completed files. A
    # single create_commit call can time out with thousands of small JPEGs.
    commit = api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=directory,
                               allow_patterns=sorted(files), delete_patterns=sorted(existing - files),
                               parent_commit=info.sha,
                               commit_message=f'Export Typology Machine {manifest["kind"]} package')
    # Verify paths, sizes and every LFS content digest against the committed revision.
    remote = {f.rfilename: f for f in api.repo_info(repo_id, repo_type=repo_type,
                                                  revision=commit.oid, files_metadata=True).siblings}
    if set(remote) - {'.gitattributes'} != files:
        raise ValueError('Committed repository file inventory differs from the package')
    for name, metadata in manifest['files'].items():
        item = remote[name]
        if item.size != metadata['bytes'] or item.lfs and item.lfs.sha256 != metadata['sha256']:
            raise ValueError(f'Committed artifact metadata differs: {name}')
    receipt = {'repo_id': repo_id, 'repo_type': repo_type, 'private': not public,
               'revision': commit.oid, 'url': str(commit.repo_url),
               'package_sha256': sha256(directory / 'package.json')}
    write_json(directory.parent / f'{directory.name}-upload.json', receipt)
    return receipt


def publish_run(home, output, namespace=None, upload=False, public_model=False, dataset_repo_id=None):
    if namespace is None:
        from huggingface_hub import HfApi
        namespace = HfApi().whoami()['name']
    settings = read_json(home / 'publishing.json') if (home / 'publishing.json').exists() else {}
    dataset_repo = dataset_repo_id or settings.get('dataset_repo_id') or f'{namespace}/coruna-graffiti-reviewed'
    model_name = 'sd15' if read_json(home / 'training.json')['model'] == next(iter(MODEL_TYPES)) else 'flux2-klein'
    model_repo = settings.get('model_repo_id') or f'{namespace}/coruna-graffiti-{model_name}-lora'
    # Fail before either upload if the complete run cannot be packaged.
    output.mkdir(parents=True, exist_ok=True)
    export_graffiti(home / 'dataset', output / 'dataset')
    export_model(home, output / 'model', dataset_repo)
    results = {'dataset_repo': dataset_repo, 'model_repo': model_repo, 'output': str(output)}
    if upload:
        results['dataset'] = upload_package(output / 'dataset', dataset_repo)
        # Rebuild only the newly prepared local model package with the dataset commit link.
        shutil.rmtree(output / 'model')
        export_model(home, output / 'model', dataset_repo, results['dataset']['revision'])
        results['model'] = upload_package(output / 'model', model_repo, public=public_model)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    archive = commands.add_parser('archive', help='Export the full metadata snapshot as category Parquet shards')
    archive.add_argument('--state', type=Path, default=STATE)
    archive.add_argument('--site', type=Path, default=ROOT / 'data/coruna/site')
    archive.add_argument('--public-url', default=PUBLIC_URL)
    archive.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='Package the reviewed dataset and completed adapter together')
    run.add_argument('--home', type=Path, default=HOME)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--namespace')
    run.add_argument('--upload', action='store_true')
    run.add_argument('--public-model', action='store_true')
    run.add_argument('--dataset-repo-id', help='Versioned dataset repository; also configurable in HOME/publishing.json')
    upload = commands.add_parser('upload', help='Verify and upload a prepared package')
    upload.add_argument('--package', type=Path, required=True)
    upload.add_argument('--repo-id', required=True)
    upload.add_argument('--public', action='store_true')
    args = parser.parse_args()
    if args.action == 'archive':
        result = export_archive(args.state, args.site, args.output, args.public_url)
        result = {k: v for k, v in result.items() if k != 'files'}
    elif args.action == 'run':
        result = publish_run(args.home, args.output, args.namespace, args.upload, args.public_model, args.dataset_repo_id)
    else:
        result = upload_package(args.package, args.repo_id, args.public)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
