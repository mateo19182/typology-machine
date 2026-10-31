"""Prepare or run a separate, pinned FLUX.2 klein Base 4B graffiti LoRA experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from publish_huggingface import HOME, export_graffiti, read_json, validate_reviewed_dataset, verify_package, write_json

MODEL = 'black-forest-labs/FLUX.2-klein-base-4B'
MODEL_REVISION = 'a3b4f4849157f664bdbc776fd7453c2783562f4d'
DIFFUSERS_REVISION = '916a5791ed5f29d966200d2eee53f06862ec4434'
DEFAULT_HOME = Path('/mnt/data/projects/typology-machine/graffiti-flux2')


def prepare(dataset, home, steps=1200, seed=19182):
    if steps < 1:
        raise ValueError('Steps must be positive')
    if home.exists():
        raise ValueError('Choose a new FLUX experiment directory')
    # ImageFolder preserves the reviewed train/validation split. The official trainer
    # consumes only dataset["train"], including each record's text caption.
    export_graffiti(dataset, home / 'dataset')
    summary, rows = validate_reviewed_dataset(home / 'dataset')
    captions = {row['text'] for row in rows['train']}
    shared_caption = next(iter(captions)) if len(captions) == 1 else None
    quantization = home / 'nf4.json'
    write_json(quantization, {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                              'bnb_4bit_use_double_quant': True})
    script = home / 'diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein.py'
    command = [str(home / 'venv/bin/python'), str(script),
        '--pretrained_model_name_or_path', MODEL, '--revision', MODEL_REVISION,
        '--dataset_name', str(home / 'dataset'), '--image_column', 'image',
        '--instance_prompt', shared_caption or summary['caption'], '--output_dir', str(home / 'lora'),
        '--seed', str(seed), '--resolution', '512', '--use_aspect_ratio_buckets',
        '--train_batch_size', '1', '--gradient_accumulation_steps', '1', '--gradient_checkpointing',
        '--max_train_steps', str(steps), '--rank', '16', '--learning_rate', '0.0001',
        '--lr_scheduler', 'constant', '--lr_warmup_steps', '0', '--mixed_precision', 'bf16',
        '--cache_latents', '--offload', '--use_8bit_adam',
        '--bnb_quantization_config_path', str(quantization),
        '--checkpointing_steps', '250', '--checkpoints_total_limit', '6',
        '--validation_prompt', summary['caption'] + ', abstract blue lettering on weathered concrete',
        '--num_validation_images', '2', '--validation_epochs', '1', '--skip_final_inference', '--report_to', 'tensorboard']
    if shared_caption is None:
        # The official trainer caches custom embeddings on GPU. Short descriptions
        # do not need 512 text tokens per image on a 24 GB card.
        command += ['--caption_column', 'text', '--max_sequence_length', '128', '--caption_dropout', '0.1']
        # Keep the encoder resident during precomputation. Moving its 8 GB weights
        # for every caption is unnecessary with the bounded cache and idle 3090.
        command.remove('--offload')
    plan = {'model': MODEL, 'revision': MODEL_REVISION, 'diffusers_revision': DIFFUSERS_REVISION,
            'trainer': 'examples/dreambooth/train_dreambooth_lora_flux2_klein.py',
            'steps': steps, 'seed': seed, 'rank': 16, 'resolution': 512,
            'train_count': summary['train'], 'validation_count': summary['validation'],
            'dataset_manifest_sha256': summary['manifest_sha256'], 'command': command,
            'caption_strategy': 'shared prompt encoded once' if shared_caption else 'per-image captions',
            'text_max_sequence_length': 512 if shared_caption else 128,
            'caption_dropout': 0.0 if shared_caption else 0.1,
            'evaluation': 'Fixed-seed prompt previews only; no held-out denoising loss computed by this trainer.',
            'quantization': 'NF4, not FP8; RTX 3090 does not have FP8 tensor cores.'}
    write_json(home / 'training-plan.json', plan)
    write_json(home / 'training.json', {**{k: plan[k] for k in
        ('model', 'revision', 'steps', 'seed', 'rank', 'resolution', 'train_count', 'validation_count')},
        'stage': 'prepared', 'dataset_manifest_sha256': summary['manifest_sha256']})
    return plan


def train(home, publish=False, namespace=None):
    plan = read_json(home / 'training-plan.json')
    verify_package(home / 'dataset')
    summary, _ = validate_reviewed_dataset(home / 'dataset')
    if summary['manifest_sha256'] != plan['dataset_manifest_sha256']:
        raise ValueError('Dataset changed after preparation')
    if read_json(home / 'training.json')['stage'] != 'prepared' or (home / 'lora').exists():
        raise ValueError('This run already started; preserve it and prepare a new experiment')
    root = home / 'diffusers'
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != plan['diffusers_revision'] or subprocess.check_output(
            ['git', '-C', str(root), 'diff', 'HEAD', '--', 'src', 'examples/dreambooth'], text=True).strip():
        raise ValueError('Diffusers source differs from the pinned trainer revision')
    occupied = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(value.strip()) > 1024 for value in occupied.splitlines()):
        raise ValueError('GPU is already in use; run training after the current GPU work finishes')
    state = read_json(home / 'training.json')
    state.update(stage='training', started=time.time(), recipe=plan)
    write_json(home / 'training.json', state)
    try:
        subprocess.run(plan['command'], check=True)
        if not (home / 'lora/pytorch_lora_weights.safetensors').is_file():
            raise ValueError('Trainer completed without the expected final adapter')
    except Exception:
        state.update(stage='failed', finished=time.time())
        write_json(home / 'training.json', state)
        raise
    state.update(stage='trained', step=plan['steps'], finished=time.time())
    write_json(home / 'training.json', state)
    if publish:
        from publish_huggingface import publish_run
        return publish_run(home, home / f'hub-{int(time.time())}', namespace, upload=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train'))
    parser.add_argument('--dataset', type=Path, default=HOME / 'dataset')
    parser.add_argument('--home', type=Path, default=DEFAULT_HOME)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=19182)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--namespace')
    args = parser.parse_args()
    home = args.home.resolve()
    result = prepare(args.dataset, home, args.steps, args.seed) if args.action == 'prepare' else train(home, args.publish, args.namespace)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
