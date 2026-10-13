# Graffiti model experiment

`graffiti.html` generates 512px images with Stable Diffusion 1.5 and a small LoRA trained on selected archive crops. Model weights, selected images and outputs stay outside Git, under `/mnt/data/projects/typology-machine/graffiti-ai` by default. Set `TYPOLOGY_GRAFFITI_HOME` or pass `--home` to change it.

The first run used 225 training images and 29 held-out images, 300 optimization steps, rank 4, 512px inputs, and seed 19182. Only UNet attention adapters train; the base model stays frozen. The held-out denoising loss changed from 0.154809 to 0.154611. That tiny change does not establish improved visual quality. Fixed-seed base and adapted samples both generated successfully and differed visibly.

## Environment and model

Keep optional image-generation packages out of the website's normal environment. On the deployed machine, an isolated environment on the data disk reuses the existing project's Torch/Transformers packages through a `.pth` file, while installing Diffusers and PEFT separately:

```bash
export GRAFFITI_HOME=/mnt/data/projects/typology-machine/graffiti-ai
uv venv --python .venv/bin/python "$GRAFFITI_HOME/venv"
"$GRAFFITI_HOME/venv/bin/python" - <<'PY'
from pathlib import Path
import site
project_packages = Path.cwd() / '.venv/lib/python3.13/site-packages'
assert project_packages.is_dir()
(Path(site.getsitepackages()[0]) / 'typology-base.pth').write_text(str(project_packages) + '\n')
PY
UV_CACHE_DIR="$GRAFFITI_HOME/uv-cache" uv pip install --python "$GRAFFITI_HOME/venv/bin/python" --no-deps -r explore/graffiti-requirements.txt
```

Adjust the Python-version directory if needed. The deployed versions are Diffusers 0.41.0 and PEFT 0.21.2. The existing project supplies Torch, Transformers, Accelerate, Pillow, NumPy and Hugging Face Hub. Do not remove that base environment while this service uses it.

Download only the fp16 Diffusers weights and supporting files:

```bash
HF_HOME="$GRAFFITI_HOME/hf" "$GRAFFITI_HOME/venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'stable-diffusion-v1-5/stable-diffusion-v1-5',
    revision='451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
    allow_patterns=['model_index.json', '*/config.json', 'scheduler/*',
                    'tokenizer/*', '*/model.fp16.safetensors',
                    '*/diffusion_pytorch_model.fp16.safetensors',
                    'feature_extractor/*'], max_workers=3)
PY
```

The [base model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) documents its OpenRAIL license and limitations. The training follows the [Diffusers LoRA approach](https://huggingface.co/docs/diffusers/training/lora).

## Dataset and training

```bash
.venv/bin/python explore/prepare_graffiti_training.py --output "$GRAFFITI_HOME/dataset"
```

Preparation selects published graffiti at confidence 0.90 or above, rejects small, very soft, flat and repeated crops, keeps one per known panorama/group, and separates validation camera locations from training by at least 100m. It writes contact sheets and source metadata. Review every sheet, remove unsuitable records and corresponding images, record exclusions, update the manifest hashes and set `summary.json`'s `reviewed` field to true. The trainer refuses an unreviewed dataset. The first review rejected 34 of 288 candidate crops, including shop graphics, blur and partial views.

The preparer refuses to overwrite an existing dataset. Preserve the reviewed dataset and its manifest hashes alongside the adapter. Camera spacing and perceptual hashes reduce duplicate works, but cannot prove artwork identity or artist-independent evaluation.

```bash
"$GRAFFITI_HOME/venv/bin/python" explore/graffiti_model.py train --home "$GRAFFITI_HOME" --steps 300
"$GRAFFITI_HOME/venv/bin/python" explore/graffiti_model.py generate --home "$GRAFFITI_HOME" --base --output "$GRAFFITI_HOME/base-sample.png"
"$GRAFFITI_HOME/venv/bin/python" explore/graffiti_model.py generate --home "$GRAFFITI_HOME" --output "$GRAFFITI_HOME/lora-sample.png"
```

Training caches image latents and text embeddings in CPU memory, then trains only the UNet adapters with gradient checkpointing. Progress and held-out loss go to `training.json`; final adapters go to `lora/`. Do not train while the public worker is generating, and do not start a second trainer against the same output directory. Retraining currently starts fresh rather than resuming a checkpoint.

## Website worker

Install `explore/services/typology-graffiti.service` as a user service after adjusting its environment path if necessary. The API binds to `127.0.0.1:8797`; Caddy forwards `/api/graffiti/*` to it. Optional `~/.config/typology-machine/graffiti.env` can set `TYPOLOGY_GRAFFITI_HOME`.

- `GET /api/graffiti/status` exposes model progress without private paths.
- `POST /api/graffiti/generate` accepts a same-origin JSON request with `prompt`, integer `seed`, and `strength` from 0 to 1.2.
- `GET /api/graffiti/jobs/<id>` polls the request.
- `GET /api/graffiti/images/<id>.png` serves completed images.

There is one active request globally, no waiting queue, and a one-minute cooldown. Each request has fixed 512px resolution, 24 inference steps and a four-minute timeout. The worker starts a short-lived inference subprocess with CPU offload so model weights do not reserve GPU memory between images. Only the newest 200 PNG outputs remain; shared settings do not promise permanent image hosting. The frontend never generates merely from opening a shared link.

Compare archive influence 0 and 0.8 with the same prompt and seed. This prototype does not reliably spell words, prove novelty, or guarantee a recognisable local style. It needs broader visual evaluation before expanding training. The generation pipeline retains the base model's safety checker.

Validation: `python -m unittest discover -s explore -p 'test_graffiti_api.py'` covers request limits, training gates, single-job/cooldown enforcement and status redaction. Actual 512px base and LoRA image generation and browser form/polling/mobile checks were also exercised.
