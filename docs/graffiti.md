# Graffiti model experiment

`graffiti.html` generates 512px images with Stable Diffusion 1.5 and a small LoRA trained on selected archive crops. Model weights, selected images and outputs stay outside Git, under `/mnt/data/projects/typology-machine/graffiti-ai` by default. Set `TYPOLOGY_GRAFFITI_HOME` or pass `--home` to change it.

The first run used 225 training images and 29 held-out images, 300 optimization steps, rank 4, 512px inputs, and seed 19182. Only UNet attention adapters train; the base model stays frozen. The held-out denoising loss changed from 0.154809 to 0.154611. That tiny change does not establish improved visual quality. Fixed-seed base and adapted samples both generated successfully and differed visibly.

The reviewed dataset and completed adapter also have private Hugging Face
repositories. [Dataset and model publishing](huggingface.md) covers exports,
provenance, loading and uploads after future runs.

## Next experiment with FLUX.2

The next training target is
[`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B).
It has Apache 2.0 weights and an undistilled base checkpoint intended for fine-tuning.
[BFL's training guide](https://docs.bfl.ai/flux_2/flux2_klein_training) describes the
base variants. The 9B variant and 32B FLUX.2 dev have different license and memory
requirements. The 4B model is the starting point for this project's 24 GB RTX 3090.
An SD 1.5 LoRA cannot be loaded into FLUX; this requires a fresh training run.

`prepare_flux_training.py` creates a separate experiment with the same reviewed
image bytes and train/validation assignments. It pins the base model and official
Diffusers trainer revisions, starts at 512px with aspect-ratio buckets, uses rank
16 adapters, NF4 quantization, cached latents, gradient checkpointing and CPU offload.
The initial plan is 1,200 steps, not an established optimum. Compare fixed-seed
base and adapted previews and inspect checkpoints before choosing a final model.
The generic captions are a limitation; future individually reviewed descriptions
should have their own versioned dataset.

Preparation does not download weights or start GPU work:

```bash
export FLUX_HOME=/mnt/data/projects/typology-machine/graffiti-flux2
/mnt/data/projects/typology-machine/hub-venv/bin/python \
  explore/prepare_flux_training.py prepare --home "$FLUX_HOME"
```

Set up the pinned trainer in its own environment. Reuse the project's installed
Torch packages and the optional Hugging Face environment's dataset packages,
adjusting the Python version. Set up `hub-venv` as described in
[dataset and model publishing](huggingface.md) first:

```bash
git clone https://github.com/huggingface/diffusers "$FLUX_HOME/diffusers"
git -C "$FLUX_HOME/diffusers" checkout 916a5791ed5f29d966200d2eee53f06862ec4434
uv venv --python .venv/bin/python "$FLUX_HOME/venv"
"$FLUX_HOME/venv/bin/python" - <<'PY'
from pathlib import Path
import site
paths = [Path.cwd() / '.venv/lib/python3.13/site-packages',
         Path('/mnt/data/projects/typology-machine/hub-venv/lib/python3.13/site-packages')]
assert all(path.is_dir() for path in paths)
(Path(site.getsitepackages()[0]) / 'typology-base.pth').write_text(
    '\n'.join(str(path) for path in paths) + '\n')
PY
uv pip install --python "$FLUX_HOME/venv/bin/python" --no-deps \
  -e "$FLUX_HOME/diffusers" -r explore/flux-requirements.txt
```

The initial local setup passed the trainer import, argument-parser and TensorBoard
checks without downloading model weights. Its
[official requirements](https://github.com/huggingface/diffusers/blob/916a5791ed5f29d966200d2eee53f06862ec4434/examples/dreambooth/requirements_flux.txt)
are the reference. Keep this environment separate from the public SD worker.
NF4 supports this GPU; the trainer's FP8 mode requires a newer GPU architecture.

After the current GPU work finishes, run:

```bash
"$FLUX_HOME/venv/bin/python" explore/prepare_flux_training.py train \
  --home "$FLUX_HOME" --publish --namespace mateo-19182
```

The wrapper refuses a busy GPU, changed trainer source, changed dataset manifests
or an already started experiment. It records completion only after the trainer
succeeds and writes the final adapter. `--publish` then uploads the reviewed dataset
and FLUX adapter to private Hugging Face repositories. The official trainer consumes
only the `train` split. Its prompt previews do not compute a held-out loss on the
29 validation images, so the model card makes no held-out quality claim.

[BFL's LoRA walkthrough](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
also recommends testing a base-trained adapter on the distilled
`black-forest-labs/FLUX.2-klein-4B` checkpoint for four-step inference. That needs its
own visual comparison before replacing the website's SD worker. The current worker
continues to use the completed SD 1.5 experiment until a FLUX run has been evaluated.

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

There is one active request globally and no waiting queue or time-based cooldown. A new request can start immediately after the previous one finishes. Each request has fixed 512px resolution, 24 inference steps and a four-minute timeout. The worker starts a short-lived inference subprocess with CPU offload so model weights do not reserve GPU memory between images. Only the newest 200 PNG outputs remain; shared settings do not promise permanent image hosting. The frontend never generates merely from opening a shared link.

Compare archive influence 0 and 0.8 with the same prompt and seed. This prototype does not reliably spell words, prove novelty, or guarantee a recognisable local style. It needs broader visual evaluation before expanding training. The generation pipeline retains the base model's safety checker.

Validation: `python -m unittest discover -s explore -p 'test_graffiti_api.py'` covers request limits, training gates, single-job enforcement, immediate subsequent requests and status redaction. Actual 512px base and LoRA image generation and browser form/polling/mobile checks were also exercised.
