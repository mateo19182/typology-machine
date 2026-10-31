# Graffiti model experiment

**Retired 2026-10-31.** The page is no longer linked or deployed (`/graffiti.html` redirects to the map), Caddy no longer proxies `/api/graffiti/`, and `typology-graffiti.service` is disabled. Code, run directories, datasets and Hugging Face exports are kept; the notes below describe the experiment as it ran.

`graffiti.html` generates 512px images with distilled FLUX.2 klein 4B and the captioned Coruña LoRA, using four steps and a persistent worker. The adapter was trained on Klein Base 4B. The deployed run is `/mnt/data/projects/typology-machine/graffiti-flux2-captioned-20261014`, with 971 training images, 92 holdout images and 2,000 optimization steps. Prompts are passed through without a city-name prefix. Model weights, selected images and outputs stay outside Git. Set `TYPOLOGY_GRAFFITI_HOME` or pass `--home` to choose a run; `training.json` records the training model and optional `inference.json` selects the deployed distilled profile. The original SD CLI defaults to `/mnt/data/projects/typology-machine/graffiti-ai`.

The first run used 225 training images and 29 held-out images, 300 optimization steps, rank 4, 512px inputs, and seed 19182. Only UNet attention adapters train; the base model stays frozen. The held-out denoising loss changed from 0.154809 to 0.154611. That tiny change does not establish improved visual quality. Fixed-seed base and adapted samples both generated successfully and differed visibly.

The reviewed dataset and completed adapter also have private Hugging Face
repositories. [Dataset and model publishing](huggingface.md) covers exports,
provenance, loading and uploads after future runs.

## FLUX.2 experiment

The adapter's training model is
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
The initial small-dataset plan is 1,200 steps, not an established optimum. Compare fixed-seed
base and adapted previews and inspect checkpoints before choosing a final model.
The generic captions are a limitation; future individually reviewed descriptions
should have their own versioned dataset.

### Extracting captions from the database

`caption_images.py` uses the cached Qwen3.5-4B vision model at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with NF4 weights, greedy decoding and
thinking disabled. It sees the crop and a fixed instruction, without city names,
dates or other database metadata. It returns a short visual description, a
separate unverified text candidate, text legibility and uncertain details.
Descriptions target two sentences covering the subject, palette, shapes and
visible surface. City names, quoted lettering and other transcription attempts
are flagged for review. OCR and VLM text candidates are not training captions.

Results are cached in `~/.local/state/typology-machine/captions.sqlite3`, keyed
by image ID, image-byte checksum and caption recipe. Changed crops or recipes
are recaptioned; the query database and OCR fields remain separate. The recipe
records the model revision, instruction, resizing, quantization and decoding.
This produces AI labels, not verified transcriptions or calibrated confidence.
The `captions` table retains raw candidates. `caption_reviews` stores approved
descriptions and review notes separately, including corrections and whether a
human verified them. `reviewed_caption()` retrieves a reviewed description for
an exact image and recipe without silently substituting a different crop.

The worker accepts reviewed ImageFolder datasets. To select crops directly from
the query database, `collect` reads one frozen snapshot and chooses object
representatives in balanced category order. It uses existing local crop bytes
through symlinks. This selection is not a reviewed training split.

```bash
CAPTION_PY=/mnt/data/projects/typology-machine/graffiti-flux2/venv/bin/python
"$CAPTION_PY" explore/caption_images.py collect \
  --categories graffiti store_sign door window --limit 1000 --output /path/to/crop-selection
"$CAPTION_PY" explore/caption_images.py caption \
  --dataset /path/to/crop-selection --output /path/to/caption-run --batch 4
"$CAPTION_PY" explore/caption_images.py sheets \
  --dataset /path/to/crop-selection --captions /path/to/caption-run --output /path/to/review-sheets
```

Use batch 4 when sharing the GPU and pause the graffiti worker so it cannot start
another model. The larger batch 16 pass runs with the GPU services paused and an
automatic restoration script. Review caption/image pairs, correct unsupported
details, and record the exact `captions.jsonl` checksum in `review.json`. The
`build` command refuses changed input bytes, changed reviewed captions and
uncorrected location or lettering guesses. It retains every source image byte,
ID and train/validation assignment, replacing only captions and adding provenance.

For individual captions, the FLUX recipe uses `--caption_column text`, a 128-token
embedding cache and 10% caption dropout. The encoder stays on the GPU while
captions are encoded, then is removed before optimization. Generic shared-caption
runs keep their original recipe. FLUX web requests use the supplied prompt
directly, without automatically inserting a city name.

The 2026-10-14 captioned dataset retains all 971/92 images from the expanded run.
Qwen produced 1,061 descriptions; two repeated structured-output failures were
captioned directly by Codex from the images. Codex visually checked all 40 flagged
descriptions and a deterministic sample of 72 training and 36 validation images,
correcting 66 captions including the two fallbacks. Automated checks cover every
caption. This is AI review; unflagged descriptions outside the sample can still
contain visual errors. The largest encoded caption has 67 tokens including the
chat template, so none is truncated by the 128-token recipe. The source captions,
review decisions and corrected dataset are separate versions outside Git.

The fresh captioned run completed 2,000 steps at rank 16. All 2,000 loss values and
all 120 saved adapter tensors are finite. Fourteen NF4 base/adapter comparisons
use the same pipeline as the website. Across three seeds, the raw prompt `wall`
produced ordinary walls with neither adapter nor base printing Coruña. The
adapted colour/surface prompts and a bird mural generated successfully; the
explicit `HELLO` example rendered readable lettering. These examples do not
establish general visual-quality gains or reliable spelling.

The captioned dataset is private at
[`coruna-graffiti-captioned-v3`](https://huggingface.co/datasets/mateo-19182/coruna-graffiti-captioned-v3),
commit `d659780255f07e75490f4866658731117401f9da`. Its new model is private at
[`coruna-graffiti-flux2-klein-captioned-lora`](https://huggingface.co/mateo-19182/coruna-graffiti-flux2-klein-captioned-lora),
commit `047f89ead7f10e8fd2a1e15f98e7a087b67c52be`. Model provenance pins that
dataset commit. Downloaded adapter bytes match local SHA256
`3d2c71923587a9f374d9d1ad6cece59b6f3281a40e931a5fadde23fdbf7809a0`.

The expanded experiment uses 971 training crops and 92 holdout crops, drawn from
the same frozen query snapshot at confidence 0.85 or above. It retains the original
225/29 reviewed split and exact image bytes. Codex screened all 918 new candidates
on contact sheets, inspected 17 uncertain crops at full resolution, and rejected
109 additions. This is AI visual review, not a human artwork-identity audit.
The geographic buffer and cross-split ID/group/panorama/hash checks pass. The run
home is `/mnt/data/projects/typology-machine/graffiti-flux2-expanded-20261013`. It
completed 2,000 optimization steps at rank 16, with checkpoints every 250 steps.
All 2,000 recorded training losses were finite. Optimization and intermediate
previews took about 45 minutes on the RTX 3090. Its common caption is encoded
once rather than copied into a separate GPU text-embedding cache for every image.
Three fixed-seed base/adapter comparisons are in `evaluation-comparison.jpg` and
the six PNG files under `evaluation/`. These examples show the adapter's effects;
they do not establish held-out visual quality. The enlarged dataset is private at
[`mateo-19182/coruna-graffiti-reviewed-v2`](https://huggingface.co/datasets/mateo-19182/coruna-graffiti-reviewed-v2).
The completed adapter and comparisons are private at
[`mateo-19182/coruna-graffiti-flux2-klein-lora`](https://huggingface.co/mateo-19182/coruna-graffiti-flux2-klein-lora).
Its provenance pins dataset commit `a5f6a74cbd73e4deeb767a11ae0880a3f01db459`;
the model commit is `f4b4d128f9680f33e6c51a77e3f9929da13e369d`.

Expand a reviewed dataset without moving its existing holdout images into training:

```bash
/mnt/data/projects/typology-machine/hub-venv/bin/python \
  explore/prepare_graffiti_training.py \
  --expand-from /mnt/data/projects/typology-machine/graffiti-ai/dataset \
  --output /mnt/data/projects/typology-machine/graffiti-datasets/new-candidates \
  --train 1200 --validation 96 --confidence 0.85
```

Review the additions, record exclusions, update the manifests and summary hashes,
and validate the reviewed output before preparing a new FLUX run. Prior rejected
source IDs remain excluded. An expansion can produce fewer than the requested
counts when quality, deduplication and geographic filters exhaust the candidates.

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
holdout images, so the model card makes no held-out quality claim.

`evaluate_graffiti_flux.py --home RUN_HOME` generates three base/adapter pairs with
the same prompts, seeds and sampler settings. It uses a fresh process after training
and writes `evaluation-comparison.jpg` and `evaluation.json`. The expanded run uses
a user systemd job so it survives a terminal disconnect. Its cleanup restores the
temporarily paused inference container and graffiti worker on success or failure.

[BFL's LoRA walkthrough](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
recommends testing a base-trained adapter on the distilled
`black-forest-labs/FLUX.2-klein-4B` checkpoint for four-step inference.

### Distilled inference benchmark

`benchmark_graffiti_flux.py` compares the 30-step, guidance-4 base pipeline with
the distilled four-step, guidance-1 pipeline using the same completed adapter,
prompts, seeds, 512px output, 128-token limit, NF4 components and CPU offload.
The distilled revision is pinned to `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
Both snapshots must already be cached. Text encoder and VAE weights match the
base model's bytes; only the distilled transformer weights and configuration differ.
No training or adapter upload is needed.

```bash
/mnt/data/projects/typology-machine/graffiti-flux2/venv/bin/python \
  explore/benchmark_graffiti_flux.py run \
  --home /mnt/data/projects/typology-machine/graffiti-flux2-captioned-20261014 \
  --output /path/to/new-benchmark
```

The command refuses an occupied or already paused generator. It temporarily adds
its own service override to pause public requests and removes that override in a
`finally` block, restoring the API even if a worker fails. Other service overrides
are preserved. Henad stays running. Each fresh timing includes process startup,
imports, model loading, generation, PNG saving and shutdown. Separate reused-pipeline
timings measure generation and saving after the pipeline has already loaded.

On 2026-10-14, three fresh requests per variant had median total times of 29.811s
for the base model and 23.254s for distilled, a 22% reduction. The ranges were
27.409–43.841s and 17.696–28.210s respectively. Four requests per variant with a
reused pipeline had medians of 15.380s and 2.853s, about 5.4 times faster. These
are small measurements with filesystem-cache and shared-GPU variation. The
website used a fresh 30-step process for every request before the deployment
described below.

Six selected image pairs cover a red/white shutter, blue/orange lettering,
a bird mural, the raw prompt `wall`, explicit `HELLO`, and zero LoRA influence.
Codex visually inspected the pairs. Both models follow the colour/surface prompts
and show a visible change between influence zero and 1.2. The distilled `HELLO`
example is readable; its plain-wall example contains no lettering. Neither wall
example prints Coruña. Composition differs between checkpoints, and these
examples do not establish general quality or spelling reliability. Identical
shutter requests produced matching PNG checksums across fresh and reused runs
within each variant.

All sixteen benchmark images completed. Henad remained running and healthy;
total GPU memory peaked at 21,145 MiB. Reports, telemetry, raw PNGs and contact
sheets are under the run home's `inference-benchmark-20261014/`. The generated
`index.html` and its twelve selected PNGs are served at
[`/graffiti-benchmark/`](https://typology.m19182.dev/graffiti-benchmark/) for inspection.
Only that page and its generated images are copied to the public site.

`test_benchmark_graffiti_flux.py` checks restoration after a failed worker and
refusal to interrupt an occupied generator. The existing fifteen API tests cover
the original serving route, progress and library flow.

### Persistent four-step deployment

The run home's `inference.json` contains the explicit serving profile:

```json
{"profile": "distilled", "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294"}
```

This selects four steps and guidance 1 while preserving the original base-trained
adapter and `training.json`. The API snapshots its inference settings on startup.
`graffiti_worker.py` preloads one NF4 pipeline in a child process, then accepts
serialized image requests over a JSON pipe. The API stays responsive during
loading and generation. Model logs go to `generation.log`, separate from the
pipe's status messages. Each job has a 600-second deadline. A failed, timed-out
or unexpectedly exited worker is reaped and recreated for the next request.
The user service kills the API and its worker together on stop or restart.
The model remains loaded between requests, with CPU offload for GPU sharing.

Progress callbacks report four actual sampling steps. The page retains its
progress bar without timing labels, strength-1.2 default and minimal library.
New PNG metadata records the inference model ID, pinned revision, steps, guidance
and text-token limit. The public library includes the validated model ID,
revision and steps alongside the existing prompt/seed/influence fields.
Older images remain available. Timing history includes model revision and worker
mode so the old 30-step durations do not become four-step estimates.

Public browser verification completed three requests with influence 1.2, zero
and 1.2 on the same prompt/seed. API durations were 3.940s, 3.778s and 3.030s;
browser completion times were 4.767s, 4.575s and 3.513s. The worker loaded once.
All three PNG checksums match the corresponding distilled benchmark images,
including the restored influence after the zero-adapter request. Concurrent
generation returned 429, live progress remained bounded to four steps, and the
library grew from thirteen to sixteen images. Downloads, reuse, page refresh,
mobile/desktop layouts and removal of the timing/count/disclaimer copy passed.
Henad remained running and healthy. Idle GPU use after these requests was
16,687 MiB, including Henad, versus its earlier 16,314 MiB use alone.

`test_graffiti_worker.py` covers actual process reuse, recovery after an error,
a timeout with an unterminated response, four-step callbacks, adapter-strength
changes and persisted inference metadata. All twenty-one graffiti tests pass.
Verification artifacts and deployment backups are under
`distilled-deployment-20261014/` in the run home.

To return to fresh 30-step inference, move `inference.json` aside and restart
`typology-graffiti.service`. Restart after any serving-profile change so the API's
settings and the resident pipeline agree. Training and benchmark commands that
need the GPU must pause/restart the public service, which also releases its child.

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

Install `explore/services/typology-graffiti.service` as a user service after adjusting its environment path if necessary. The template uses the pinned FLUX runtime in `graffiti-flux2/venv` and the completed captioned run. The API binds to `127.0.0.1:8797`; Caddy forwards `/api/graffiti/*` to it. Optional `~/.config/typology-machine/graffiti.env` can override `TYPOLOGY_GRAFFITI_HOME`. Copy frontend changes into the exported site directory and update the JavaScript version in `graffiti.html`.

- `GET /api/graffiti/status` exposes model progress without private paths.
- `POST /api/graffiti/generate` accepts a same-origin JSON request with `prompt`, integer `seed`, and `strength` from 0 to 1.2.
- `GET /api/graffiti/jobs/<id>` polls the request.
- `GET /api/graffiti/images/<id>.png` serves completed images.
- `GET /api/graffiti/library` lists retained generations, newest first.
- `GET /api/graffiti/thumbnails/<id>.webp` serves cached 256px thumbnails.

The library shows a grid of thumbnails below the generator, eighteen at a time.
The latest saved image fills the initial preview without starting generation.
An image opens in a native dialog with its PNG download and, when available,
saved prompt/seed/influence and a button to reuse those settings. Opening the
library or reusing settings never starts GPU generation. New completed requests
write `generated/<id>.json` metadata next to their PNG, so the library survives
page refreshes and API restarts. Older PNGs without metadata remain viewable;
their settings are not guessed. Active requests are excluded. The existing
200-image retention also removes corresponding metadata and cached thumbnails.
New images appear immediately after generation and refresh for other visitors.

The page links the captioned dataset and model on Hugging Face using the compact
label `FLUX.2 klein 4B + Coruña graffitiLoRA.` Both repositories remain
private pending clearance of the source imagery and artwork rights. The model
package contains weights and generated evaluation images without source crops;
that separation alone does not establish permission to publish the trained model.

The generator defaults to archive influence 1.2 and a red/white graffiti-on-shutter
prompt. Shared links retain their specified prompt, seed and influence. No trigger
word is inserted. Compare influence zero at the same prompt and seed to see the
adapter's effect.

Generation progress comes from the inference pipeline's step-end callback.
Loading and prompt preparation show an indeterminate bar; drawing reports actual
completed steps out of 30; finishing remains visible until the worker succeeds.
Each job writes an atomic `generated/<id>.progress.json` sidecar. The API validates
phase names and step bounds and exposes elapsed time separately from progress.
Failed jobs never report completion. The page polls progress each second and
shows the bar without stage labels, elapsed time or estimated-duration text.

`generation-timings.json` retains up to 50 successful worker durations, including
model loading and subprocess cleanup. Status exposes their median, range and
sample count. New jobs capture that median as their estimated total; before any
measurements, the estimate is 40 seconds. The history survives API restarts and is
ignored when model/inference settings change. Failed requests do not affect the
estimate. Timing records contain durations without prompts or private paths.

There is one active request globally and no waiting queue or time-based cooldown. A new request can start immediately after the previous one finishes. Each request has fixed 512px resolution and a ten-minute timeout including any model loading. The deployed distilled FLUX profile uses four steps and guidance 1.0 with a reused worker; the base FLUX fallback uses 30 steps and guidance 4.0, and the retained SD path uses 24 steps and guidance 7.0. Status reports the selected model, inference settings, worker state and training counts.

For the captioned FLUX run, inference uses the same 128-token text length as
training. `TYPOLOGY_GRAFFITI_PAUSED=1` keeps status available and refuses new GPU
jobs during retraining and evaluation. A separate `progress.json` can report
bounded training progress without changing the training record. Remove the
temporary service override after validating the completed adapter.

For FLUX serving, both the transformer and Qwen3 text encoder load in NF4 with double quantization and bfloat16 computation. CPU offload moves the components onto the GPU as needed. This differs from the full bfloat16 comparison samples and can affect output. The worker starts a short-lived inference subprocess so weights do not reserve GPU memory between images. It shares the 24 GB RTX 3090 with the running `henad_inference` container. Only the newest 200 PNG outputs remain; shared settings do not promise permanent image hosting. The frontend never generates merely from opening a shared link.

Compare archive influence 0 and 0.8 with the same prompt and seed. This prototype does not reliably spell words, prove novelty, or guarantee a recognisable local style. It needs broader visual evaluation before expanding training.

Validation: `python -m unittest discover -s explore -p 'test_graffiti_api.py'` covers request limits, training gates, single-job enforcement, immediate subsequent requests, status redaction, model selection and releasing the slot after worker failure. Actual 512px base and LoRA image generation and browser form/polling/mobile checks were also exercised.

The FLUX deployment was checked through the public form on 2026-10-14. Two
same-prompt, same-seed requests at influence 0.8 and 0 produced different 512px
PNGs in 46 and 52 seconds, including model loading. Mobile and desktop layouts,
polling and the download link passed, with no browser errors. Total GPU usage
peaked at 21,125 MiB while `henad_inference` stayed healthy, then returned to
16,294 MiB after generation. The report, test script and screenshots are in the
run's `web-verification/` directory outside Git.

The captioned release passed a second public browser check on mobile and desktop.
Raw `wall` requests at influence 0.8 and 0 completed in 37.5 and 29.3 seconds,
with two different 512px PNGs, working downloads and no browser errors. Total GPU
usage peaked at 21,145 MiB while the other service stayed healthy. All 25 caption,
API and publisher tests passed. Both Hugging Face repositories are private;
downloaded weights and the model's dataset commit were verified after upload.

The progress display and strong default passed public mobile/desktop checks on
2026-10-14. Three new requests completed in 42.7, 37.6 and 27.0 seconds, including
loading and subprocess cleanup, with a median of 37.6 seconds. Browser display
took 44.6, 38.8 and 28.1 seconds. Every request exposed loading, prompt preparation,
drawing and finishing phases with increasing drawing-step counts. Same-seed
influence 1.2 and zero produced visibly different red/white graffiti on shutters.
Total GPU use peaked at 21,145 MiB, and `henad_inference` remained healthy. Browser
checks also cover exceeded estimates, failures, busy responses, preserving shared
settings, and avoiding automatic generation when opening the page. Twelve API
tests pass, including bounded progress and timing persistence across restarts.
Artifacts are in the captioned run's `progress-verification/` directory.

The library release passed 15 API tests and a public browser check on mobile and
desktop. Twelve existing generations appeared, including older PNGs without
settings. One real request increased the library to thirteen and retained its
prompt, seed and influence. PNG downloads, cached WebP thumbnails, native dialog
close/Escape, reusing settings without generation, persistence after reload and
the eighteen-image paging control passed. The obsolete model/count labels and
timing text are absent. `henad_inference` stayed healthy. Artifacts are in the
captioned run's `library-verification/` directory.
