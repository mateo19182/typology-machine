# Typology Machine

Find repeated objects in street-level photos and browse them as a map or a wall
of image crops. The current study covers A Coruña: shop signs, street names,
graffiti, street furniture, people, and pets.

[Open the viewer](https://typology.m19182.dev)

The pipeline detects objects, extracts crops, reads sign text, and proposes
conservative duplicate groups for review. The viewer supports category, barrio,
source, and capture-date filters, with links to individual objects and downloads
of the filtered records. Detections and OCR are experimental, not verified labels.

## Experiments

[Open the experiments](https://typology.m19182.dev/experiments.html):

- **Visual wandering** follows saved visual matches between real street photos, holding the detected object at a fixed screen position during each crossfade. The initial graph samples 2,500 objects; no model runs while browsing.
- **Coruña alphabet** composes text from an editable starter of recognized letters on real signs. [Alphabet setup and limitations](docs/alphabet.md).
- **Living typology book** creates printable A4 contact sheets with object links and PDF output through browser print.
- **Graffiti generator** runs a local SD 1.5 LoRA trained on 225 selected crops, with 29 held out. This is an early experiment, not an established quality improvement. [Training and serving](docs/graffiti.md).

Build the wandering graph from an existing query index and locally cached DINOv2 model:

```sh
uv run python explore/build_wander.py --per-category 500
```

The graph is written to ignored `data/coruna/site/wander-data.json`; crop descriptors are cached under `~/.local/state/typology-machine/wander-features`. Rebuilding does not rerun detection. Full street photos use the existing context service and decode on first access. The map and wall do not load experiment assets.

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

```sh
git clone https://github.com/mateo19182/typology-machine.git
cd typology-machine
uv sync
cp .env.example .env
```

Add a Mapillary token to `.env` only if using the Mapillary scripts. Detection
and model experiments may need a compatible GPU and access to the selected model.
Credentials, downloaded imagery, model weights, and generated datasets are not
included. A fresh clone does not contain the data needed to run the viewer.

With an existing exported dataset under `data/coruna/site`, build the query index
and start the API:

```sh
uv run explore/object_api.py build
uv run explore/object_api.py serve
```

In another terminal, from the repository root, run Caddy:

```sh
caddy run --config site/Caddyfile --adapter caddyfile
```

Open `http://127.0.0.1:8790`. See [deployment](docs/deployment.md) for exports,
full-image support, background services, custom paths, and optional R2 delivery.

## How it works

- `explore/` contains acquisition, detection, OCR, review, and export scripts.
- `data/` holds local inputs and generated outputs. It is ignored by Git and can
  be a symlink to a larger disk.
- `explore/object_api.py` serves map counts, paginated records, and downloads from
  SQLite. Its index lives outside the image archive.
- `site/` is a plain HTML/CSS/JavaScript viewer. Caddy serves its exported copy and
  proxies the API. Crops can be served locally or from R2.

[Research notes](docs/research-notes.md) describe individual experiments and their
run IDs. [NOTES.md](NOTES.md), [LOG.md](LOG.md), [IDEAS.md](IDEAS.md), and
[PRIOR_ART.md](PRIOR_ART.md) retain the project notebook and references.

## Checks

These checks use temporary fixtures and do not need the city dataset or a GPU:

```sh
uv run python -m unittest discover -s explore -p test_object_api.py
uv run python -m unittest discover -s explore -p test_export_layers.py
uv run python -m unittest discover -s explore -p test_publish_crops_r2.py
node --test site/test-crop-storage.cjs
```

## License and data

Project code is licensed under [MIT](LICENSE). External imagery, map data, and
model weights are separate. See [data sources and scope](docs/data.md), including
the municipal barrio boundaries bundled with the viewer.
