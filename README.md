# Typology Machine

Find repeated objects in street-level photos and browse them as a map or a wall
of image crops. The current study covers A Coruña: shop signs, street names,
graffiti, street furniture, people, and pets.

[Open the viewer](https://typology.m19182.dev)

The pipeline detects objects, extracts crops, reads sign text, and proposes
conservative duplicate groups for review. The viewer supports category, barrio,
source, and capture-date filters, with links to individual objects and downloads
of the filtered records. Detections and OCR are experimental, not verified labels.

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
