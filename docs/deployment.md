# Deployment

Run commands from the repository root. This viewer needs an exported dataset and
the query API; opening `site/index.html` alone is not enough.

## Dataset and exports

Acquisition and detection scripts write beneath `data/<city>/`. The current viewer
uses `data/coruna/site`. The repo does not include a downloadable city dataset.
[Research notes](research-notes.md) explain the experiments and their input files.

`explore/export_site.py` exports legacy shop-sign records and frontend assets.
`explore/export_layers.py` publishes category snapshots from completed detection
runs. Inspect their `--help` output and choose your own run directories; the run
IDs in the service examples belong to the Coruña study.

After publishing or changing legacy records or barrio boundaries:

```sh
uv run explore/object_api.py build
```

The SQLite index stays in `~/.local/state/typology-machine/query` by default.
`--site` and `--state` override the input and index directories. Index publication
is atomic; the previous database remains available to open tabs. The browser
fetches counts and 48 records per page instead of whole category files.

`data` may be a symlink to a mounted data disk. Keep the query index on the SSD.
Copy and verify the dataset, pause writers for the final sync, then switch the
symlink before removing the old copy. Deployment does not perform a migration.

## Serve locally

Run these in separate terminals:

```sh
uv run explore/object_api.py serve
uv run explore/proposal_context.py --host 127.0.0.1 --port 8793 --decoders 1
caddy run --config site/Caddyfile --adapter caddyfile
```

Caddy serves port 8790, forwards `/api/*` to port 8794, and forwards
`/proposal-context/*` to port 8793. The context service supplies full street images
when their local source files are available.

The Caddyfile serves `./data/coruna/site` relative to its working directory.
Set `TYPOLOGY_SITE_ROOT` to an absolute path when running Caddy elsewhere.
Set `TYPOLOGY_PUBLIC_URL` on the API process, or pass `--origin`, so exported image
links point to your deployment. Review-generation scripts use the same variable
for context-image links. Defaults point to `http://127.0.0.1:8790`.

The Caddy example listens on port 8790 without TLS. Configure your own HTTPS
frontend when deploying publicly. The separate review tools bind to localhost by
default; keep their write-enabled interfaces private.

Frontend changes must reach the exported directory as well as `site/`. Re-export
or copy the changed frontend files, including favicons. Change the `app.js?v=...`
version in `index.html` whenever the script changes to avoid stale edge caches.

## Optional systemd services

`explore/services/` contains user-service examples. They assume the checkout is
at `%h/projects/typology-machine`, where `%h` means the user's home directory.
If yours is elsewhere, edit `WorkingDirectory` and executable paths in the copied
units. Review dataset paths and run IDs before enabling a worker. Install only
the services needed for your deployment.

For example, after building the query index:

```sh
mkdir -p ~/.config/systemd/user
cp explore/services/typology-object-api.service ~/.config/systemd/user/
# Edit the copied unit if your checkout is in another directory.
systemctl --user daemon-reload
systemctl --user enable --now typology-object-api.service
```

Units optionally read `~/.config/typology-machine/deployment.env`:

```ini
TYPOLOGY_PUBLIC_URL=https://your-viewer.example
TYPOLOGY_SITE_ROOT=/absolute/path/to/data/coruna/site
TYPOLOGY_DATA_DISK=/mnt/data
```

These variables are not automatically loaded from the repository's `.env` by the
API or Caddy. Export them in the shell for manual runs. Keep deployment-specific
configuration outside Git. Existing installed units are independent of these
examples; copying a changed template is an explicit deployment step.

Layer refresh rebuilds the query index after exporting. OCR, follow-up, and R2
have separate timers. The optional background monitor checks this deployment's
services, disk space, migration, uploads, OCR, and local API. It also checks the
public API when `TYPOLOGY_PUBLIC_URL` is set. It expects the configured data disk
to be mounted and the corresponding workers to be installed.

The monitor writes `~/.local/state/typology-machine/background-status.json` and
logs to `journalctl --user -u typology-background-watch.service`. Its timer runs
every five minutes. It reports failures but does not restart workers, delete data,
or send notifications. Install its `.service` and `.timer` together if needed.

## Optional R2 crops

Local crop serving is the default. R2 is only needed to move crop delivery off the
web server. The uploader accepts JPEGs referenced by published records; it does
not upload originals, masks, or review artifacts.

Create a bucket and its public HTTPS URL. Save this configuration in
`~/.config/typology-machine/r2.env`, with file permissions `0600`:

```ini
R2_ACCOUNT_ID=your-account-id
R2_BUCKET=your-crop-bucket
R2_PUBLIC_URL=https://your-crops.example
AWS_ACCESS_KEY_ID=your-bucket-scoped-access-key
AWS_SECRET_ACCESS_KEY=your-bucket-scoped-secret
```

Use bucket-scoped Object Read & Write credentials. The uploader reads this file;
`--env-file`, `--account`, `--bucket`, and `--public-url` override its defaults.

```sh
uv run explore/publish_crops_r2.py --inventory-only
uv run explore/publish_crops_r2.py --limit 32
uv run explore/publish_crops_r2.py --activate
```

The first command counts referenced crops. The second uploads a bounded sample.
The last resumes a full upload and activates delivery only after public checksum
checks pass. Upload checkpoints live outside `data`, under
`~/.local/state/typology-machine/r2`. Use a separate `--state` directory for a
different account or bucket. The hourly timer can publish later additions.

Activation updates the exported `crop-storage.json`. Missing remote crops retry
locally once. To return to local delivery, stop the uploader and timer and set
that file to `{"base_url":null}`. Uploaded objects are retained; the uploader does
not delete bucket contents.
