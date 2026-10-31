#!/usr/bin/env bash
# Publish the viewer to Cloudflare: sync site/ into the export, upload new crops to R2 when the
# published inventory changed, build the static shell, keep /api etc. on the tunnel, deploy.
# Usage: explore/deploy_cloudflare.sh [--build-only]
set -euo pipefail
cd "$(dirname "$0")/.."

SITE=${TYPOLOGY_SITE_ROOT:-data/coruna/site}
OUT=data/coruna/cloudflare-site
STATE=$HOME/.local/state/typology-machine/cloudflare
ZONE=918c5dabc9228cda6e75e0d1578317d1
HOST=typology.m19182.dev
# Paths served by the tunnel origin (Caddy), bypassing the Worker so they cost no Worker requests.
ORIGIN_PATHS=(/api/ /proposal-context/ /img/ /proposal-assets/)

mkdir -p "$STATE"
exec > >(tee -a "$STATE/deploy-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
set -a; . "${CLOUDFLARE_ENV:-$HOME/.secrets/cloudflare-pages.env}"; set +a
wrangler() { npx -y wrangler@4 "$@"; }

echo "== frontend: site/ -> $SITE"
for file in site/*.html site/*.js site/*.css site/*.png site/*.svg; do
  case $(basename "$file") in graffiti.*) continue ;; esac
  cmp -s "$file" "$SITE/$(basename "$file")" || cp -v "$file" "$SITE/"
done

echo "== crops: R2 upload when the published inventory changed"
signature=$(cat "$SITE/layers/manifest.json" "$SITE/data.json" | sha256sum | cut -d' ' -f1)
if [ "$signature" != "$(jq -r .signature ~/.local/state/typology-machine/r2/inventory.json 2>/dev/null)" ]; then
  uv run explore/publish_crops_r2.py --site "$SITE"
else
  echo "unchanged ($signature)"
fi

echo "== static shell: $OUT"
mkdir -p "$OUT"
rsync -a --delete --delete-excluded --exclude='/graffiti*' --exclude='*.before-*' \
  --include='/*.html' --include='/*.js' --include='/*.css' --include='/*.png' --include='/*.svg' \
  --include='/barrios.geojson' --include='/alphabet-data.json' --include='/wander-data.json' --include='/crop-storage.json' \
  --exclude='*' "$SITE/" "$OUT/"
# Legacy context photos are frozen; hard links avoid a second 2.6 GB copy.
rsync -a --delete --link-dest="$(realpath "$SITE/ctx")" "$SITE/ctx/" "$OUT/ctx/"
cp explore/cloudflare/_headers explore/cloudflare/_redirects "$OUT/"
files=$(find "$OUT" -type f | wc -l)
large=$(find "$OUT" -type f -size +25M | wc -l)
echo "$files files, $(du -sh --apparent-size "$OUT" | cut -f1)"
if [ "$files" -gt 20000 ] || [ "$large" -gt 0 ]; then
  echo "Workers static assets allow 20,000 files and 25 MiB per file" >&2; exit 1
fi
if [ "${1:-}" = --build-only ]; then wrangler deploy --dry-run; exit; fi

echo "== routes without a Worker"
api=https://api.cloudflare.com/client/v4/zones/$ZONE/workers/routes
auth="Authorization: Bearer $CLOUDFLARE_API_TOKEN"
existing=$(curl -fsS -H "$auth" "$api" | jq -r '.result[].pattern')
for path in "${ORIGIN_PATHS[@]}"; do
  pattern="$HOST$path*"
  if ! grep -qxF "$pattern" <<<"$existing"; then
    curl -fsS -H "$auth" -H 'Content-Type: application/json' -d "{\"pattern\":\"$pattern\"}" "$api" | jq -c '{success, pattern: .result.pattern}'
  fi
done

echo "== R2 CORS (alphabet canvas exports)"
wrangler r2 bucket cors set typology-crops --file explore/cloudflare/r2-cors.json --force

echo "== deploy"
wrangler deploy

echo "== smoke test"
curl -fsS -o /dev/null -w "/ %{http_code}\n" "https://$HOST/"
curl -fsS -o /dev/null -w "/api/meta %{http_code}\n" "https://$HOST/api/meta"
