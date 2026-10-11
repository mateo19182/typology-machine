"""Crop every Mapillary map feature in a city from the best image that sees it.

A map feature is a triangulated point with a list of images. For each feature we score its images by
distance (close but not too close), fetch their segmentation detections, and pick the detection of the
right class whose horizontal position best matches the bearing from camera to feature. The crop comes
from thumb_2048 and gets padded a bit.

Everything is cached under data/<city>/cache so reruns are cheap.
Usage: uv run explore/crop_features.py [city] [--limit N] [--value object--sign--store]
"""
import argparse
import base64
import csv
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import mapbox_vector_tile as mvt
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv(".env")
TOKEN = os.environ["MAPILLARY_TOKEN"]
GRAPH = "https://graph.mapillary.com"
IMG_FIELDS = "id,thumb_2048_url,computed_compass_angle,compass_angle,camera_parameters,camera_type,is_pano,width,height,captured_at,computed_geometry,geometry"
CANDIDATES = 3  # images per feature whose detections we look at
PAD = 0.15
MIN_SIDE = 24  # px in the 2048 thumb; smaller crops are useless

session = requests.Session()
lock = threading.Lock()


def get(url, **params):
    for attempt in range(4):
        try:
            r = session.get(url, params={"access_token": TOKEN, **params}, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
    raise RuntimeError(f"failed {url} {params}")


def cached_json(path, fetch):
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def meters(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
    dy = (lat2 - lat1) * 111320
    return dx, dy


def predicted_x(img, flon, flat):
    """Horizontal position (0..1) where the feature should appear, or None if unknown/behind."""
    lon, lat = (img.get("computed_geometry") or img["geometry"])["coordinates"]
    dx, dy = meters(lon, lat, flon, flat)
    bearing = math.degrees(math.atan2(dx, dy)) % 360
    heading = img.get("computed_compass_angle", img.get("compass_angle"))
    if heading is None:
        return None
    rel = (bearing - heading + 540) % 360 - 180
    if img.get("is_pano") or img.get("camera_type") in ("spherical", "equirectangular"):
        return 0.5 + rel / 360
    if abs(rel) > 80:
        return None
    w, h = img.get("width") or 4, img.get("height") or 3
    f = (img.get("camera_parameters") or [0.8])[0] * max(w, h) / w  # focal in units of image width
    return 0.5 + f * math.tan(math.radians(rel))


def decode_polygon(b64):
    tile = mvt.decode(base64.b64decode(b64), default_options={"y_coord_down": True})
    layer = next(iter(tile.values()))
    ext = layer["extent"]
    g = layer["features"][0]["geometry"]
    rings = g["coordinates"] if g["type"] == "Polygon" else [r for p in g["coordinates"] for r in p]
    pts = [(x / ext, y / ext) for ring in rings for x, y in ring]
    return pts


def process(feature, cache, crops_dir):
    fid, value = feature["id"], feature["value"]
    out = crops_dir / value / f"{fid}.jpg"
    flon, flat = float(feature["lon"]), float(feature["lat"])
    fmeta = cached_json(cache / "features" / f"{fid}.json", lambda: get(f"{GRAPH}/{fid}", fields="id,images,aligned_direction").json())
    imgs = fmeta.get("images", {}).get("data", [])
    if not imgs:
        return None

    def dist_score(i):
        dx, dy = meters(*i["geometry"]["coordinates"], flon, flat)
        d = math.hypot(dx, dy)
        return abs(d - 9)  # ~9 m: big in frame but not cut off

    best = None
    for cand in sorted(imgs, key=dist_score)[:CANDIDATES]:
        iid = cand["id"]
        img = cached_json(cache / "images" / f"{iid}.json", lambda: get(f"{GRAPH}/{iid}", fields=IMG_FIELDS).json())
        dets = cached_json(cache / "detections" / f"{iid}.json", lambda: get(f"{GRAPH}/{iid}/detections", fields="id,value,geometry").json().get("data", []))
        px = predicted_x(img, flon, flat)
        for d in dets:
            if d["value"] != value:
                continue
            pts = decode_polygon(d["geometry"])
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
            area = (box[2] - box[0]) * (box[3] - box[1])
            cx = (box[0] + box[2]) / 2
            err = abs(cx - px) if px is not None else 0.25
            if err > 0.2:
                continue
            score = math.sqrt(area) * (1 - err / 0.2)
            if best is None or score > best["score"]:
                best = {"score": score, "image": img, "box": box, "poly": pts, "err": err, "det": d["id"]}
    if best is None:
        return None

    img = best["image"]
    jpg = cache / "jpg" / f"{img['id']}.jpg"
    if not jpg.exists():
        jpg.parent.mkdir(parents=True, exist_ok=True)
        jpg.write_bytes(session.get(img["thumb_2048_url"], timeout=60).content)
    im = Image.open(jpg)
    W, H = im.size
    x0, y0, x1, y1 = best["box"]
    pw, ph = (x1 - x0) * PAD, (y1 - y0) * PAD
    box = (max(0, (x0 - pw) * W), max(0, (y0 - ph) * H), min(W, (x1 + pw) * W), min(H, (y1 + ph) * H))
    if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    im.crop(tuple(int(v) for v in box)).convert("RGB").save(out, quality=90)
    return {
        "feature": fid, "value": value, "lon": flon, "lat": flat, "image": img["id"], "detection": best["det"],
        "captured_at": img.get("captured_at"), "is_pano": img.get("is_pano"), "box": best["box"],
        "crop_px": [int(box[2] - box[0]), int(box[3] - box[1])], "bearing_err": round(best["err"], 3),
        "poly": best["poly"], "file": str(out.relative_to(crops_dir.parent)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--value")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    base = Path("data") / a.city
    feats = list(csv.DictReader(open(base / "features.csv")))
    if a.value:
        feats = [f for f in feats if f["value"] == a.value]
    if a.limit:
        feats = feats[: a.limit]
    cache, crops_dir = base / "cache", base / "crops"

    done = ok = 0
    results = []
    with ThreadPoolExecutor(a.workers) as ex, open(base / "crops.jsonl", "a") as log:
        for res in ex.map(lambda f: _safe(process, f, cache, crops_dir), feats):
            done += 1
            if res:
                ok += 1
                with lock:
                    log.write(json.dumps(res) + "\n")
            if done % 200 == 0:
                print(f"{done}/{len(feats)} processed, {ok} cropped", flush=True)
    print(f"done: {done} processed, {ok} cropped")


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # one bad feature shouldn't kill a 20k run
        print("error", args[0]["id"], e, flush=True)
        return None


if __name__ == "__main__":
    main()
