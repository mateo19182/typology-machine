"""Save cluster metadata and an interactive atlas from an existing clustering run.

CPU only; never recomputes embeddings. Open the generated index.html directly.
Usage: uv run explore/cluster_report.py coruna store_sign --source apple
"""
import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def build_report(base, emb="dinov2", parameters=None):
    base = Path(base)
    clusters = json.loads((base / f"clusters_{emb}.json").read_text())
    ids = json.loads((base / f"embeddings_{emb}_ids.json").read_text())
    vectors = np.load(base / f"embeddings_{emb}.npy", mmap_mode="r")
    members = [i for group in clusters.values() for i in group]
    if len(set(members)) != len(members) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate crop IDs in cluster memberships or embedding index")
    if len(ids) != len(vectors) or not set(members) <= set(ids):
        raise ValueError("Cluster memberships do not match the saved embedding index")
    rows = {r["id"]: r for r in map(json.loads, (base / "instances.jsonl").read_text().splitlines())}
    positions = {}
    if (base / "sign_positions.jsonl").exists():
        positions = {p["rep"]: p for p in map(json.loads, (base / "sign_positions.jsonl").read_text().splitlines())}
    names_path = base / f"cluster_names_{emb}.json"
    names = json.loads(names_path.read_text()) if names_path.exists() else {}
    index = {sid: i for i, sid in enumerate(ids)}
    X = np.array(vectors[[index[sid] for sid in members]])
    if not len(X) or not np.isfinite(X).all():
        raise ValueError("Embeddings must be nonempty and finite")
    projection = PCA(n_components=min(2, *X.shape), random_state=0)
    xy = projection.fit_transform(X) if len(X) > 1 else np.zeros((1, 2))
    if xy.shape[1] < 2:
        xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])))
    # Scope labels and colors to a content-addressed run; numeric labels can change on reclustering.
    run_id = digest({"clusters": clusters, "embedding": emb, "vectors": hashlib.sha256(X.tobytes()).hexdigest()})
    previous_path = base / f"cluster_metadata_{emb}.json"
    if parameters is None and previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous["run_id"] == run_id:
            parameters = previous["clustering"]
    metadata = {"schema_version": 1, "run_id": run_id, "source": base.parent.name,
                "embedding": {"model": "facebook/dinov2-base" if emb == "dinov2" else emb,
                              "dimensions": X.shape[1], "file": f"embeddings_{emb}.npy",
                              "pooling": "L2-normalized CLS + mean patch" if emb == "dinov2" else "unknown"},
                "clustering": parameters or {"algorithm": "unknown", "note": "Imported existing labels; original parameters were not saved."},
                "projection": {"method": "PCA", "dimensions": 2, "note": "Display only; distances lose information."},
                "coverage": {"assigned_crops": len(members), "total_crops": len(rows)}, "clusters": [], "assignments": []}
    points = dict(zip(members, xy))
    features = []
    for label, group in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        hue = (int(label) * 0.61803398875) % 1
        rgb = colorsys.hsv_to_rgb(hue, .58, .88)
        color = "#777777" if label == "-1" else "#" + "".join(f"{round(c * 255):02x}" for c in rgb)
        key = f"{run_id}:{label}"
        # Names are provisional descriptions, never per-crop classification labels.
        name = "Unclustered / noise" if label == "-1" else names.get(label, {}).get("name", f"Cluster {label}")
        metadata["clusters"].append({"id": key, "label": int(label), "name": name, "color": color,
                                     "count": len(group), "description": names.get(label, {}).get("description", "")})
        for sid in group:
            r, p = rows[sid], positions.get(sid)
            location = {"lat": p["lat"] if p else r["lat"], "lon": p["lon"] if p else r["lon"],
                        "kind": ("triangulated_sign" if p["n_rays"] >= 2 else "estimated_sign") if p else "camera",
                        "n_rays": p["n_rays"] if p else None}
            item = {"crop_id": sid, "cluster_id": key, "label": int(label), "color": color,
                    "xy": [round(float(v), 5) for v in points[sid]], "date": r.get("date"),
                    "crop": r["crop"], "location": location}
            metadata["assignments"].append(item)
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [location["lon"], location["lat"]]},
                             "properties": {k: v for k, v in item.items() if k not in ("xy", "location")} | {"location_kind": location["kind"], "run_id": run_id}})
    out = base / "cluster_reports" / emb / run_id
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out / "metadata.json").write_text(payload)
    (out / "assignments.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in metadata["assignments"]))
    (out / "clusters.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False))
    template = (Path(__file__).parent / "cluster_report.html").read_text()
    (out / "index.html").write_text(template.replace("__CLUSTER_DATA__", payload.replace("<", "\\u003c")))
    # The site exporter uses this latest snapshot, joining strictly by crop ID.
    (base / f"cluster_metadata_{emb}.json").write_text(payload)
    print(f"{len(members)} crops, {len(clusters)} groups including noise -> {out / 'index.html'}")
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("slug", nargs="?", default="store_sign")
    ap.add_argument("--source", default="apple")
    ap.add_argument("--emb", default="dinov2")
    args = ap.parse_args()
    build_report(Path("data") / args.city / args.source / args.slug, args.emb)
