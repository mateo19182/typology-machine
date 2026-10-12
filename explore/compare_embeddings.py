"""Compare DINOv2/v3 on a frozen, source-balanced sample; checkpoint every batch.

uv run explore/compare_embeddings.py coruna --per-source 512 --batch 8
Rerun the same command after granting DINOv3 access to finish a partial comparison.
Outputs are separate from production embeddings, groups, and clusters.
"""
import argparse
import colorsys
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import HfApi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from transformers import AutoImageProcessor, AutoModel

MODELS = {"dinov2": "facebook/dinov2-base", "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m"}


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def make_sample(city, per_source, seed):
    samples = []
    for source in ("apple", "gsv"):
        base = Path("data") / city / source / "store_sign"
        reps = None
        if (base / "groups.jsonl").exists():
            reps = {json.loads(l)["rep"] for l in open(base / "groups.jsonl")}
        # One crop per panorama reduces repeated views without using either model's similarity.
        candidates = {}
        for line in open(base / "instances.jsonl"):
            r = json.loads(line)
            if r["score"] < .6 or (reps is not None and r["id"] not in reps):
                continue
            if min(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) < 48:
                continue
            previous = candidates.get(r["pano"])
            if previous is None or r["score"] > previous["score"]:
                candidates[r["pano"]] = r
        rows = sorted(candidates.values(), key=lambda r: r["id"])
        chosen = random.Random(seed).sample(rows, min(per_source, len(rows)))
        for r in chosen:
            image = base / r["crop"]
            samples.append({"id": f"{source}:{r['id']}", "crop_id": r["id"], "source": source,
                            "date": r.get("date"), "path": str(image), "lat": r["lat"], "lon": r["lon"],
                            "location_kind": "camera", "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
    return samples


def pool_tokens(hidden, registers):
    """DINOv3 registers follow CLS; only image patches enter the mean."""
    e = torch.cat([hidden[:, 0], hidden[:, 1 + registers:].mean(1)], dim=-1).float()
    return torch.nn.functional.normalize(e, dim=-1)


def embed(name, samples, out, batch):
    dest = out / name
    dest.mkdir(exist_ok=True)
    metadata_path = dest / "embedding.json"
    if (dest / "embeddings.npy").exists() and (dest / "ids.json").exists() and metadata_path.exists():
        saved_ids = json.loads((dest / "ids.json").read_text())
        metadata = json.loads(metadata_path.read_text())
        X = np.load(dest / "embeddings.npy")
        if saved_ids != [s["id"] for s in samples] or X.shape != (len(samples), metadata["dimensions"]):
            raise ValueError(f"Saved embedding IDs or dimensions do not match the sample: {name}")
        if not np.isfinite(X).all() or not np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4):
            raise ValueError(f"Invalid saved embedding values: {name}")
        print(f"{name}: reusing {len(X)} saved embeddings", flush=True)
        return X
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        revision = metadata["revision"]
    else:
        revision = HfApi().model_info(MODELS[name]).sha
    processor = AutoImageProcessor.from_pretrained(MODELS[name], revision=revision)
    model = AutoModel.from_pretrained(MODELS[name], revision=revision, dtype=torch.float16).cuda().eval()
    registers = getattr(model.config, "num_register_tokens", 0)
    metadata = {"model": MODELS[name], "revision": revision,
                "pooling": "L2-normalized concatenation of CLS and mean image patch tokens; registers excluded",
                "dtype": "float16 inference, float32 output", "dimensions": model.config.hidden_size * 2,
                "processor": processor.to_dict(), "register_tokens": registers}
    save_json(metadata_path, metadata)
    all_vectors = []
    elapsed, computed = 0., 0
    for start in range(0, len(samples), batch):
        subset = samples[start:start + batch]
        key = hashlib.sha256(json.dumps([s["id"] for s in subset]).encode()).hexdigest()[:12]
        checkpoint = dest / f"batch-{start:06d}-{key}.npy"
        if checkpoint.exists():
            vectors = np.load(checkpoint)
        else:
            images = []
            for s in subset:
                with Image.open(s["path"]) as im:
                    images.append(im.convert("RGB"))
            torch.cuda.synchronize()
            begin = time.monotonic()
            inputs = processor(images=images, return_tensors="pt").to("cuda", torch.float16)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                vectors = pool_tokens(hidden, registers).cpu().numpy()
            elapsed += time.monotonic() - begin
            computed += len(subset)
            tmp = checkpoint.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                np.save(f, vectors)
            tmp.replace(checkpoint)
        if vectors.shape != (len(subset), metadata["dimensions"]) or not np.isfinite(vectors).all():
            raise ValueError(f"Invalid checkpoint: {checkpoint}")
        all_vectors.append(vectors)
        if start % (batch * 16) == 0:
            print(f"{name}: {start + len(subset)}/{len(samples)}", flush=True)
    X = np.concatenate(all_vectors)
    np.save(dest / "embeddings.npy", X)
    save_json(dest / "ids.json", [s["id"] for s in samples])
    if computed:
        save_json(dest / "timing.json", {"computed_crops": computed, "seconds": elapsed,
                                         "crops_per_second": computed / elapsed,
                                         "note": "Includes preprocessing and inference while sharing GPU; not an isolated throughput benchmark."})
    del model
    torch.cuda.empty_cache()
    return X


def analyze(X, samples, name, out, k, seed):
    pca = PCA(min(50, len(X) - 1, X.shape[1]), random_state=seed)
    Z = pca.fit_transform(X)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    similarity = X @ X.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1)[:, :8]
    xy = PCA(2, random_state=seed).fit_transform(X)
    result = {"model": name, "embedding": json.loads((out / name / "embedding.json").read_text()),
              "parameters": {"algorithm": "kmeans", "k": k, "pca_dimensions": pca.n_components_, "seed": seed, "n_init": 10},
              "metrics": {"silhouette_in_model_space": float(silhouette_score(Z, labels)),
                          "pca_explained_variance": float(pca.explained_variance_ratio_.sum())},
              "assignments": [], "neighbors": neighbors.tolist()}
    for i, (s, label) in enumerate(zip(samples, labels)):
        rgb = colorsys.hsv_to_rgb((int(label) * .61803398875) % 1, .58, .88)
        result["assignments"].append({"id": s["id"], "crop_id": s["crop_id"], "source": s["source"],
                                      "cluster_id": f"{out.name}:{name}:{k}:{seed}:{label}", "label": int(label),
                                      "color": "#" + "".join(f"{round(c * 255):02x}" for c in rgb),
                                      "xy": [float(v) for v in xy[i]]})
    save_json(out / name / "analysis.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--per-source", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    a = ap.parse_args()
    if a.per_source < 2 or a.batch < 1:
        ap.error("per-source must be at least 2 and batch must be positive")
    root = Path("data") / a.city / "embedding_comparison"
    root.mkdir(parents=True, exist_ok=True)
    sample_path = root / f"sample-{a.per_source}-{a.seed}.json"
    if not sample_path.exists():
        save_json(sample_path, make_sample(a.city, a.per_source, a.seed))
    samples = json.loads(sample_path.read_text())
    if not 2 <= a.k < len(samples):
        ap.error("k must be at least 2 and smaller than the sample")
    for sample in samples:
        if hashlib.sha256(Path(sample["path"]).read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"Sample image changed: {sample['id']}")
    run_id = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()[:16]
    out = root / run_id
    out.mkdir(exist_ok=True)
    save_json(out / "sample.json", samples)
    results, status = {}, {}
    for name in a.models:
        try:
            X = embed(name, samples, out, a.batch)
            results[name] = analyze(X, samples, name, out, a.k, a.seed)
            status[name] = {"state": "complete"}
        except (OSError, RuntimeError) as e:
            status[name] = {"state": "blocked", "reason": str(e).splitlines()[0]}
            print(name, status[name], flush=True)
            torch.cuda.empty_cache()
        save_json(out / "status.json", status)
    comparison = {"run_id": run_id, "status": status, "models": results, "samples": samples,
                  "queries": random.Random(a.seed).sample(range(len(samples)), min(24, len(samples))),
                  "caveats": ["Exploratory sample, not a labeled accuracy benchmark.",
                              "Equal source counts; Apple candidates use existing representatives, Google candidates are not deduplicated.",
                              "Same pooling recipe and clustering settings; each model uses its official image preprocessing.",
                              "Cluster numbers and colors do not correspond between models.",
                              "Silhouette measures internal separation, not human agreement or which model is better."]}
    if len(results) == 2:
        left, right = (results[n] for n in ("dinov2", "dinov3"))
        comparison["agreement"] = {"adjusted_rand": float(adjusted_rand_score([a["label"] for a in left["assignments"]], [a["label"] for a in right["assignments"]])),
                                   "mean_top8_neighbor_overlap": float(np.mean([len(set(l) & set(r)) / 8 for l, r in zip(left["neighbors"], right["neighbors"])]))}
    for sample in samples:
        sample["image_url"] = os.path.relpath(sample["path"], out)
    save_json(out / "comparison.json", comparison)
    payload = json.dumps(comparison, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html").read_text()
    (out / "index.html").write_text(template.replace("__DATA__", payload))
    print("Report:", out / "index.html", flush=True)


if __name__ == "__main__":
    main()
