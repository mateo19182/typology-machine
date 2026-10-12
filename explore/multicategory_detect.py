"""Resumable multi-category SAM3 detection; one image encoding per source view.

uv run explore/multicategory_detect.py coruna --source apple
Outputs are isolated from legacy single-prompt results. Categories are proposals,
not verified classifier labels. Masks are saved separately from rectangular crops.
"""
import argparse
import hashlib
import json
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from PIL import Image
import torch

from apple_detect import DOWN, MODEL, PAD, Faces, select_panos

PROMPTS = {
    "store_sign": ["store sign"],
    "street_name_plate": ["street name sign", "street name plate"],
    "graffiti": ["graffiti", "mural"],
    "traffic_sign": ["traffic sign"],
    "street_light": ["street light"],
    "bin": ["litter bin", "waste container"],
    "bench": ["bench"],
    "manhole_cover": ["manhole cover"],
    "junction_box": ["electrical cabinet"],
    "door": ["door"],
    "shutter": ["roller shutter"],
}


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    os.replace(temp, path)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


def merge_candidates(rows):
    """Conservative same-category duplicate grouping, retaining every proposal."""
    kept = []
    for row in sorted(rows, key=lambda r: r["score"], reverse=True):
        match = next((r for r in kept if r["category_proposal"] == row["category_proposal"]
                      and iou(r["box"], row["box"]) >= .8), None)
        if match is None:
            kept.append(row)
        else:
            row["duplicate_of"] = match["id"]
    return rows


def legacy_results(base, source):
    folder = base / source / "store_sign"
    done = set((folder / "done.txt").read_text().split()) if (folder / "done.txt").exists() else set()
    rows = {}
    if (folder / "instances.jsonl").exists():
        with (folder / "instances.jsonl").open() as stream:
            lines = stream.readlines()
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a live legacy writer may have an incomplete final line
            key = f"{r['pano']}_{r['face']}" if source == "apple" else f"{r['pano']}_{int(r['view_yaw'])}"
            completion = key if source == "apple" else r["pano"]
            if completion in done:
                rows.setdefault(key, []).append(r)
    return done, rows


def views(base, a):
    if a.source == "apple":
        fetcher = Faces(1)
        try:
            for row in sorted(select_panos(base / "lookaround.csv", a), key=lambda r: r["date"], reverse=True):
                for face in (1, 3):
                    key = f"{row['id']}_{face}"
                    path = base / "apple/faces" / f"{key}.heic"
                    yield key, {**row, "face": face}, path, lambda r=row, f=face: fetcher.fetch(r, f)
        finally:
            fetcher.ex.shutdown(wait=False)
    else:
        from gsv_detect import cut_views
        rows = [json.loads(l) for l in (base / "gsv/panos.jsonl").open()]
        rows = [r for r in rows if r.get("is_current")]
        if a.limit:
            rows = rows[:a.limit]
        for row in rows:
            path = base / "gsv/panos" / f"{row['id']}.jpg"
            if not path.exists():
                continue
            with Image.open(path) as im:
                for yaw, view in cut_views(im.convert("RGB"), 8):
                    yield f"{row['id']}_{int(yaw)}", {**row, "view_yaw": yaw, "view_pitch": 8}, path, lambda v=view: (v, None)


def prefetch(items, load):
    """Load one item ahead while the caller processes the current item."""
    pending = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        for item in items:
            had_previous = pending is not None
            previous = pending.result() if had_previous else None
            pending = pool.submit(load, item)
            if had_previous:
                yield previous
        if pending is not None:
            yield pending.result()


def loaded_views(base, a, out, worker_index, should_stop):
    def pending_views():
        for entry in views(base, a):
            if should_stop():
                break
            key, metadata, source_path, fetch = entry
            if worker_for_panorama(metadata["id"], a.workers) != worker_index:
                continue
            if (out / "views" / key / "result.json").exists():
                continue
            yield entry

    def load(entry):
        key, metadata, source_path, fetch = entry
        if a.source == "apple" and source_path.exists():
            with Image.open(source_path) as im:
                image = im.convert("RGB")
        else:
            result = fetch()
            if result is None:
                return key, metadata, source_path, None
            image, raw = result
            if raw is not None:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(raw)
        return key, metadata, source_path, image

    yield from prefetch(pending_views(), load)


def tensor_tree(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: tensor_tree(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_tree(v, fn) for v in value)
    return value


def worker_for_panorama(pano, workers):
    # Keep all projections of a panorama together, including shared source files.
    return int.from_bytes(hashlib.sha256(str(pano).encode()).digest()[:8], "big") % workers


def reuse_completed_views(out, config, selected_keys):
    """Reuse immutable results when only panorama selection settings differ."""
    def inference_config(value):
        return {k: v for k, v in value.items() if k not in {"thin", "year", "bbox"}}
    reused = 0
    (out / "views").mkdir(exist_ok=True)
    for prior_config in sorted(out.parent.glob("*/config.json")):
        if prior_config.parent == out:
            continue
        if inference_config(json.loads(prior_config.read_text())) != inference_config(config):
            continue
        for result in (prior_config.parent / "views").glob("*/result.json"):
            key = result.parent.name
            target = out / "views" / key
            if key not in selected_keys or target.exists():
                continue
            target.symlink_to(result.parent.resolve(), target_is_directory=True)
            reused += 1
    return reused



def runtime_config(out, config):
    """Extend future views only, leaving the original run identity immutable."""
    path = out / "additional_prompts.json"
    if not path.exists():
        return config
    raw = path.read_bytes()
    extension = json.loads(raw)
    if extension.get("schema_version") != 1 or any(extension.get(k) != config[k] for k in ("model", "revision")):
        raise ValueError("Additional prompts must pin the same model and revision")
    prompts = extension.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("Additional prompts must be a nonempty category mapping")
    for category, values in prompts.items():
        if not isinstance(category, str) or not category or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in category):
            raise ValueError("Invalid additional category name")
        if category in config["prompts"]:
            raise ValueError("Additional prompts cannot replace existing categories")
        if not isinstance(values, list) or not values or any(not isinstance(p, str) or not p.strip() for p in values) or len(set(values)) != len(values):
            raise ValueError("Each additional category needs unique nonempty prompts")
    return {**config, "prompts": {**config["prompts"], **prompts},
            "additional_prompts_sha256": hashlib.sha256(raw).hexdigest()}


def attempted_provenance(config):
    # Includes categories producing zero boxes, distinguishing them from old
    # views where a new category has never been attempted.
    return {"attempted_categories": list(config["prompts"]),
            "attempted_prompts": config["prompts"],
            "additional_prompts_sha256": config.get("additional_prompts_sha256")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("city", nargs="?", default="coruna")
    ap.add_argument("--source", choices=["apple", "gsv"], default="apple")
    ap.add_argument("--year")
    ap.add_argument("--bbox")
    ap.add_argument("--thin", type=float, default=6)
    ap.add_argument("--limit", type=int, help="limit panoramas, not source views")
    ap.add_argument("--threshold", type=float, default=.5)
    ap.add_argument("--cache-samples", type=int, default=0, help="persist this many feature samples and measure reload time")
    ap.add_argument("--torch-threads", type=int, default=2, help="CPU torch threads per GPU worker")
    ap.add_argument("--workers", type=int, default=1, help="concurrent GPU processes with disjoint panorama assignments")
    ap.add_argument("--resume-run", type=Path, help="resume an existing run using its pinned model revision; repeat its selection flags")
    ap.add_argument("--max-new-views", type=int, help="stop each worker after this many new views, for throughput checks")
    a = ap.parse_args()
    if a.workers < 1 or a.torch_threads < 1 or (a.max_new_views is not None and a.max_new_views < 1):
        ap.error("workers and max-new-views must be positive")
    from huggingface_hub import HfApi
    import transformers

    base = (Path("data") / a.city).resolve()
    recorded = json.loads((a.resume_run / "config.json").read_text()) if a.resume_run else None
    revision = recorded["revision"] if recorded else HfApi().model_info(MODEL).sha
    config = {"model": MODEL, "revision": revision, "transformers": transformers.__version__,
              "prompts": PROMPTS, "threshold": a.threshold, "mask_threshold": .5,
              "mask_downsample": DOWN, "padding": PAD, "min_side": 32,
              "source": a.source, "year": a.year, "bbox": a.bbox, "thin": a.thin,
              "gsv_projection": {"fov": [70, 55], "size": [1600, 2048], "pitch": 8, "views": 8},
              "schema_version": 1, "legacy_store_sign": "reuse completed .5 threshold detections"}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out = base / a.source / "multicategory" / run_id
    if recorded is not None and (config != recorded or a.resume_run.resolve() != out):
        ap.error("Resume configuration differs. Repeat the recorded source, thin, year, bbox and threshold flags.")
    out.mkdir(parents=True, exist_ok=True)
    # Keep a process lock for the entire run, preventing concurrent writes on resume.
    import fcntl
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(out / "config.json", config)
    print(f"output: {out}", flush=True)
    reused = 0
    if a.source == "apple":
        selected = {f"{row['id']}_{face}" for row in select_panos(base / "lookaround.csv", a) for face in (1, 3)}
        reused = reuse_completed_views(out, config, selected)
    print(f"reused {reused} completed views from compatible prior runs", flush=True)
    config = runtime_config(out, config)
    atomic_json(out / "execution.json", {"workers": a.workers, "started_at": time.time(),
        "reused_prior_views": reused, "effective_prompts": config["prompts"],
        "additional_prompts_sha256": config.get("additional_prompts_sha256"), "cached_text_embeddings": True, "prefetch_views_per_worker": 1,
        "assignment": "sha256(panorama id) modulo workers", "torch_threads_per_worker": a.torch_threads})
    if a.workers == 1:
        run_worker(base, a, out, config, 0)
        return
    context = multiprocessing.get_context("spawn")
    children = []
    stop_requested = False
    def stop_children(*_):
        nonlocal stop_requested
        stop_requested = True
        for child in children:
            if child.is_alive():
                try:
                    os.kill(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    try:
        for index in range(a.workers):
            if stop_requested:
                break
            child = context.Process(target=run_worker, args=(base, a, out, config, index))
            child.start()
            children.append(child)
        while any(child.is_alive() for child in children):
            if any(child.exitcode not in (None, 0) for child in children):
                stop_children()
                raise RuntimeError("SAM worker failed; stopping peers. Completed views remain resumable.")
            for child in children:
                child.join(timeout=.2)
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError("SAM worker failed")
    finally:
        stop_children()
        for child in children:
            child.join()


def run_worker(base, a, out, config, worker_index):
    from transformers import Sam3Model, Sam3Processor
    torch.set_num_threads(getattr(a, "torch_threads", 2))
    revision = config["revision"]
    done, legacy = legacy_results(base, a.source)
    processor = Sam3Processor.from_pretrained(MODEL, revision=revision)
    model = Sam3Model.from_pretrained(MODEL, revision=revision, dtype=torch.bfloat16).cuda().eval()
    print(f"worker {worker_index}: attention backend {model.config._attn_implementation}", flush=True)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        # Signal handlers only set state: buffered stdout can be reentered.

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Text prompts are constant across images. Retain the attention mask as well.
    text_cache = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for prompts in config["prompts"].values():
            for prompt in prompts:
                text = processor(text=prompt, return_tensors="pt").to("cuda")
                text_cache[prompt] = (model.get_text_features(**text, return_dict=True), text.get("attention_mask"))
    processed = cached = 0
    begin = time.monotonic()
    for key, metadata, source_path, image in loaded_views(base, a, out, worker_index, lambda: stopping):
        if stopping:
            break
        case = out / "views" / key
        if shutil.disk_usage(out).free < 20 * 1024**3:
            raise RuntimeError("Stopping with 20 GiB free; completed views are resumable")
        if image is None:
            print(f"fetch failed: {key}", flush=True)
            continue
        case.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            vision = model.get_vision_features(pixel_values=inputs.pixel_values)
        torch.cuda.synchronize()
        encode_seconds = time.monotonic() - start
        if cached < a.cache_samples:
            payload = {"config": config, "metadata": metadata, "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                       "image_size": image.size, "processor": processor.image_processor.to_dict(),
                       "features": tensor_tree(dict(vision), lambda t: t.detach().cpu())}
            cache_path = case / "features.pt"
            torch.save(payload, cache_path)
            t = time.monotonic()
            loaded = torch.load(cache_path, weights_only=True)
            restored = type(vision)(**tensor_tree(loaded["features"], lambda t: t.cuda()))
            torch.cuda.synchronize()
            atomic_json(case / "cache_measurement.json", {"bytes": cache_path.stat().st_size,
                        "reload_to_gpu_seconds": time.monotonic()-t, "encode_seconds": encode_seconds})
            vision = restored  # exercise the persisted representation in actual inference
            del payload, loaded, restored
            cached += 1
        predictions = []
        legacy_key = key if a.source == "apple" else str(metadata["id"])
        reuse = a.threshold == .5 and legacy_key in done
        if reuse:
            for row in legacy.get(key, []):
                predictions.append({**row, "id": f"legacy:{a.source}:{row['id']}", "category_proposal": "store_sign",
                    "prompt": "store sign", "legacy": True,
                    "crop": str(base / a.source / "store_sign" / row["crop"]),
                    "mask": str(base / a.source / "store_sign" / row["mask"])})
        w, h = image.size
        for category, prompts in config["prompts"].items():
            if category == "store_sign" and reuse:
                continue
            for prompt_index, prompt in enumerate(prompts):
                text_embeds, attention_mask = text_cache[prompt]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(vision_embeds=vision, text_embeds=text_embeds, attention_mask=attention_mask)
                result = processor.post_process_instance_segmentation(outputs, threshold=a.threshold,
                    mask_threshold=.5, target_sizes=[(h//DOWN, w//DOWN)])[0]
                # Transfer once per prompt, avoiding GPU synchronization per coordinate/crop.
                for field in ("scores", "boxes", "masks"):
                    result[field] = result[field].cpu()
                for index, (score, box, mask) in enumerate(zip(result["scores"], result["boxes"], result["masks"])):
                    x0, y0, x1, y1 = [float(v)*DOWN for v in box]
                    if min(x1-x0, y1-y0) < 32:
                        continue
                    pw, ph = (x1-x0)*PAD, (y1-y0)*PAD
                    crop_box = [int(max(0, x0-pw)), int(max(0, y0-ph)), int(min(w, x1+pw)), int(min(h, y1+ph))]
                    name = f"{category}-{prompt_index}-{index}"
                    image.crop(crop_box).save(case / f"{name}.jpg", quality=90)
                    Image.fromarray(mask.cpu().numpy().astype(np.uint8)*255).crop(
                        tuple(int(v/DOWN) for v in crop_box)).save(case / f"{name}.png")
                    predictions.append({"id": f"{a.source}:{key}:{name}", "category_proposal": category,
                        "prompt": prompt, "score": float(score), "box": [x0,y0,x1,y1], "crop_box": crop_box,
                        "crop": str(case / f"{name}.jpg"), "mask": str(case / f"{name}.png")})
                del outputs, result
        atomic_json(case / "result.json", {"source": a.source, "view_id": key, "metadata": metadata,
            "source_path": str(source_path), "image_size": [w,h], "legacy_store_sign_reused": reuse,
            "encode_seconds": encode_seconds, "seconds": time.monotonic()-start,
            "predictions": merge_candidates(predictions), **attempted_provenance(config)})
        del vision, inputs
        processed += 1
        print(f"worker {worker_index}: {processed} views; {key}: {len(predictions)} candidates; {(time.monotonic()-begin)/processed:.2f}s/view", flush=True)
        if a.max_new_views and processed >= a.max_new_views:
            break
    print(f"worker {worker_index}: saved {processed} views; output: {out}", flush=True)


if __name__ == "__main__":
    main()
