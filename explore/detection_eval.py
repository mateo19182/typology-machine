"""Small SAM prompt/threshold review on 30 retained source views, sharing image features.

uv run explore/detection_eval.py
Masks are retained as compressed arrays; the viewer draws boxes only.
"""
import json
import time
from pathlib import Path

import numpy as np
import pillow_heif
import py360convert
import torch
from huggingface_hub import HfApi
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from category_eval import BASE, ROOT, read_jsonl, write_json

PROMPTS = ["store sign", "street name sign", "street name plate", "graffiti", "mural",
           "store sign, street name sign, graffiti"]
MODEL = "JobHarvest/sam3"


def main():
    pillow_heif.register_heif_opener()
    out = BASE / "detector"
    out.mkdir(exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        config = json.loads(manifest.read_text())
    else:
        sample = json.loads((BASE / "sample.json").read_text())
        views = []
        for source in ("apple", "gsv"):
            ids = {i["id"].split(":", 1)[1] for i in sample["items"] if i["source"] == source}
            rows = {r["id"]: r for r in read_jsonl(ROOT / "data/coruna" / source / "store_sign/instances.jsonl") if r["id"] in ids}
            for sid in sorted(ids):
                r = rows[sid]
                path = ROOT / "data/coruna" / source / (f"faces/{r['pano']}_{r['face']}.heic" if source == "apple" else f"panos/{r['pano']}.jpg")
                if path.exists():
                    views.append({"id": f"{source}-{sid}", "source": source, "path": str(path), "yaw": r.get("view_yaw"), "date": r["date"]})
                if sum(v["source"] == source for v in views) == 15:
                    break
        config = {"model": MODEL, "revision": HfApi().model_info(MODEL).sha, "prompts": PROMPTS,
                  "min_threshold": .3, "mask_threshold": .5, "mask_downsample": 4, "views": views,
                  "note": "Source views selected from the crop review sample, not a random coverage survey. Combined prompt is an experiment and does not assign individual category labels."}
        write_json(manifest, config)
    processor = Sam3Processor.from_pretrained(MODEL, revision=config["revision"])
    model = Sam3Model.from_pretrained(MODEL, revision=config["revision"], dtype=torch.bfloat16).cuda().eval()
    for index, view in enumerate(config["views"]):
        case = out / view["id"]
        case.mkdir(exist_ok=True)
        if (case / "result.json").exists():
            continue
        with Image.open(view["path"]) as source:
            image = source.convert("RGB")
        if view["source"] == "gsv":
            image = Image.fromarray(py360convert.e2p(np.asarray(image), fov_deg=(70., 55.), u_deg=view["yaw"], v_deg=8., out_hw=(1600, 2048), mode="bilinear"))
        preview = image.copy()
        preview.thumbnail((1600, 1600))
        preview.save(case / "image.jpg", quality=90)
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        begin = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            vision = model.get_vision_features(pixel_values=inputs.pixel_values)
        predictions = []
        h, w = image.height // 4, image.width // 4
        for j, prompt in enumerate(config["prompts"]):
            text = processor(text=prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(vision_embeds=vision, **text)
            result = processor.post_process_instance_segmentation(outputs, threshold=.3, mask_threshold=.5, target_sizes=[(h, w)])[0]
            boxes = result["boxes"].float().cpu().numpy() / np.array([w, h, w, h])
            scores = result["scores"].float().cpu().numpy()
            masks = result["masks"].cpu().numpy().astype(bool)
            np.savez_compressed(case / f"masks-{j}.npz", masks=masks)
            predictions.append({"prompt": prompt, "boxes": boxes.tolist(), "scores": scores.tolist(), "mask_file": f"masks-{j}.npz"})
            del outputs, result
        write_json(case / "result.json", {"view": view, "predictions": predictions, "seconds": time.monotonic() - begin})
        print(f"{index+1}/{len(config['views'])} source views", flush=True)
        del vision, inputs
    print("Detector review complete", out)


if __name__ == "__main__":
    main()
