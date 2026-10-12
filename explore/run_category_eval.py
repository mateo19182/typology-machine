"""Run one category model on the frozen review set. Results checkpoint per crop.

uv run explore/run_category_eval.py --model intern2
uv run explore/run_category_eval.py --model qwen2
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from PIL import Image

from category_eval import BASE, read_jsonl, write_json

MODELS = {"intern2": "internlm/Intern-Decision-2B", "qwen2": "Qwen/Qwen3.5-2B"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--revision")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--gpu-memory-fraction", type=float)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    base = args.base
    sample = json.loads((base / "sample.json").read_text())
    repo = MODELS[args.model]
    revision = args.revision or HfApi().model_info(repo).sha
    config = {"sample_id": sample["id"], "model": repo, "revision": revision,
              "instruction": sample["instruction"], "categories": sample["categories"],
              "input": "Frozen rectangular RGB crop, aspect preserved, longest side <= 768px; no mask",
              "mode": "native decision head" if args.model == "intern2" else "greedy category generation; thinking disabled",
              "schema_version": 1}
    run_id = args.model + "-" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out = base / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run.json", config)
    done = {r["id"] for r in read_jsonl(out / "predictions.jsonl") if r.get("category")}
    pending = [r for r in sample["items"] if r["id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("All predictions cached", run_id)
        return
    print(f"{repo}: {len(pending)} remaining", flush=True)
    if args.device == "cuda" and args.gpu_memory_fraction is not None:
        import torch
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        if free < 8 * 1024**3:
            raise RuntimeError("Need at least 8 GiB free before shared GPU evaluation")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    if args.model == "intern2":
        path = snapshot_download(repo, revision=revision, local_files_only=args.offline)
        spec = importlib.util.spec_from_file_location("intern_category_inference", path + "/inference.py")
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        engine = module.DecisionEngine(checkpoint=path, device=args.device)

        def predict(item):
            answer = engine.predict({"state": "Street photograph crop from A Coruña, Spain.",
                                     "images": [str(base / item["image"])],
                                     "questions": {"category": {"type": "choice", "instructions": sample["instruction"],
                                                                 "criteria": sample["categories"]}}})["answers"]["category"]
            return {"category": answer["choice"], "confidence": answer["confidence"], "probabilities": answer["probabilities"]}
    else:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(repo, revision=revision)
        model = AutoModelForImageTextToText.from_pretrained(repo, revision=revision, dtype=torch.bfloat16).to(args.device).eval()
        prompt = sample["instruction"] + "\n\nCategories:\n" + "\n".join(f"{k}: {v}" for k, v in sample["categories"].items())

        def predict(item):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            with Image.open(base / item["image"]) as image:
                inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to(args.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            category = raw.strip('"\'` \n')
            if category not in sample["categories"]:
                return {"category": None, "error": "invalid_category", "raw": raw}
            return {"category": category, "confidence": None, "raw": raw}
    with open(out / "predictions.jsonl", "a") as log:
        for i, item in enumerate(pending):
            if hashlib.sha256((base / item["image"]).read_bytes()).hexdigest() != item["evaluation_sha256"]:
                raise ValueError(f"Evaluation image changed: {item['id']}")
            begin = time.monotonic()
            result = predict(item)
            elapsed = time.monotonic() - begin
            record = {"id": item["id"], "run_id": run_id, "seconds": elapsed, **result}
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            if i % 20 == 0:
                print(f"{i+1}/{len(pending)} {elapsed:.2f}s/crop", flush=True)
    print("Done", out, flush=True)


if __name__ == "__main__":
    main()
