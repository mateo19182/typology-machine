"""Build a bounded, offline DINO visual-wandering graph from the published index.

No detection or request-time inference. Images and descriptors remain outside Git.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'facebook/dinov2-base'
CATEGORIES = ['graffiti', 'store_sign', 'street_name_plate', 'door', 'shutter']


def neighbors(embeddings, rows, count=16):
    """Exclude self, same source panorama, and saved duplicate groups."""
    result = []
    def pano(row):
        return row['source'], row['view_id'].rsplit('_', 1)[0]
    for start in range(0, len(rows), 128):
        scores = embeddings[start:start+128] @ embeddings.T
        for offset, values in enumerate(scores):
            i = start + offset
            ranked = np.argsort(-values, kind='stable')
            result.append([int(j) for j in ranked if j != i and pano(rows[i]) != pano(rows[j])
                           and rows[i]['object_id'] != rows[j]['object_id']][:count])
    return result


def build(args):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(2)
    state = Path.home() / '.local/state/typology-machine/query'
    meta = json.loads((state / 'current.json').read_text())
    db = sqlite3.connect(f"file:{state / (meta['query_snapshot']+'.sqlite3')}?mode=ro", uri=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    proc = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    device = args.device
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(MODEL, dtype=dtype, local_files_only=True).to(device).eval()
    allrows = []; allneighbors = []; counts = {}; failures = 0
    for category in CATEGORIES:
        ids = [r[0] for r in db.execute('SELECT id FROM objects WHERE category=? AND score>=.75', (category,))]
        # Stable sample across the archive, not a single neighbourhood or identifier range.
        selected = heapq.nsmallest(args.per_category * 2, ids, key=lambda x: hashlib.sha256(x.encode()).digest())
        rows = []; vectors = []; pending = []; pending_images = []
        def flush():
            if not pending:
                return
            inputs = proc(images=pending_images, return_tensors='pt', do_resize=False, do_center_crop=False).to(device, dtype)
            with torch.inference_mode():
                h = model(**inputs).last_hidden_state
                e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float()
                e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
            for (row, cache), vector in zip(pending, e):
                np.save(cache, vector)
                rows.append(row); vectors.append(vector)
            pending.clear(); pending_images.clear()
        for identity in selected:
            record = json.loads(db.execute('SELECT payload FROM records WHERE id=?', (identity,)).fetchone()[0])
            if not record.get('bx') or not record.get('view_id') or not record.get('raw', '').startswith('/proposal-context/'):
                continue
            crop = args.site / record['crop'].lstrip('/')
            if not crop.is_file():
                failures += 1; continue
            box = record['bx']
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            row = {k: record.get(k) for k in ('id', 'category', 'source', 'view_id', 'raw', 'crop', 'bx', 'w', 'h', 'd', 'barrio', 'la', 'lo', 'provider_url')}
            row['object_id'] = record.get('duplicate_group') or identity
            row['text'] = str(record.get('text') or record.get('t') or '')[:120]
            stat = crop.stat()
            key = hashlib.sha256(f'{MODEL}|letterbox224|{identity}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()
            cached = args.cache / (key+'.npy')
            if cached.exists():
                vector = np.load(cached, allow_pickle=False)
                rows.append(row); vectors.append(vector)
            else:
                try:
                    with Image.open(crop) as im:
                        im = ImageOps.pad(im.convert('RGB'), (224, 224), color=(128, 128, 128))
                except (OSError, ValueError):
                    failures += 1; continue
                pending.append((row, cached)); pending_images.append(im)
                if len(pending) >= args.batch:
                    flush()
            if len(rows) + len(pending) >= args.per_category:
                break
        flush()
        if rows:
            links = neighbors(np.asarray(vectors), rows)
            base = len(allrows)
            allneighbors.extend([[base+j for j in link] for link in links]); allrows.extend(rows)
        counts[category] = len(rows)
        print(category, len(rows), 'indexed', flush=True)
    if not allrows:
        raise ValueError('No eligible crops; existing graph preserved')
    result = dict(version=1, snapshot=meta['query_snapshot'], built_at=datetime.now(timezone.utc).isoformat(),
                  model=MODEL, method='DINO crop similarity; different panoramas, same category. Similarity is not object identity.',
                  confidence=.75, categories=counts, skipped_images=failures, items=allrows, neighbors=allneighbors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')));os.replace(tmp,args.output)
    print('Published',len(allrows),'items to',args.output,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',type=Path,default=ROOT/'data/coruna/site')
    p.add_argument('--output',type=Path,default=ROOT/'data/coruna/site/wander-data.json')
    p.add_argument('--cache',type=Path,default=Path.home()/'.local/state/typology-machine/wander-features')
    p.add_argument('--per-category',type=int,default=500)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    build(p.parse_args())
