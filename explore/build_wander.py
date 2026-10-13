"""Build a bounded, offline DINO visual-wandering graph from the published index.

No detection or request-time inference. Images and descriptors remain outside Git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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


def neighbors(embeddings, rows, count=8, palettes=None, min_similarity=.65):
    """Rank strong, similarly proportioned objects in shape and colour modes."""
    result = []; colored = []; stats = []
    aspect = np.array([r.get('aspect', 1.) for r in rows])
    def pano(row):
        return row['source'], row['view_id'].rsplit('_', 1)[0]
    for start in range(0, len(rows), 128):
        scores = embeddings[start:start+128] @ embeddings.T
        palette_scores = palettes[start:start+128] @ palettes.T if palettes is not None else None
        for offset, values in enumerate(scores):
            i = start + offset
            ratio = np.abs(np.log(aspect / aspect[i]))
            eligible = np.array([j != i and pano(rows[i]) != pano(row)
                                and rows[i]['object_id'] != row['object_id']
                                for j, row in enumerate(rows)])
            eligible &= (values >= min_similarity) & (ratio <= np.log(1.6))
            candidates = np.flatnonzero(eligible)
            if len(candidates):
                candidates = candidates[values[candidates] >= values[candidates].max() - .10]
            shape = values - .10 * ratio
            order = candidates[np.argsort(-shape[candidates], kind='stable')][:count]
            result.append([int(j) for j in order])
            if palettes is not None:
                # Histograms are square-root normalized: dot product is palette affinity.
                affinity = palette_scores[offset]
                color_score = .82 * values + .18 * affinity - .10 * ratio
                color_order = candidates[np.argsort(-color_score[candidates], kind='stable')][:count]
            else:
                color_order = order
            colored.append([int(j) for j in color_order])
            stats.extend(float(values[j]) for j in order)
    return result, colored, stats


def palette(image):
    """Small joint Lab histogram; captures muted palettes as well as saturated ones."""
    import cv2
    pixels = cv2.cvtColor(np.asarray(image.resize((64, 64))), cv2.COLOR_RGB2LAB)
    hist = cv2.calcHist([pixels], [0, 1, 2], None, [8, 8, 8], [0, 256]*3).ravel()
    return np.sqrt(hist / max(float(hist.sum()), 1.))


def load_crop(record, args):
    """Prefetch a bounded batch from disk; decode a crop at most once per build."""
    if not record.get('bx') or not record.get('view_id') or not record.get('raw', '').startswith('/proposal-context/'):
        return None
    box = record['bx']
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    crop = args.site / record['crop'].lstrip('/')
    try:
        stat = crop.stat()
        row = {k: record.get(k) for k in ('id', 'category', 'source', 'view_id', 'raw', 'crop', 'bx', 'w', 'h', 'd', 'barrio', 'la', 'lo', 'provider_url')}
        row['object_id'] = record.get('duplicate_group') or record['id']
        row['text'] = str(record.get('text') or record.get('t') or '')[:120]
        key = hashlib.sha256(f'{MODEL}|letterbox224|{record["id"]}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()
        cached = args.cache / (key+'.npy')
        palette_cache = args.cache / (key+'.palette.npz')
        vector = None; color = None; image = None
        if cached.exists():
            try:
                vector = np.load(cached, allow_pickle=False)
                if vector.shape != (1536,) or not np.isfinite(vector).all(): vector = None
            except (OSError, ValueError): pass
        if palette_cache.exists():
            try:
                with np.load(palette_cache, allow_pickle=False) as saved:
                    color = saved['color']; row['aspect'] = float(saved['aspect'])
            except (OSError, ValueError): pass
        if color is None or vector is None:
            with Image.open(crop) as original:
                image = original.convert('RGB')
            if color is None:
                row['aspect'] = image.width / image.height
                color = palette(image)
                np.savez(palette_cache, color=color, aspect=row['aspect'])
        if vector is None:
            image = ImageOps.pad(image, (224, 224), color=(128, 128, 128))
        return row, cached, color, vector, image
    except (OSError, ValueError):
        return None


def build(args):
    import torch
    import cv2
    cv2.setNumThreads(2)
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
    allrows = []; allneighbors = []; color_neighbors = []; counts = {}; failures = 0; similarities = []
    for category in CATEGORIES:
        ids = [r[0] for r in db.execute('SELECT id FROM objects WHERE category=? AND score>=.75', (category,))]
        # Stable sample across the archive, not a single neighbourhood or identifier range.
        selected = heapq.nsmallest(args.per_category * 2, ids, key=lambda x: hashlib.sha256(x.encode()).digest())
        rows = []; vectors = []; colors = []; pending = []; pending_images = []
        def flush():
            if not pending:
                return
            inputs = proc(images=pending_images, return_tensors='pt', do_resize=False, do_center_crop=False).to(device, dtype)
            with torch.inference_mode():
                h = model(**inputs).last_hidden_state
                e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float()
                e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
            for (row, cache, color), vector in zip(pending, e):
                np.save(cache, vector)
                rows.append(row); vectors.append(vector); colors.append(color)
            pending.clear(); pending_images.clear()
        # Queue only 64 crops at a time. Concurrent reads avoid serial seek waits
        # on the archive disk; inference and graph ordering remain deterministic.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(selected), 64):
                records = [json.loads(db.execute('SELECT payload FROM records WHERE id=?', (identity,)).fetchone()[0])
                           for identity in selected[start:start+64]]
                for prepared in pool.map(lambda record: load_crop(record, args), records):
                    if prepared is None:
                        failures += 1; continue
                    row, cached, color, vector, image = prepared
                    if vector is not None:
                        rows.append(row); vectors.append(vector); colors.append(color)
                    else:
                        pending.append((row, cached, color)); pending_images.append(image)
                        if len(pending) >= args.batch: flush()
                    if len(rows) + len(pending) >= args.per_category: break
                if start % 512 == 0:
                    print(category, len(rows)+len(pending), '/', args.per_category, 'crops', flush=True)
                if len(rows) + len(pending) >= args.per_category: break
        flush()
        if rows:
            links, color_links, scores = neighbors(np.asarray(vectors), rows, palettes=np.asarray(colors), min_similarity=args.min_similarity)
            similarities.extend(scores)
            base = len(allrows)
            allneighbors.extend([[base+j for j in link] for link in links]); allrows.extend(rows)
            color_neighbors.extend([[base+j for j in link] for link in color_links])
        counts[category] = len(rows)
        print(category, len(rows), 'indexed', flush=True)
    if not allrows:
        raise ValueError('No eligible crops; existing graph preserved')
    # All alignment work stays offline. Feature extraction is done once per involved crop.
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    from wander_alignment import extract_features, fit_alignment, detection_features
    pairs = sorted({(i, j) for mode in (allneighbors, color_neighbors)
                    for i, links in enumerate(mode) for j in links[:6]})
    involved = sorted({n for pair in pairs for n in pair})
    features = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(involved), 128):
            batch = involved[start:start+128]
            extracted = pool.map(lambda n: extract_features(args.site / allrows[n]['crop'].lstrip('/')), batch)
            features.update(zip(batch, extracted))
            if start % 512 == 0:
                print('Alignment features', len(features), '/', len(involved), flush=True)
    alignments = {}; detection_cache = {}
    def object_features(n):
        if n not in detection_cache:
            row = allrows[n]
            path = (args.site / row['crop'].lstrip('/')).resolve()
            try:
                result = json.loads((path.parent/'result.json').read_text())
                prediction = next(p for p in result['predictions'] if p['id'] == row['id'])
                width, height = result['image_size']
                box = [max(0, min(width if k%2 == 0 else height, v)) for k,v in enumerate(prediction['box'])]
                normalized = [v/(width if k%2 == 0 else height) for k,v in enumerate(box)]
                if not np.allclose(normalized, row['bx'], atol=.000002):
                    raise ValueError('Published object box differs from detection')
                detection_cache[n] = detection_features(features[n], box, prediction.get('crop_box', prediction['box']))
            except (OSError, ValueError, KeyError, StopIteration):
                detection_cache[n] = None
        return detection_cache[n]
    for done, (i, j) in enumerate(pairs, 1):
        # Both directions are fitted independently, keeping the same acceptance rules.
        match = fit_alignment(features[i], features[j])
        if match:
            source, destination = object_features(i), object_features(j)
            match = fit_alignment(source, destination) if source is not None and destination is not None else None
            if match:
                alignments[f'{i}:{j}'] = match
        if done % 1000 == 0:
            print('Alignment pairs', done, '/', len(pairs), 'accepted', len(alignments), flush=True)
    print('Shape cosine quantiles', np.quantile(similarities, [0, .25, .5, .75, 1]).tolist() if similarities else [], flush=True)
    result = dict(version=2, snapshot=meta['query_snapshot'], built_at=datetime.now(timezone.utc).isoformat(),
                  model=MODEL, method='DINO similarity with proportion filtering and optional Lab palette ranking; robust local-feature alignment. Similarity is not object identity.',
                  confidence=.75, categories=counts, skipped_images=failures, items=allrows, neighbors=allneighbors, color_neighbors=color_neighbors, alignments=alignments,
                  matching=dict(min_similarity=args.min_similarity, max_aspect_ratio=1.6, max_neighbors=8,
                                alignment_pairs=len(pairs), aligned_pairs=len(alignments)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')));os.replace(tmp,args.output)
    print('Published',len(allrows),'items to',args.output,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',type=Path,default=ROOT/'data/coruna/site')
    p.add_argument('--output',type=Path,default=ROOT/'data/coruna/site/wander-data.json')
    p.add_argument('--cache',type=Path,default=Path.home()/'.local/state/typology-machine/wander-features')
    p.add_argument('--per-category',type=int,default=3000)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--min-similarity', type=float, default=.65)
    args = p.parse_args()
    if not 1 <= args.per_category <= 10000 or not .5 <= args.min_similarity <= .95:
        p.error('per-category must be 1–10000 and min-similarity .5–.95')
    build(args)
