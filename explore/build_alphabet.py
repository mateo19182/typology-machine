"""Build a small editable alphabet starter from OCR on existing sign crops.

Runs bounded CPU OCR, never detection or panorama decoding. Character boxes come from the recognizer's CTC alignment and remain editable.
Generated imagery stays in the ignored site data directory.
"""
import argparse
import base64
import io
import hashlib
from collections import Counter
import json
import os
from pathlib import Path
import string
import urllib.parse
import urllib.request


def character_regions(line):
    """Use the recognizer's CTC-aligned character boxes, never equal text slices.

    RapidOCR may omit spaces/punctuation from its character labels while keeping
    their confidences. Reject mismatched output rather than attach wrong scores.
    """
    if len(line) != 6:
        return []
    _, text, confidence, boxes, characters, scores = line
    if confidence < .96 or len(boxes) != len(characters) or len(scores) != len(characters):
        return []
    return [(char.upper(), box, float(score)) for char, box, score in zip(characters, boxes, scores)
            if len(char) == 1 and score >= .98]


def padded_character_box(polygon, previous=None, following=None):
    """Add at most four pixels horizontally without crossing neighbor midpoints."""
    import numpy as np
    quad = np.asarray(polygon, dtype=np.float32).copy()
    direction = quad[1] - quad[0]
    width = float(np.linalg.norm(direction))
    if width <= 0:
        return quad
    direction /= width
    left = right = min(4., width * .12)
    if previous is not None:
        gap = float(np.dot(quad[0] - np.asarray(previous)[1], direction))
        left = min(left, max(0., gap / 2))
    if following is not None:
        gap = float(np.dot(np.asarray(following)[0] - quad[1], direction))
        right = min(right, max(0., gap / 2))
    vertical = quad[3] - quad[0]
    height = float(np.linalg.norm(vertical))
    if height > 0:
        vertical *= min(3., height * .03) / height
    quad[[0, 3]] -= direction * left
    quad[[1, 2]] += direction * right
    quad[[0, 1]] -= vertical
    quad[[2, 3]] += vertical
    return quad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--api', default='http://localhost:8794')
    ap.add_argument('--site', type=Path, default=Path('data/coruna/site'))
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--variants', type=int, default=8)
    ap.add_argument('--output', type=Path, help='Write a review candidate file instead of publishing the starter')
    args = ap.parse_args()
    if not 1 <= args.limit <= 1500:
        ap.error('--limit must be between 1 and 1500')
    if not 1 <= args.variants <= 40:
        ap.error('--variants must be between 1 and 40')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import cv2
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR
    cv2.setNumThreads(2)
    model = Path(__file__).resolve().parent.parent / 'data/models/ocr/latin_PP-OCRv5_rec_mobile.onnx'
    engine = RapidOCR(rec_model_path=str(model), intra_op_num_threads=2, inter_op_num_threads=1,
                      det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False)

    def api(route, **params):
        with urllib.request.urlopen(args.api + route + '?' + urllib.parse.urlencode(params), timeout=40) as response:
            return json.load(response)

    candidates = []
    seen = set()
    # Round-robin searches keep rare characters represented in a larger sample.
    queries = ['k', 'w', 'x', 'j', 'ñ', 'z', 'q', 'á', 'ú', 'ü', '0', 'farmacia', 'coruña', 'bar', '']
    for offset in range(0, 240, 80):
        batches = [api('/api/query', layers='store_sign,street_name_plate', confidence='.65',
                       q=query, limit=80, offset=offset, map=0)['items'] for query in queries]
        for batch in batches:
            batch.sort(key=lambda row: min(row.get('w',0), row.get('h',0)), reverse=True)
        for i in range(80):
            for rows in batches:
                if i >= len(rows):
                    continue
                row = rows[i]
                if row['id'] not in seen:
                    seen.add(row['id']); candidates.append(row)
        if len(candidates) >= args.limit:
            break
    cache_dir = args.site.parent / 'alphabet-ocr-cache'
    cache_dir.mkdir(exist_ok=True)
    rejected = Counter()
    choices = {}
    processed = 0
    for row in candidates[:args.limit]:
        path = args.site / row['crop'].lstrip('/')
        if not path.is_file():
            continue
        try:
            image = Image.open(path).convert('RGB')
            cache = cache_dir / (hashlib.sha256(row['crop'].encode()).hexdigest() + '.json')
            if cache.exists():
                lines = json.loads(cache.read_text())
            else:
                lines, _ = engine(str(path), return_word_box=True)
                cache.write_text(json.dumps(lines))
        except Exception as error:
            print(f'Skipped crop: {type(error).__name__}', flush=True)
            continue
        processed += 1
        for line in lines or []:
            text = line[1]
            for letter, polygon, confidence in character_regions(line):
                if letter not in string.ascii_uppercase + 'ÑÁÉÍÓÚÜ0123456789':
                    continue
                index = line[3].index(polygon)
                quad = padded_character_box(polygon, line[3][index-1] if index else None,
                                           line[3][index+1] if index+1 < len(line[3]) else None)
                quad[:,0] = np.clip(quad[:,0], 0, image.width-1)
                quad[:,1] = np.clip(quad[:,1], 0, image.height-1)
                width = int(max(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[2]-quad[3])))
                height = int(max(np.linalg.norm(quad[3]-quad[0]), np.linalg.norm(quad[2]-quad[1])))
                if height < 28 or width < 8 or height / width > 12 or width / height > 2:
                    continue
                warp = cv2.getPerspectiveTransform(quad, np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32))
                cut = Image.fromarray(cv2.warpPerspective(np.asarray(image), warp, (width,height)))
                luminance = np.asarray(cut.convert('L'))
                contrast = float(np.percentile(luminance, 95) - np.percentile(luminance, 5))
                if contrast < 45:
                    continue  # Empty or nearly uniform boxes are not useful letters.
                normalized = cv2.resize(luminance, (max(8,round(64*width/height)),64))
                sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
                if sharpness < 35:
                    rejected['blur'] += 1; continue
                recognized, _ = engine(np.asarray(cut), use_det=False, use_cls=False)
                if not recognized or recognized[0][0].strip().upper() != letter or recognized[0][1] < .90:
                    rejected['isolated_recognition'] += 1; continue
                isolated_confidence = float(recognized[0][1])
                cut.thumbnail((100,120))
                out = io.BytesIO(); cut.quantize(colors=128).save(out,format='PNG')
                low=np.maximum(quad.min(axis=0),[0,0]); high=np.minimum(quad.max(axis=0),[image.width,image.height])
                item = {'letter':letter,'image':'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode(),
                        'id':row['id'],'text':text,'estimated':True,'method':'ctc_character_alignment',
                        'confidence':confidence,'isolated_confidence':isolated_confidence,
                        'pixel_height':height,'sharpness':round(sharpness,2),'contrast':round(contrast,2),'crop':row['crop'],
                        'rect':[float(low[0]/image.width),float(low[1]/image.height),float((high[0]-low[0])/image.width),float((high[1]-low[1])/image.height)]}
                quality = isolated_confidence * min(height,120) * min(1., sharpness/300) * min(1., contrast/150)
                choices.setdefault(letter,[]).append((quality,item))
                choices[letter] = sorted(choices[letter], key=lambda pair:pair[0], reverse=True)[:80]
        if processed % 25 == 0:
            print(f'{processed} crops; {len(choices)} characters', flush=True)
    letters = []
    sources = {}
    for variants in choices.values():
        chosen_ids = set()
        for _, item in sorted(variants,key=lambda pair:pair[0],reverse=True):
            if item['id'] in chosen_ids:
                continue
            chosen_ids.add(item['id'])
            if item['id'] not in sources:
                detail=api('/api/object',id=item['id'],layers='store_sign,street_name_plate',confidence='.65')
                sources[item['id']]=detail.get('u','')
            item['source']=sources[item['id']]
            letters.append(item)
            if len(chosen_ids) >= args.variants:
                break
    output = args.output or args.site / 'alphabet-data.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_suffix('.json.tmp')
    temp.write_text(json.dumps({'version':1,'method':'RapidOCR Latin PP-OCRv5 character recognition with CTC-aligned boxes; editable, not human verified','processed_crops':processed,'selection':{'min_pixel_height':28,'min_contrast':45,'min_normalized_sharpness':35,'min_isolated_confidence':.90,'variants_per_character':args.variants,'rejected':dict(rejected)},'letters':letters},ensure_ascii=False))
    temp.replace(output)
    print(f'Wrote {len(letters)} variants, missing A-Z: {sorted(set(string.ascii_uppercase)-choices.keys())}',flush=True)


if __name__ == '__main__':
    main()
