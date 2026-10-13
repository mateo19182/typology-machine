"""Build a small editable alphabet starter from OCR on existing sign crops.

Runs bounded CPU OCR, never detection or panorama decoding. Character boxes come from the recognizer's CTC alignment and remain editable.
Generated imagery stays in the ignored site data directory.
"""
import argparse
import base64
import io
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--api', default='http://localhost:8794')
    ap.add_argument('--site', type=Path, default=Path('data/coruna/site'))
    ap.add_argument('--limit', type=int, default=100)
    args = ap.parse_args()
    if not 1 <= args.limit <= 150:
        ap.error('--limit must be between 1 and 150')
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
    # Search less common letters first so the bounded sample has useful coverage.
    for query in ['k', 'w', 'x', 'j', 'ñ', 'z', 'farmacia', 'coruña', 'bar', '']:
        rows = api('/api/query', layers='store_sign,street_name_plate', confidence='.65', q=query, limit=16, map=0)['items']
        for row in rows:
            if row['id'] not in seen:
                seen.add(row['id']); candidates.append(row)
    choices = {}
    processed = 0
    for row in candidates[:args.limit]:
        path = args.site / row['crop'].lstrip('/')
        if not path.is_file():
            continue
        try:
            image = Image.open(path).convert('RGB')
            lines, _ = engine(str(path), return_word_box=True)
        except Exception as error:
            print(f'Skipped crop: {type(error).__name__}', flush=True)
            continue
        processed += 1
        for line in lines or []:
            text = line[1]
            for letter, polygon, confidence in character_regions(line):
                if letter not in string.ascii_uppercase + 'ÑÁÉÍÓÚÜ0123456789':
                    continue
                quad = np.asarray(polygon, dtype=np.float32)
                width = int(max(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[2]-quad[3])))
                height = int(max(np.linalg.norm(quad[3]-quad[0]), np.linalg.norm(quad[2]-quad[1])))
                if height < 16 or width < 5 or height / width > 12 or width / height > 2:
                    continue
                warp = cv2.getPerspectiveTransform(quad, np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32))
                cut = Image.fromarray(cv2.warpPerspective(np.asarray(image), warp, (width,height)))
                luminance = np.asarray(cut.convert('L'))
                if np.percentile(luminance, 95) - np.percentile(luminance, 5) < 20:
                    continue  # Empty or nearly uniform boxes are not useful letters.
                cut.thumbnail((100,120))
                out = io.BytesIO(); cut.save(out,format='PNG')
                low=np.maximum(quad.min(axis=0),[0,0]); high=np.minimum(quad.max(axis=0),[image.width,image.height])
                item = {'letter':letter,'image':'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode(),
                        'id':row['id'],'text':text,'estimated':True,'method':'ctc_character_alignment',
                        'confidence':confidence,'crop':row['crop'],
                        'rect':[float(low[0]/image.width),float(low[1]/image.height),float((high[0]-low[0])/image.width),float((high[1]-low[1])/image.height)]}
                quality = float(confidence) * min(height,120) / (1 + len(text)*.02)
                choices.setdefault(letter,[]).append((quality,item))
        if processed % 10 == 0:
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
            if len(chosen_ids) >= 3:
                break
    output = args.site / 'alphabet-data.json'
    temp = output.with_suffix('.json.tmp')
    temp.write_text(json.dumps({'version':1,'method':'RapidOCR Latin PP-OCRv5 character recognition with CTC-aligned boxes; editable, not human verified','processed_crops':processed,'letters':letters},ensure_ascii=False))
    temp.replace(output)
    print(f'Wrote {len(letters)} variants, missing A-Z: {sorted(set(string.ascii_uppercase)-choices.keys())}',flush=True)


if __name__ == '__main__':
    main()
