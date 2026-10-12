"""Incremental CPU OCR for completed SAM street-name and shop-sign crops.

Writes append-only ocr.jsonl in the run; preserves every input crop. Run repeatedly
with --limit to process bounded batches. Existing Florence OCR is reused by ID.
Recognition checkpoint (save under data/models/ocr/):
https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx
The SHA256 is verified before inference; detection uses bundled PP-OCRv4.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import hashlib
import json
from pathlib import Path
import time


def read_records(path):
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted final append is retried.


def select_pending(run, done, categories, limit):
    queues = defaultdict(list)
    for path in sorted((run / 'views').glob('*/result.json')):
        result = json.loads(path.read_text())
        for p in result['predictions']:
            category = p['category_proposal']
            if category in categories and p['id'] not in done and not p.get('duplicate_of'):
                queues[category].append((p, result['source'], result['view_id']))
    # Round robin prevents the much larger shop-sign class starving street names.
    selected = []
    index = 0
    while len(selected) < limit:
        progressed = False
        for category in categories:
            if index < len(queues[category]):
                selected.append(queues[category][index])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
        index += 1
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--categories', nargs='+', default=['street_name_plate', 'store_sign'])
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / '.ocr.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('OCR worker already running', flush=True)
            return
        out = run / 'ocr.jsonl'
        # Repair only an incomplete trailing record so future appends stay readable.
        if out.exists():
            raw = out.read_bytes()
            if raw and not raw.endswith(b'\n'):
                try:
                    json.loads(raw.rsplit(b'\n', 1)[-1])
                    with out.open('ab') as stream:
                        stream.write(b'\n')
                except json.JSONDecodeError:
                    with out.open('r+b') as stream:
                        stream.truncate(raw.rfind(b'\n') + 1)
        done = {r['id'] for r in read_records(out)}
        todo = select_pending(run, done, args.categories, args.limit)
        legacy = {r['id']: r['text'] for r in read_records(run.parent.parent / 'store_sign' / 'ocr.jsonl')}
        print(f'{len(todo)} pending selected; {len(done)} completed previously', flush=True)
        if not todo:
            return
        from rapidocr_onnxruntime import RapidOCR
        rec_model=Path(__file__).resolve().parent.parent / 'data/models/ocr/latin_PP-OCRv5_rec_mobile.onnx'
        digest=hashlib.sha256(rec_model.read_bytes()).hexdigest()
        if digest != 'b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a':
            raise ValueError('Latin OCR checkpoint checksum mismatch')
        engine = RapidOCR(rec_model_path=str(rec_model), intra_op_num_threads=2, inter_op_num_threads=1, det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False)
        model = 'rapidocr_onnxruntime/' + importlib.metadata.version('rapidocr_onnxruntime') + '/latin_PP-OCRv5_rec_mobile'
        model_sha256=digest
        started = time.monotonic()
        with out.open('a') as stream:
            for index, (p, source, view) in enumerate(todo, 1):
                record = {'id': p['id'], 'category': p['category_proposal'], 'source': source, 'view_id': view,
                          'processed_at': datetime.now(timezone.utc).isoformat()}
                old_id = p['id'].split(':', 2)[-1] if p['id'].startswith('legacy:') else None
                if old_id in legacy:
                    record.update(text=legacy[old_id], model='florence-community/Florence-2-large', confidence=None, reused_from='store_sign/ocr.jsonl')
                else:
                    try:
                        result, _ = engine(str(p['crop']))
                    except Exception as error:
                        print(f"Failed {p['id']}: {error}", flush=True)
                        continue
                    lines = result or []
                    record['model_sha256']=model_sha256
                    record.update(text='\n'.join(line[1] for line in lines).strip(), model=model,
                                  confidence=round(sum(float(line[2]) for line in lines) / len(lines), 4) if lines else None)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                if index % 20 == 0 or index == len(todo):
                    print(f'{index}/{len(todo)} crops in {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
