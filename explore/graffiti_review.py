"""Build a reproducible graffiti proposal review from completed SAM views."""
import base64
import io
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw
import pillow_heif
pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'data/coruna/apple/multicategory/765844a02f75'
OUT = ROOT / 'data/coruna/comparison_share'


def encode(im, size):
    im.thumbnail(size)
    b = io.BytesIO()
    im.save(b, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


def main():
    candidates = []
    files = sorted((RUN / 'views').glob('*/result.json'))
    for path in files:
        result = json.loads(path.read_text())
        for p in result['predictions']:
            if p['category_proposal'] == 'graffiti' and not p.get('duplicate_of'):
                candidates.append((p, result))
    rng = random.Random(42)
    chosen, seen = [], set()
    for prompt in ['graffiti', 'mural']:
        rows = [r for r in candidates if r[0]['prompt'] == prompt]
        shuffled = rng.sample(rows, len(rows))
        for label, ordered in [('random', shuffled), ('lowest score', sorted(rows, key=lambda r:r[0]['score'])), ('highest score', sorted(rows, key=lambda r:-r[0]['score']))]:
            n = 0
            for p, r in ordered:
                pano = r['metadata']['id']
                if pano in seen:
                    continue
                seen.add(pano)
                chosen.append((p, r, label))
                n += 1
                if n == 10:
                    break
    cases, manifest = [], []
    for p, r, sample in chosen:
        with Image.open(p['crop']) as im:
            crop = encode(im.convert('RGB'), (900, 650))
        with Image.open(r['source_path']) as im:
            context = im.convert('RGB')
            draw = ImageDraw.Draw(context)
            draw.rectangle(p['box'], outline='#ff3355', width=max(6, context.width//200))
            context = encode(context, (1000, 1000))
        record = {**p, 'view_id':r['view_id'], 'metadata':r['metadata'], 'source_path':r['source_path'], 'sample':sample}
        manifest.append(record)
        cases.append({**record, 'crop_image':crop, 'context_image':context})
    data = {'run':RUN.name, 'completed_views':len(files), 'candidate_count':len(candidates), 'cases':cases}
    template = Path(__file__).with_name('graffiti_review.html').read_text()
    OUT.mkdir(exist_ok=True)
    (OUT/'graffiti-review.html').write_text(template.replace('__DATA__', json.dumps(data).replace('</','<\\/')))
    (OUT/'graffiti-review-samples.json').write_text(json.dumps({'run':RUN.name,'completed_views':len(files),'candidate_count':len(candidates),'samples':manifest}, indent=2))
    print(json.dumps({'cases':len(cases),'candidates':len(candidates),'completed_views':len(files)}))


if __name__ == '__main__':
    main()
