"""Show every retained graffiti proposal in selected review scenes."""
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from graffiti_review import ROOT, OUT, encode


def main():
    manifest = json.loads((OUT/'graffiti-review-samples.json').read_text())
    scenes = []
    for sample in manifest['samples']:
        path = ROOT/'data/coruna/apple/multicategory'/manifest['run']/'views'/sample['view_id']/'result.json'
        result = json.loads(path.read_text())
        all_rows = [p for p in result['predictions'] if p['category_proposal']=='graffiti']
        rows = [p for p in all_rows if not p.get('duplicate_of')]
        scenes.append((result, rows, len(all_rows)-len(rows)))
    # Include the first reviewed scene, then the most crowded, without repeats.
    selected = [scenes[0]]
    selected += sorted(scenes[1:], key=lambda s:-len(s[1]))[:11]
    sections, records = [], []
    colors = ['#ff3355','#00e5ff','#ffdd00','#55ff77','#ff88ff','#ff9933']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
    for result, rows, duplicate_count in selected:
        with Image.open(result['source_path']) as src:
            context = src.convert('RGB')
        original_width = context.width
        context.thumbnail((1600,1600))
        scale = context.width/original_width
        draw = ImageDraw.Draw(context)
        cards = []
        for i,p in enumerate(rows,1):
            color = colors[(i-1)%len(colors)]
            box = [v*scale for v in p['box']]
            draw.rectangle(box,outline=color,width=3)
            x,y=box[:2]
            label=str(i)
            bounds=draw.textbbox((x,y),label,font=font)
            draw.rectangle(bounds,fill='#111111')
            draw.text((x,y),label,fill=color,font=font)
            with Image.open(p['crop']) as im:
                crop=encode(im.convert('RGB'),(480,360))
            cards.append(f'<article><p style="color:{color}">#{i} · {html.escape(p["prompt"])} · score {p["score"]:.3f}</p><img loading="lazy" src="{crop}" alt="Graffiti proposal {i}"></article>')
        encoded=encode(context,(1600,1600))
        title=f'{result["view_id"]}: {len(rows)} proposals'
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{duplicate_count} overlapping proposals flagged as duplicates are omitted from this display; their saved records remain intact.</p><img class="scene" loading="lazy" src="{encoded}" alt="Street scene with every retained graffiti proposal numbered"><div class="crops">{"".join(cards)}</div></section>')
        records.append({'view_id':result['view_id'],'source_path':result['source_path'],'proposals':rows,'flagged_duplicates':duplicate_count})
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All graffiti in a scene</title><style>body{font:16px system-ui;background:#15191e;color:#eee;max-width:1300px;margin:auto;padding:20px}a{color:#9dcfff}p{line-height:1.5}h2{overflow-wrap:anywhere}section{border-top:1px solid #718195;margin-top:40px;padding-top:15px}.scene{max-width:100%;max-height:1000px;display:block;margin:auto}.crops{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}.crops img{width:100%;height:240px;object-fit:contain;background:#080b10}article{min-width:0}</style><h1>All graffiti detected in each scene</h1><p>12 scenes from the existing review: its first image, then 11 with the most proposals. Every unflagged graffiti or mural proposal is numbered, with its crop below. There is no one-object limit. Check for visible graffiti with no box, or several distinct pieces inside one box. Detection completeness has not been measured.</p><p>Current filters: SAM score ≥ 0.5; detection at least 32 pixels wide and tall. Overlapping same-category boxes at IoU ≥ 0.8 are flagged as duplicates. The model may produce overlapping boxes or group several marks together.</p><p><a href="graffiti-review.html">Individual crop review</a> · <a href="graffiti-scene-samples.json">Scene metadata</a></p>'''+''.join(sections)+'</html>'
    (OUT/'graffiti-scenes.html').write_text(page)
    (OUT/'graffiti-scene-samples.json').write_text(json.dumps({'run':manifest['run'],'scenes':records},indent=2))
    print(json.dumps({'scenes':len(selected),'displayed_proposals':sum(len(s[1]) for s in selected)}))


if __name__=='__main__':
    main()
