"""Serve cached JPEG context views for SAM proposals, decoding on first request.

Run behind the static-site proxy:
  .venv/bin/python explore/proposal_context.py --host 127.0.0.1 --port 8793
Serves context JPEGs and crop geometry JSON under /proposal-context/.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import urlsplit

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parent.parent
GEOMETRY_ROUTE = re.compile(r'/proposal-context/(apple|gsv)/([a-f0-9]{12})/([A-Za-z0-9_-]{1,160})/([A-Za-z0-9_-]{1,160})\.json')
ROUTE = re.compile(r'/proposal-context/(apple|gsv)/([a-f0-9]{12})/([A-Za-z0-9_-]{1,160})\.jpg')


class ContextStore:
    def __init__(self, base, cache=None, max_side=1600, decoders=1):
        self.base = Path(base).resolve()
        self.cache = Path(cache or self.base / 'proposal_context_cache').resolve()
        self.max_side = max_side
        self.decoding = threading.BoundedSemaphore(decoders)

    def geometry(self, request_path):
        match = GEOMETRY_ROUTE.fullmatch(urlsplit(request_path).path)
        if not match:
            raise FileNotFoundError('Unknown geometry route')
        source, run, view, name = match.groups()
        run_root = (self.base / source / 'multicategory').resolve()
        # Published assets can point to a different contributing run. Read the
        # result beside the actual crop, not an unpadded detector box or alias.
        crop = (self.base / 'site/proposal-assets' / source / run / view / (name + '.jpg')).resolve()
        if not crop.is_relative_to(run_root) or not crop.is_file():
            raise FileNotFoundError('Crop unavailable')
        relative = crop.relative_to(run_root).parts
        if len(relative) != 4 or relative[1] != 'views' or not re.fullmatch(r'[a-f0-9]{12}', relative[0]):
            raise FileNotFoundError('Unexpected crop location')
        result = json.loads((crop.parent / 'result.json').read_text())
        if result.get('source') != source or result.get('view_id') != crop.parent.name:
            raise FileNotFoundError('Result identity does not match crop')
        prediction = next((p for p in result.get('predictions', []) if Path(p.get('crop', '')).name == crop.name), None)
        if prediction is None or not prediction.get('crop_box'):
            raise FileNotFoundError('Crop geometry unavailable')
        width, height = result['image_size']
        left, top, right, bottom = prediction['crop_box']
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise FileNotFoundError('Invalid crop geometry')
        return {'context_url': f'/proposal-context/{source}/{relative[0]}/{crop.parent.name}.jpg',
                'image_size': [width, height],
                'crop_box': [left / width, top / height, right / width, bottom / height]}

    def get(self, request_path):
        match = ROUTE.fullmatch(urlsplit(request_path).path)
        if not match:
            raise FileNotFoundError('Unknown context route')
        source, run, view = match.groups()
        run_root = self.base / source / 'multicategory'
        result_path = (run_root / run / 'views' / view / 'result.json').resolve()
        if not result_path.is_relative_to(run_root.resolve()):
            raise FileNotFoundError('Invalid result path')
        result = json.loads(result_path.read_text())
        if result.get('source') != source or result.get('view_id') != view:
            raise FileNotFoundError('Result identity does not match request')
        source_path = Path(result['source_path'])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        source_path = source_path.resolve()
        if not source_path.is_relative_to(self.base / source) or not source_path.is_file():
            raise FileNotFoundError('Context source unavailable')
        target = self.cache / source / run / f'{view}.jpg'
        if target.is_file():
            return target
        with self.decoding:
            if target.is_file():
                return target
            with Image.open(source_path) as image:
                image = image.convert('RGB')
                if source == 'gsv':
                    import numpy as np
                    import py360convert
                    config = json.loads((run_root / run / 'config.json').read_text())
                    projection = config['gsv_projection']
                    width, height = result['image_size']
                    # Preserve the detector's perspective projection, not the panorama.
                    scale = min(1, self.max_side / max(width, height))
                    image = Image.fromarray(py360convert.e2p(
                        np.asarray(image), fov_deg=tuple(projection['fov']),
                        u_deg=result['metadata']['view_yaw'], v_deg=result['metadata']['view_pitch'],
                        out_hw=(round(height * scale), round(width * scale)), mode='bilinear'))
                elif list(image.size) != result['image_size']:
                    raise ValueError('Context dimensions do not match detection geometry')
                image.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix='.context-', suffix='.jpg', dir=target.parent)
                os.close(fd)
                try:
                    image.save(temporary, format='JPEG', quality=85, optimize=True)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        return target


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.serve_context()

        def do_HEAD(self):
            self.serve_context(head=True)

        def serve_context(self, head=False):
            try:
                geometry = GEOMETRY_ROUTE.fullmatch(urlsplit(self.path).path)
                if geometry:
                    content = json.dumps(store.geometry(self.path), separators=(',', ':')).encode()
                else:
                    target = store.get(self.path)
                    content = target.read_bytes()
            except FileNotFoundError:
                self.send_error(404, 'Context unavailable')
                return
            except Exception as error:
                self.log_error('Context decode failed: %s', type(error).__name__)
                self.send_error(500, 'Context could not be decoded')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if geometry else 'image/jpeg')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'public, max-age=604800, immutable')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            if not head:
                self.wfile.write(content)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'data/coruna')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8793)
    parser.add_argument('--max-side', type=int, default=1600)
    parser.add_argument('--decoders', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    store = ContextStore(args.base, args.cache, args.max_side, args.decoders)
    print(f'Context JPEG server listening on {args.host}:{args.port}', flush=True)
    ThreadingHTTPServer((args.host, args.port), handler_for(store)).serve_forever()


if __name__ == '__main__':
    main()
