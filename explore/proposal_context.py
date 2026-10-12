"""Serve cached JPEG context views for SAM proposals, decoding on first request.

Run behind the static-site proxy:
  .venv/bin/python explore/proposal_context.py --host 127.0.0.1 --port 8793
Only /proposal-context/<apple|gsv>/<12-hex-run>/<view>.jpg is served.
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
ROUTE = re.compile(r'/proposal-context/(apple|gsv)/([a-f0-9]{12})/([A-Za-z0-9_-]{1,160})\.jpg')


class ContextStore:
    def __init__(self, base, cache=None, max_side=1600, decoders=1):
        self.base = Path(base).resolve()
        self.cache = Path(cache or self.base / 'proposal_context_cache').resolve()
        self.max_side = max_side
        self.decoding = threading.BoundedSemaphore(decoders)

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
            self.send_header('Content-Type', 'image/jpeg')
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
