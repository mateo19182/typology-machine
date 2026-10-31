"""One reusable GPU pipeline in a bounded, restartable child process.

Importing the client does not import Torch. The child reserves stdout for its
JSON protocol and sends model output and tracebacks to the service's log.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import threading
import time
import traceback


class PersistentWorker:
    def __init__(self, home, settings, command=None):
        self.home = home
        self.settings = settings
        self.command = command or [sys.executable, str(Path(__file__).resolve()), '--home', str(home)]
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.process = None
        self.buffer = b''
        self.stage = 'stopped'
        self.loads = 0
        self.generations = 0

    def status(self):
        with self.state_lock:
            process = self.process
            stage = 'stopped' if self.stage == 'ready' and (process is None or process.poll() is not None) else self.stage
            return {'stage': stage, 'loads': self.loads, 'generations': self.generations}

    def set_stage(self, stage):
        with self.state_lock:
            self.stage = stage

    def read_event(self, deadline):
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('GPU worker timed out')
            if not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('GPU worker timed out')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('GPU worker exited before completing its request')
            self.buffer += chunk
            if len(self.buffer) > 65536:
                raise ValueError('GPU worker returned an invalid response')
        line, self.buffer = self.buffer.split(b'\n', 1)
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError('GPU worker returned an invalid response')
        return event

    def stop_locked(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
        self.buffer = b''
        self.set_stage('stopped')

    def load_locked(self, deadline):
        if self.process is not None and self.process.poll() is None and self.stage == 'ready':
            return
        self.stop_locked()
        self.set_stage('loading')
        with (self.home / 'generation.log').open('a') as log:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=log, bufsize=0,
                                            env={**os.environ, 'OMP_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'})
        event = self.read_event(deadline)
        if event.get('event') != 'ready' or any(event.get(key) != self.settings[key] for key in ('model_id', 'revision', 'steps')):
            raise RuntimeError('GPU worker could not load the selected inference model')
        with self.state_lock:
            self.stage = 'ready'
            self.loads += 1

    def warm(self, timeout=600):
        with self.lock:
            try:
                self.load_locked(time.monotonic() + timeout)
            except Exception:
                self.stop_locked()
                raise

    def generate(self, job, params, timeout=600):
        with self.lock:
            try:
                deadline = time.monotonic() + timeout
                self.load_locked(deadline)
                payload = json.dumps({'id': job, **params}).encode() + b'\n'
                self.process.stdin.write(payload)
                event = self.read_event(deadline)
                if event.get('event') != 'complete' or event.get('id') != job:
                    raise RuntimeError('GPU worker failed its image request')
                if not (self.home / 'generated' / f'{job}.png').is_file():
                    raise RuntimeError('GPU worker did not save its image')
                with self.state_lock:
                    self.generations += 1
            except Exception:
                self.stop_locked()
                raise

    def close(self):
        with self.lock:
            self.stop_locked()


def serve(home):
    from graffiti_model import generate_flux, load_flux_pipeline, model_settings
    protocol = sys.stdout

    def send(event):
        print(json.dumps(event), file=protocol, flush=True)

    with redirect_stdout(sys.stderr):
        try:
            settings = model_settings(home)
            pipe = load_flux_pipeline(home, model_id=settings['model_id'], revision=settings['revision'])
            send({'event': 'ready', **{key: settings[key] for key in ('model_id', 'revision', 'steps')}})
        except Exception:
            traceback.print_exc()
            send({'event': 'failed'})
            return 1
        for line in sys.stdin:
            try:
                request = json.loads(line)
                job = request['id']
                if not isinstance(job, str) or not re.fullmatch(r'[a-f0-9]{32}', job):
                    raise ValueError('Invalid worker request ID')
                output = home / 'generated' / f'{job}.png'
                generate_flux(home, request['prompt'], output, request['seed'], request['strength'],
                              progress=output.with_suffix('.progress.json'), pipe=pipe, settings=settings)
                send({'event': 'complete', 'id': job})
            except Exception:
                traceback.print_exc()
                send({'event': 'failed'})
                # Recreate the pipeline after an error rather than reuse a damaged CUDA context.
                return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(serve(args.home))


if __name__ == '__main__':
    main()
