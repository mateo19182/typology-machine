"""Bounded, local AI generation worker behind the website's same-origin proxy."""
from __future__ import annotations
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import uuid
from graffiti_model import DEFAULT_HOME


def validate_request(data):
    if not isinstance(data, dict): raise ValueError('Expected a JSON object')
    prompt=data.get('prompt','')
    if not isinstance(prompt,str) or not 3<=len(prompt.strip())<=500: raise ValueError('Describe the image in 3–500 characters.')
    seed=data.get('seed',19182)
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=2147483647: raise ValueError('Invalid seed')
    strength=data.get('strength',.8)
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not 0<=strength<=1.2: raise ValueError('Invalid archive influence')
    return {'prompt':prompt.strip(),'seed':seed,'strength':strength}


class Jobs:
    def __init__(self,home):
        self.home=home;self.lock=threading.Lock();self.jobs={};self.active=None;self.last_started=0
        (home/'generated').mkdir(parents=True,exist_ok=True)
    def training(self):
        try: value=json.loads((self.home/'training.json').read_text())
        except (OSError,ValueError): return {'stage':'preparing'}
        return {key:value[key] for key in ('stage','step','steps','train_count','validation_count','updated') if key in value}
    def start(self,params):
        with self.lock:
            if self.training().get('stage')!='trained': return 503,{'error':'The archive model is still being prepared.'}
            if self.active: return 429,{'error':'The GPU is making another image. Try again shortly.'}
            if time.monotonic()-self.last_started<60: return 429,{'error':'Please wait a minute between images.'}
            job=uuid.uuid4().hex
            self.jobs[job]={'id':job,'stage':'running',**params,'created':time.time()};self.active=job;self.last_started=time.monotonic()
            threading.Thread(target=self.run,args=(job,params),daemon=True).start()
            return 202,dict(self.jobs[job])
    def run(self,job,params):
        output=self.home/'generated'/f'{job}.png'
        command=[sys.executable,str(Path(__file__).with_name('graffiti_model.py')),'generate','--home',str(self.home),'--prompt',params['prompt'],'--seed',str(params['seed']),'--strength',str(params['strength']),'--output',str(output)]
        try:
            with (self.home/'generation.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240,env={**os.environ,'OMP_NUM_THREADS':'2'})
            result={'stage':'complete','image':f'/api/graffiti/images/{job}.png'}
        except Exception:
            output.unlink(missing_ok=True);result={'stage':'failed','error':'Image generation failed. Try again later.'}
        with self.lock:
            self.jobs[job].update(result);self.active=None
            while len(self.jobs)>200: self.jobs.pop(next(iter(self.jobs)))
        # A fixed output budget prevents a public page from filling the disk.
        for old in sorted((self.home/'generated').glob('*.png'),key=lambda p:p.stat().st_mtime,reverse=True)[200:]: old.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    server_version='TypologyGraffiti/1'
    def json(self,status,value):
        raw=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        path=urlparse(self.path).path; jobs=self.server.jobs
        if path=='/api/graffiti/status': return self.json(200,{'training':jobs.training(),'busy':bool(jobs.active),'model':'Stable Diffusion 1.5 + Coruña LoRA','resolution':512,'steps':24})
        match=re.fullmatch(r'/api/graffiti/jobs/([a-f0-9]{32})',path)
        if match:
            with jobs.lock: value=dict(jobs.jobs.get(match[1],{}))
            return self.json(200,value) if value else self.json(404,{'error':'Image request expired.'})
        match=re.fullmatch(r'/api/graffiti/images/([a-f0-9]{32})\.png',path)
        if match:
            file=jobs.home/'generated'/f'{match[1]}.png'
            if not file.is_file(): return self.json(404,{'error':'Image unavailable.'})
            raw=file.read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Cache-Control','public,max-age=86400');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        self.json(404,{'error':'Not found'})
    def do_POST(self):
        if self.path!='/api/graffiti/generate': return self.json(404,{'error':'Not found'})
        origin=self.headers.get('Origin')
        if not origin or urlparse(origin).netloc!=self.headers.get('Host'): return self.json(403,{'error':'Open the generator on this website.'})
        if self.headers.get_content_type()!='application/json': return self.json(415,{'error':'Expected JSON'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096: raise ValueError('Request too large')
            self.connection.settimeout(10)
            params=validate_request(json.loads(self.rfile.read(length)))
        except (ValueError,TimeoutError): return self.json(400,{'error':'Invalid image request.'})
        status,value=self.server.jobs.start(params);self.json(status,value)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--home',type=Path,default=DEFAULT_HOME);parser.add_argument('--port',type=int,default=8797);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.jobs=Jobs(args.home);server.serve_forever()
if __name__=='__main__': main()
