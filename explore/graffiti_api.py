"""Bounded, local AI generation worker behind the website's same-origin proxy."""
from __future__ import annotations
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import uuid
from graffiti_model import DEFAULT_HOME, model_settings, write_json
from graffiti_worker import PersistentWorker


def validate_request(data):
    if not isinstance(data, dict): raise ValueError('Expected a JSON object')
    prompt=data.get('prompt','')
    if not isinstance(prompt,str) or not 3<=len(prompt.strip())<=500: raise ValueError('Describe the image in 3–500 characters.')
    seed=data.get('seed',19182)
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=2147483647: raise ValueError('Invalid seed')
    strength=data.get('strength',1.2)
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not 0<=strength<=1.2: raise ValueError('Invalid archive influence')
    return {'prompt':prompt.strip(),'seed':seed,'strength':strength}


class Jobs:
    def __init__(self,home):
        self.home=home;self.lock=threading.Lock();self.jobs={};self.active=None
        (home/'generated').mkdir(parents=True,exist_ok=True)
        self.settings=model_settings(home)
        self.worker=PersistentWorker(home,self.settings) if self.settings.get('worker_mode')=='persistent' else None
        self.timing_settings={key:self.settings.get(key) for key in ('model_id','revision','steps','max_sequence_length','worker_mode')}
        self.timings=deque(maxlen=50)
        try:
            history=json.loads((home/'generation-timings.json').read_text())
            if history.get('settings')==self.timing_settings:
                for seconds in history.get('seconds',[])[-50:]:
                    if isinstance(seconds,(int,float)) and not isinstance(seconds,bool) and math.isfinite(seconds) and 0<seconds<=600:
                        self.timings.append(seconds)
        except (OSError,ValueError,TypeError,AttributeError): pass
    def preload(self):
        if self.worker is None or self.training().get('stage')!='trained' or os.environ.get('TYPOLOGY_GRAFFITI_PAUSED')=='1': return
        def warm():
            try: self.worker.warm()
            except Exception: print('Could not preload the graffiti model; the next request will retry.',file=sys.stderr,flush=True)
        threading.Thread(target=warm,daemon=True).start()
    def close(self):
        if self.worker is not None: self.worker.close()
    def timing(self):
        values=list(self.timings)
        return {'sample_count':len(values),'typical_seconds':round(statistics.median(values),1) if values else 40,
                'min_seconds':round(min(values),1) if values else None,'max_seconds':round(max(values),1) if values else None}
    def library(self):
        with self.lock: active=self.active
        saved=[]
        for file in (self.home/'generated').glob('*.png'):
            if not re.fullmatch(r'[a-f0-9]{32}',file.stem) or file.stem==active: continue
            try: saved.append((file.stat().st_mtime,file))
            except OSError: continue
        images=[]
        for created,file in sorted(saved,reverse=True)[:200]:
            item={'id':file.stem,'image':f'/api/graffiti/images/{file.stem}.png',
                  'thumbnail':f'/api/graffiti/thumbnails/{file.stem}.webp','created':created}
            try:
                metadata=json.loads(file.with_suffix('.json').read_text())
                if not isinstance(metadata,dict): raise ValueError('Invalid image metadata')
                if all(key in metadata for key in ('prompt','seed','strength')):
                    item.update(validate_request(metadata))
                inference=metadata.get('inference',{})
                if (isinstance(inference,dict) and inference.get('model_id') in ('black-forest-labs/FLUX.2-klein-base-4B','black-forest-labs/FLUX.2-klein-4B')
                    and isinstance(inference.get('revision'),str) and re.fullmatch(r'[a-f0-9]{40}',inference['revision'])
                    and type(inference.get('steps')) is int and inference['steps'] in (4,30)):
                    item['inference']={key:inference[key] for key in ('model_id','revision','steps')}
            except (OSError,ValueError,TypeError): pass
            images.append(item)
        return {'images':images}
    def thumbnail(self,job):
        from PIL import Image
        with self.lock:
            source=self.home/'generated'/f'{job}.png'
            if not source.is_file() or job==self.active: return None
            cached=source.with_suffix('.thumbnail.webp')
            try:
                if cached.is_file(): return cached.read_bytes()
                with Image.open(source) as image:
                    image=image.convert('RGB');image.thumbnail((256,256),Image.Resampling.LANCZOS)
                    buffer=io.BytesIO();image.save(buffer,format='WEBP',quality=80)
                raw=buffer.getvalue()
                temp=cached.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(cached)
                return raw
            except OSError: return None
    def training(self):
        try: value=json.loads((self.home/'training.json').read_text())
        except (OSError,ValueError): return {'stage':'preparing'}
        result={key:value[key] for key in ('stage','step','steps','train_count','validation_count','updated') if key in value}
        if value.get('stage')=='training':
            try: progress=json.loads((self.home/'progress.json').read_text())
            except (OSError,ValueError): progress={}
            step=progress.get('step')
            if isinstance(step,int) and not isinstance(step,bool) and 0<=step<=value.get('steps',0): result['step']=step
        return result
    def status(self):
        with self.lock:
            return {'training':self.training(),'busy':bool(self.active),'paused':os.environ.get('TYPOLOGY_GRAFFITI_PAUSED')=='1',
                    'generation_timing':self.timing(),**self.settings,
                    'inference_worker':self.worker.status() if self.worker is not None else {'stage':'on-demand'}}
    def get(self,job):
        with self.lock:
            value=dict(self.jobs.get(job,{}))
        if not value: return value
        total=self.settings['steps']
        progress={'phase':'loading','completed_steps':0,'total_steps':total}
        try:
            reported=json.loads((self.home/'generated'/f'{job}.progress.json').read_text())
            completed=reported.get('completed_steps')
            if reported.get('phase') in ('loading','encoding','generating','finishing') and isinstance(completed,int) and not isinstance(completed,bool) and 0<=completed<=total:
                progress.update(phase=reported['phase'],completed_steps=completed)
        except (OSError,ValueError,AttributeError): pass
        if value['stage']=='complete': progress.update(phase='complete',completed_steps=total)
        if value['stage']=='failed': progress['phase']='failed'
        value['progress']=progress
        value['elapsed_seconds']=round(value.get('duration_seconds',max(0,time.time()-value['created'])),1)
        return value
    def start(self,params):
        with self.lock:
            if os.environ.get('TYPOLOGY_GRAFFITI_PAUSED')=='1': return 503,{'error':'Training and evaluation are in progress. Try again later.'}
            if self.training().get('stage')!='trained': return 503,{'error':'The archive model is still being prepared.'}
            if self.active: return 429,{'error':'The GPU is making another image. Try again shortly.'}
            job=uuid.uuid4().hex
            self.jobs[job]={'id':job,'stage':'running',**params,'created':time.time(),
                            'estimated_seconds':self.timing()['typical_seconds']};self.active=job
            threading.Thread(target=self.run,args=(job,params),daemon=True).start()
            return 202,dict(self.jobs[job])
    def run(self,job,params):
        started=time.monotonic()
        output=self.home/'generated'/f'{job}.png'
        command=[sys.executable,str(Path(__file__).with_name('graffiti_model.py')),'generate','--home',str(self.home),'--prompt',params['prompt'],'--seed',str(params['seed']),'--strength',str(params['strength']),'--output',str(output),'--progress',str(output.with_suffix('.progress.json'))]
        try:
            if self.worker is not None:
                self.worker.generate(job,params,timeout=600)
            else:
                with (self.home/'generation.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,env={**os.environ,'OMP_NUM_THREADS':'2'})
            inference={key:self.settings[key] for key in ('model_id','revision','steps','guidance_scale','max_sequence_length') if key in self.settings}
            try: write_json(output.with_suffix('.json'),{'id':job,**params,'created':self.jobs[job]['created'],'inference':inference})
            except OSError: pass
            result={'stage':'complete','image':f'/api/graffiti/images/{job}.png'}
        except Exception:
            output.unlink(missing_ok=True);result={'stage':'failed','error':'Image generation failed. Try again later.'}
        with self.lock:
            duration=time.monotonic()-started
            self.jobs[job].update(result,duration_seconds=round(duration,3));self.active=None
            if result['stage']=='complete':
                self.timings.append(duration)
                try: write_json(self.home/'generation-timings.json',{'settings':self.timing_settings,'seconds':list(self.timings)})
                except OSError: pass
            while len(self.jobs)>200: self.jobs.pop(next(iter(self.jobs)))
        # A fixed output budget prevents a public page from filling the disk.
        for old in sorted((self.home/'generated').glob('*.png'),key=lambda p:p.stat().st_mtime,reverse=True)[200:]:
            old.unlink(missing_ok=True);old.with_suffix('.progress.json').unlink(missing_ok=True)
            old.with_suffix('.json').unlink(missing_ok=True);old.with_suffix('.thumbnail.webp').unlink(missing_ok=True)
        if result['stage']=='failed': output.with_suffix('.progress.json').unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    server_version='TypologyGraffiti/1'
    def json(self,status,value):
        raw=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        path=urlparse(self.path).path; jobs=self.server.jobs
        if path=='/api/graffiti/status': return self.json(200,jobs.status())
        if path=='/api/graffiti/library': return self.json(200,jobs.library())
        match=re.fullmatch(r'/api/graffiti/jobs/([a-f0-9]{32})',path)
        if match:
            value=jobs.get(match[1])
            return self.json(200,value) if value else self.json(404,{'error':'Image request expired.'})
        match=re.fullmatch(r'/api/graffiti/images/([a-f0-9]{32})\.png',path)
        if match:
            file=jobs.home/'generated'/f'{match[1]}.png'
            if not file.is_file(): return self.json(404,{'error':'Image unavailable.'})
            raw=file.read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Cache-Control','public,max-age=86400');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        match=re.fullmatch(r'/api/graffiti/thumbnails/([a-f0-9]{32})\.webp',path)
        if match:
            raw=jobs.thumbnail(match[1])
            if raw is None: return self.json(404,{'error':'Image unavailable.'})
            self.send_response(200);self.send_header('Content-Type','image/webp');self.send_header('Cache-Control','public,max-age=86400');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
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
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.jobs=Jobs(args.home);server.jobs.preload()
    try: server.serve_forever()
    finally: server.jobs.close();server.server_close()
if __name__=='__main__': main()
