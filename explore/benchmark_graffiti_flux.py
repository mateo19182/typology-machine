"""Compare the deployed LoRA on pinned base and distilled FLUX.2 Klein models.

Run with the serving Python environment after caching both snapshots. The public
generator is paused during GPU work and restored in a finally block. Henad stays
running. Fresh-process timings match the website's loading on every request;
separate reused-pipeline timings show what a persistent worker might achieve.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from html import escape
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

from graffiti_model import FLUX_MODEL, load_flux_pipeline, model_settings, write_json

DISTILLED_MODEL = 'black-forest-labs/FLUX.2-klein-4B'
DISTILLED_REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'
CASES = [
    {'name': 'shutter', 'seed': 19182, 'strength': 1.2,
     'prompt': 'A close-up photograph of bold red and white bubble-letter graffiti with black outlines on a weathered corrugated metal shutter, layered paint and faded tags.'},
    {'name': 'concrete', 'seed': 19200, 'strength': 1.2,
     'prompt': 'A photograph of blue and orange graffiti lettering on a weathered concrete wall, layers of faded tags and chipped paint.'},
    {'name': 'bird', 'seed': 19184, 'strength': 1.2,
     'prompt': 'A colorful mural of a bird on a weathered concrete wall, daylight'},
    {'name': 'wall', 'seed': 19185, 'strength': 1.2, 'prompt': 'wall'},
    {'name': 'hello', 'seed': 19186, 'strength': 1.2,
     'prompt': 'graffiti lettering that says "HELLO" on a concrete wall'},
    {'name': 'shutter-no-lora', 'seed': 19182, 'strength': 0,
     'prompt': 'A close-up photograph of bold red and white bubble-letter graffiti with black outlines on a weathered corrugated metal shutter, layered paint and faded tags.'},
]


def profiles(home):
    state = json.loads((home / 'training.json').read_text())
    if state.get('stage') != 'trained' or state.get('model') != FLUX_MODEL:
        raise ValueError('Choose a completed FLUX.2 Klein Base 4B training run')
    return {
        'current': {'model_id': FLUX_MODEL, 'revision': state['revision'], 'steps': 30, 'guidance_scale': 4.0},
        'fast': {'model_id': DISTILLED_MODEL, 'revision': DISTILLED_REVISION, 'steps': 4, 'guidance_scale': 1.0},
    }


def worker(specification):
    spec = json.loads(specification.read_text())
    home, output = Path(spec['home']), Path(spec['output'])
    profile = spec['profile']
    imported = time.monotonic()
    import torch
    torch.cuda.synchronize()
    loading = time.monotonic()
    pipe = load_flux_pipeline(home, model_id=profile['model_id'], revision=profile['revision'])
    torch.cuda.synchronize()
    loaded = time.monotonic()
    if bool(pipe.config.is_distilled) != (profile['model_id'] == DISTILLED_MODEL):
        raise ValueError('Pipeline distillation setting does not match the benchmark profile')
    encoder_times = []
    encode_prompt = pipe.encode_prompt

    def timed_encode(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.monotonic()
        value = encode_prompt(*args, **kwargs)
        torch.cuda.synchronize()
        encoder_times.append(time.monotonic() - start)
        return value

    pipe.encode_prompt = timed_encode
    results = []
    for index, case in enumerate(spec['cases']):
        pipe.set_adapters(['coruna'], adapter_weights=[case['strength']])
        torch.cuda.synchronize()
        start = time.monotonic()
        encoder_times.clear()
        steps = []

        def step_end(pipeline, step, timestep, values):
            steps.append(time.monotonic())
            return values

        image = pipe(prompt=case['prompt'], height=512, width=512,
                     num_inference_steps=profile['steps'], guidance_scale=profile['guidance_scale'],
                     max_sequence_length=model_settings(home)['max_sequence_length'],
                     callback_on_step_end=step_end, callback_on_step_end_tensor_inputs=[],
                     generator=torch.Generator('cpu').manual_seed(case['seed'])).images[0]
        file = output / f"{spec['name']}-{index}-{case['name']}.png"
        image.save(file)
        finished = time.monotonic()
        if len(steps) != profile['steps'] or image.size != (512, 512):
            raise ValueError('Unexpected sampling steps or image dimensions')
        result = {**case, 'file': file.name, 'generation_seconds': round(finished - start, 3),
                  'encoder_seconds': round(sum(encoder_times), 3),
                  'first_image_latency_seconds': round(finished - spec['started_monotonic'], 3) if index == 0 else None,
                  'completed_steps': len(steps), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
        results.append(result)
        print(json.dumps({'event': 'image', 'job': spec['name'], **result}), flush=True)
    write_json(output / f"{spec['name']}.json", {
        'profile': profile, 'load_seconds': round(loaded - loading, 3),
        'torch_import_seconds': round(loading - imported, 3), 'images': results,
        'cuda_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    })


def service(command):
    subprocess.run(['systemctl', '--user', *command], check=True, capture_output=True, text=True)


def generator_status():
    deadline = time.monotonic() + 10
    while True:
        try:
            return json.loads(urlopen('http://127.0.0.1:8797/api/graffiti/status', timeout=2).read())
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.1)


@contextmanager
def pause_generator():
    status = generator_status()
    if status['busy'] or status['paused']:
        raise RuntimeError('Wait for the current website generation or maintenance to finish')
    dropin = Path.home() / '.config/systemd/user/typology-graffiti.service.d/90-inference-benchmark.conf'
    dropin.parent.mkdir(parents=True, exist_ok=True)
    with dropin.open('x') as file:
        file.write('[Service]\nEnvironment=TYPOLOGY_GRAFFITI_PAUSED=1\n')
    try:
        service(['daemon-reload'])
        service(['restart', 'typology-graffiti.service'])
        status = generator_status()
        if not status['paused']:
            raise RuntimeError('Could not pause public generation')
        yield
    finally:
        dropin.unlink(missing_ok=True)
        service(['daemon-reload'])
        service(['restart', 'typology-graffiti.service'])


def henad_status():
    return subprocess.check_output(['docker', 'inspect', '--format',
                                   '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}',
                                   'henad_inference'], text=True).strip()


def monitor_gpu(stopped, samples):
    while not stopped.is_set():
        try:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True, timeout=10)
            memory, utilization = [int(value.strip()) for value in raw.split(',')]
            samples.append({'at': time.time(), 'memory_used_mib': memory, 'utilization_percent': utilization})
        except (subprocess.SubprocessError, ValueError):
            pass
        stopped.wait(1)


def comparisons(output, jobs):
    from PIL import Image, ImageChops, ImageDraw, ImageStat
    selected = {}
    for job in jobs:
        for image in job['images']:
            selected.setdefault((job['key'], image['name']), image)
    for filename, names in [('comparison-graffiti.jpg', ['shutter', 'concrete', 'bird']),
                            ('comparison-checks.jpg', ['wall', 'hello', 'shutter-no-lora'])]:
        sheet = Image.new('RGB', (1024, 570 * len(names)), '#151518')
        draw = ImageDraw.Draw(sheet)
        for row, name in enumerate(names):
            for column, key in enumerate(['current', 'fast']):
                item = selected[key, name]
                with Image.open(output / item['file']) as image:
                    sheet.paste(image, (column * 512, row * 570))
                draw.text((column * 512 + 10, row * 570 + 523),
                          f"{name} | {'Base 30 steps' if key == 'current' else 'Distilled 4 steps'} | LoRA {item['strength']}", fill='#d8d3c8')
        sheet.save(output / filename, quality=94)
    differences = {}
    for key in ['current', 'fast']:
        with Image.open(output / selected[key, 'shutter']['file']) as adapted, Image.open(output / selected[key, 'shutter-no-lora']['file']) as base:
            differences[key] = round(sum(ImageStat.Stat(ImageChops.difference(adapted, base)).mean) / 3, 3)
    return differences


def review_page(output, report):
    selected = {}
    for job in report['jobs']:
        for image in job['images']:
            selected.setdefault((job['key'], image['name']), image)
    rows = []
    for case in CASES:
        name = case['name']
        images = []
        for key, label in [('current', 'Base · 30 steps'), ('fast', 'Distilled · 4 steps')]:
            item = selected[key, name]
            file = escape(item['file'], quote=True)
            images.append(f'<figure><figcaption>{label}</figcaption><a href="{file}" target="_blank" rel="noopener"><img src="{file}" width="512" height="512" loading="lazy" alt="{escape(name)} using {key} inference"></a></figure>')
        rows.append(f'<section><h2>{escape(name.replace("-", " "))}</h2><p class="settings">Seed {case["seed"]} · archive influence {case["strength"]}</p><details><summary>Prompt</summary><p>{escape(case["prompt"])}</p></details><div class="pair">{"".join(images)}</div></section>')
    current, fast = report['summary']['current'], report['summary']['fast']
    fresh_speedup = current['fresh_process_median_seconds'] / fast['fresh_process_median_seconds']
    warm_speedup = current['reused_pipeline_median_seconds'] / fast['reused_pipeline_median_seconds']
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference comparison · Typology Machine</title><link rel="icon" href="../favicon.svg" type="image/svg+xml">
<style>
:root{{color-scheme:dark;font-family:"SFMono-Regular",Consolas,monospace;background:#0a0b0d;color:#d8d3c8}}*{{box-sizing:border-box}}body{{margin:0;font-size:13px;line-height:1.6}}header{{padding:16px;border-bottom:1px solid #292a2e}}a{{color:inherit}}header a{{text-decoration:none}}main{{max-width:1100px;margin:auto;padding:32px 20px}}h1{{font-size:28px;font-weight:400;margin:0 0 8px}}h2{{font-size:15px;font-weight:400;margin:0}}.muted,.settings,details{{color:#99948a}}.settings{{font-size:11px;margin:4px 0 10px}}table{{width:100%;border-collapse:collapse;margin:24px 0}}th,td{{text-align:left;padding:10px 8px;border-bottom:1px solid #292a2e}}th{{font-weight:400;color:#99948a}}section{{border-top:1px solid #292a2e;padding-top:20px;margin-top:34px}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}}figure{{margin:0;min-width:0}}figcaption{{font-size:11px;color:#99948a;margin-bottom:7px}}img{{width:100%;height:auto;display:block;border:1px solid #292a2e}}details p{{max-width:700px}}summary{{cursor:pointer}}a:focus-visible,summary:focus-visible{{outline:2px solid #d79a3a;outline-offset:3px}}@media(max-width:600px){{main{{padding:24px 14px}}table{{font-size:11px}}th,td{{padding:8px 4px}}.pair{{grid-template-columns:1fr;gap:18px}}}}
</style></head><body><header><a href="../graffiti.html">← Graffiti</a></header><main>
<h1>Inference comparison</h1><p class="muted">The same Coruña LoRA, prompts and seeds. Base 4B versus distilled Klein 4B.</p>
<table><thead><tr><th>Median time per image</th><th>Base</th><th>Distilled</th><th>Speedup</th></tr></thead><tbody>
<tr><td>Including model loading</td><td>{current['fresh_process_median_seconds']:.1f}s</td><td>{fast['fresh_process_median_seconds']:.1f}s</td><td>{fresh_speedup:.1f}×</td></tr>
<tr><td>With the pipeline already loaded</td><td>{current['reused_pipeline_median_seconds']:.1f}s</td><td>{fast['reused_pipeline_median_seconds']:.1f}s</td><td>{warm_speedup:.1f}×</td></tr>
</tbody></table>
<p class="muted">Three fresh requests and four requests with a reused pipeline per variant, on the RTX 3090 with Henad running. Images are 512 × 512. Fresh-request timings include process startup and model loading; reused-pipeline timings exclude loading. These examples are a small visual check; click an image to inspect it.</p>
{''.join(rows)}
</main></body></html>'''
    (output / 'index.html').write_text(page)


def run(home, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise ValueError('Choose a new output directory; this benchmark already completed')
    configurations = profiles(home)
    jobs, samples = [], []
    before = henad_status()
    if before != 'running healthy':
        raise RuntimeError('Benchmark requires the usual Henad GPU service to be healthy')
    plan = [('current', 'cold-shutter', [0]), ('fast', 'cold-shutter', [0]),
            ('fast', 'cold-concrete', [1]), ('current', 'cold-concrete', [1]),
            ('current', 'cold-bird', [2]), ('fast', 'cold-bird', [2]),
            ('fast', 'reuse', [0, 0, 3, 4, 5]), ('current', 'reuse', [0, 0, 3, 4, 5])]
    stop = threading.Event()
    telemetry = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
    telemetry.start()
    try:
        with pause_generator():
            for key, mode, indexes in plan:
                name = f'{key}-{mode}'
                specification = output / f'{name}.request.json'
                value = {'name': name, 'home': str(home), 'output': str(output),
                         'profile': configurations[key], 'cases': [CASES[index] for index in indexes],
                         'started_monotonic': time.monotonic()}
                write_json(specification, value)
                print(json.dumps({'event': 'start', 'job': name}), flush=True)
                with (output / f'{name}.log').open('w') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker', '--specification', str(specification)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600,
                                   env={**os.environ, 'OMP_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'})
                elapsed = time.monotonic() - value['started_monotonic']
                job = json.loads((output / f'{name}.json').read_text())
                job.update(key=key, mode=mode, process_seconds=round(elapsed, 3))
                jobs.append(job)
                write_json(output / 'progress.json', {'completed_jobs': len(jobs), 'total_jobs': len(plan), 'jobs': jobs})
                print(json.dumps({'event': 'complete', 'job': name, 'process_seconds': job['process_seconds']}), flush=True)
    finally:
        stop.set()
        telemetry.join(timeout=15)
        write_json(output / 'gpu.json', samples)
    after = henad_status()
    summary = {}
    for key in configurations:
        cold = [job['process_seconds'] for job in jobs if job['key'] == key and job['mode'].startswith('cold-')]
        reuse = next(job for job in jobs if job['key'] == key and job['mode'] == 'reuse')
        warm = [image['generation_seconds'] for image in reuse['images'][1:]]
        summary[key] = {**configurations[key], 'fresh_process_seconds': cold,
                        'fresh_process_median_seconds': round(statistics.median(cold), 3),
                        'reused_pipeline_seconds': warm, 'reused_pipeline_median_seconds': round(statistics.median(warm), 3)}
    report = {'started_at': samples[0]['at'] if samples else None, 'finished_at': time.time(),
              'home': str(home), 'lora_sha256': hashlib.sha256((home / 'lora/pytorch_lora_weights.safetensors').read_bytes()).hexdigest(),
              'resolution': 512, 'max_sequence_length': model_settings(home)['max_sequence_length'],
              'runtime': 'NF4 transformer and text encoder, bf16 compute, CPU offload, OMP_NUM_THREADS=2',
              'henad_before': before, 'henad_after': after,
              'gpu_peak_total_used_mib': max((s['memory_used_mib'] for s in samples), default=None),
              'summary': summary, 'jobs': jobs, 'lora_pixel_difference_mean': comparisons(output, jobs),
              'limitation': 'Three fresh requests and four reused-pipeline requests per variant. Same seeds and prompts do not imply identical images between different checkpoints. Visual examples are not a quality score.'}
    write_json(output / 'report.json', report)
    review_page(output, report)
    print(json.dumps({'event': 'report', 'summary': summary, 'henad_after': after}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'worker'])
    parser.add_argument('--home', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--specification', type=Path)
    args = parser.parse_args()
    if args.action == 'worker':
        if not args.specification:
            parser.error('worker requires --specification')
        worker(args.specification)
    else:
        if not args.home or not args.output:
            parser.error('run requires --home and --output')
        run(args.home.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
