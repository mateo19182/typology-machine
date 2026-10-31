"""Generate fixed-seed base/LoRA comparisons in a fresh process after FLUX training."""
import argparse
import json
from pathlib import Path


def evaluate(home, quantized=False):
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image, ImageDraw
    torch.set_num_threads(4)
    state = json.loads((home / 'training.json').read_text())
    if state['stage'] != 'trained':
        raise ValueError('Only evaluate a completed adapter')
    if quantized:
        from graffiti_model import load_flux_pipeline
        pipe = load_flux_pipeline(home, model_id=state['model'], revision=state['revision'])
    else:
        pipe = Flux2KleinPipeline.from_pretrained(state['model'], revision=state['revision'],
                                                 torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.load_lora_weights(str(home / 'lora'), adapter_name='coruna')
    prompts = [
        'abstract blue and coral lettering on weathered concrete, daylight',
        'bold red and white bubble lettering on a corrugated metal shutter, daylight',
        'a colorful mural of a bird on a weathered concrete wall, daylight',
    ]
    per_image = state.get('recipe', {}).get('caption_strategy') == 'per-image captions'
    if per_image:
        prompts = ['wall', 'wall', 'wall', *prompts,
                   'graffiti lettering that says "HELLO" on a concrete wall']
    output = home / 'evaluation'
    output.mkdir(exist_ok=True)
    sheet = Image.new('RGB', (2 * 512, len(prompts) * 550), '#e8e5dc')
    draw = ImageDraw.Draw(sheet)
    results = []
    for index, subject in enumerate(prompts):
        prompt = subject if per_image else 'a photograph of c0runa graffiti, painted street art on a wall, ' + subject
        seed = state['seed'] + index
        for column, strength in enumerate((0., .8)):
            pipe.set_adapters(['coruna'], adapter_weights=[strength])
            image = pipe(prompt=prompt, height=512, width=512, num_inference_steps=30,
                         max_sequence_length=state.get('recipe', {}).get('text_max_sequence_length', 512),
                         guidance_scale=4., generator=torch.Generator('cpu').manual_seed(seed)).images[0]
            name = f'{index + 1}-' + ('base.png' if column == 0 else 'lora.png')
            image.save(output / name)
            if index == 0:
                image.save(home / ('base-sample.png' if column == 0 else 'lora-sample.png'))
            sheet.paste(image, (column * 512, index * 550))
            draw.text((column * 512 + 8, index * 550 + 515),
                      f'{"Base" if column == 0 else "LoRA 0.8"}, seed {seed}', fill='black')
            results.append({'prompt': prompt, 'seed': seed, 'strength': strength,
                            'file_name': name, 'path': 'evaluation/' + name})
    sheet.save(home / 'evaluation-comparison.jpg', quality=92)
    report = {'method': 'Same prompt, seed, scheduler and sampling settings; LoRA strength 0 versus 0.8.',
              'prompt_mode': 'raw user prompt; no hidden prefix' if per_image else 'legacy c0runa prefix',
              'quantization': 'NF4 transformer and text encoder, bf16 compute' if quantized else 'bf16 weights',
              'steps': 30, 'guidance_scale': 4., 'resolution': 512, 'samples': results,
              'limitation': 'Illustrative comparisons, not a held-out quality score.'}
    (home / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--quantized', action='store_true', help='Use the same NF4 pipeline as the website')
    args=parser.parse_args()
    evaluate(args.home.resolve(),args.quantized)
