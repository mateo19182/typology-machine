"""Train the original SD experiment and generate with trained SD or FLUX adapters."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
FLUX_MODEL = 'black-forest-labs/FLUX.2-klein-base-4B'
FLUX_DISTILLED_MODEL = 'black-forest-labs/FLUX.2-klein-4B'
FLUX_DISTILLED_REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'
DEFAULT_HOME = Path(os.environ.get('TYPOLOGY_GRAFFITI_HOME', '/mnt/data/projects/typology-machine/graffiti-ai'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def model_path(home):
    return home / 'hf/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots' / REVISION


def model_settings(home):
    """Choose the runtime from the completed training record, without GPU imports."""
    try:
        state = json.loads((home / 'training.json').read_text())
    except (OSError, ValueError):
        state = {}
    model = state.get('model', MODEL)
    if model == FLUX_MODEL:
        settings = {'model': 'FLUX.2 klein Base 4B + Coruña graffitiLoRA', 'model_id': model,
                    'revision': state.get('revision'), 'resolution': 512, 'steps': 30, 'guidance_scale': 4.0,
                    'worker_mode': 'fresh', 'max_sequence_length': state.get('recipe', {}).get('text_max_sequence_length', 512)}
        if (home / 'inference.json').is_file():
            inference = json.loads((home / 'inference.json').read_text())
            if inference.get('profile') != 'distilled' or inference.get('revision') != FLUX_DISTILLED_REVISION:
                raise ValueError('Unsupported graffiti inference profile or revision')
            settings.update(model='FLUX.2 klein 4B + Coruña graffitiLoRA', model_id=FLUX_DISTILLED_MODEL,
                            revision=FLUX_DISTILLED_REVISION, steps=4, guidance_scale=1.0, worker_mode='persistent')
        return settings
    if model == MODEL:
        return {'model': 'Stable Diffusion 1.5 + Coruña LoRA', 'model_id': model, 'resolution': 512, 'steps': 24}
    raise ValueError('Unsupported graffiti base model')


def train(home, steps=300, seed=19182):
    from publish_huggingface import validate_reviewed_dataset
    summary, rows = validate_reviewed_dataset(home / 'dataset')
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    random.seed(seed)
    dtype = torch.bfloat16
    base = str(model_path(home))
    state = {'stage': 'encoding', 'model': MODEL, 'revision': REVISION, 'steps': steps, 'step': 0, 'started': time.time(), 'seed': seed}
    def report(**changes):
        state.update(changes, updated=time.time())
        write_json(home / 'training.json', state)
        print(json.dumps(changes), flush=True)
    report()
    report(train_count=len(rows['train']), validation_count=len(rows['validation']), dataset_summary=summary,
           dataset_manifest_sha256=summary['manifest_sha256'])
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder='tokenizer', local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(base, subfolder='text_encoder', variant='fp16', torch_dtype=dtype, local_files_only=True).to('cuda').eval()
    vae = AutoencoderKL.from_pretrained(base, subfolder='vae', variant='fp16', torch_dtype=dtype, local_files_only=True).to('cuda').eval()
    vae.requires_grad_(False); encoder.requires_grad_(False)
    cached = {}
    with torch.no_grad():
        for split, records in rows.items():
            latents, embeds = [], []
            for index, row in enumerate(records):
                image = ImageOps.fit(Image.open(home/'dataset'/split/row['file_name']).convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
                pixel = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).unsqueeze(0).to(device='cuda',dtype=dtype)/127.5-1
                latent = vae.encode(pixel).latent_dist.sample()*vae.config.scaling_factor
                tokens = tokenizer(row['text'],padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt').input_ids.to('cuda')
                latents.append(latent.cpu()); embeds.append(encoder(tokens)[0].cpu())
                if index % 16 == 0: report(encoding_split=split,encoded=index+1)
            cached[split] = (torch.cat(latents),torch.cat(embeds))
    del vae,encoder,pixel,latent,tokens
    gc.collect();torch.cuda.empty_cache()
    unet=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',torch_dtype=dtype,local_files_only=True).to('cuda')
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_k','to_q','to_v','to_out.0']))
    for param in unet.parameters():
        if param.requires_grad: param.data=param.data.float()
    unet.enable_gradient_checkpointing()
    optimizer=torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad],lr=1e-4)
    scheduler=DDPMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True)
    def loss_for(split,index,generator=None):
        latent=cached[split][0][index:index+1].to('cuda');embed=cached[split][1][index:index+1].to('cuda')
        noise=torch.randn(latent.shape,device='cuda',dtype=dtype,generator=generator)
        timestep=torch.randint(0,scheduler.config.num_train_timesteps,(1,),device='cuda',generator=generator)
        noisy=scheduler.add_noise(latent,noise,timestep)
        with torch.autocast('cuda',dtype=dtype):
            prediction=unet(noisy,timestep,encoder_hidden_states=embed).sample
        target=scheduler.get_velocity(latent,noise,timestep) if scheduler.config.prediction_type=='v_prediction' else noise
        return torch.nn.functional.mse_loss(prediction.float(),target.float())
    def validation():
        unet.eval();g=torch.Generator(device='cuda').manual_seed(seed+1)
        with torch.no_grad(): values=[loss_for('validation',i,g).item() for i in range(min(16,len(rows['validation']))) ]
        unet.train();return sum(values)/len(values)
    before=validation()
    report(stage='training',validation_loss_before=before,rank=4,resolution=512)
    unet.train()
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss=loss_for('train',random.randrange(len(rows['train'])))
        if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad],1.)
        optimizer.step()
        if step==1 or step%10==0: report(step=step,loss=float(loss.detach()))
    after=validation()
    output=home/'lora';output.mkdir(parents=True,exist_ok=True)
    weights=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(str(output),unet_lora_layers=weights,safe_serialization=True)
    report(stage='trained',step=steps,validation_loss_after=after,finished=time.time())


def generate(home, prompt, output, seed=19182, strength=.8, use_lora=True, progress=None):
    settings = model_settings(home)
    options = {'progress': progress} if progress is not None else {}
    if settings['model_id'] in (FLUX_MODEL, FLUX_DISTILLED_MODEL):
        return generate_flux(home, prompt, output, seed, strength, use_lora, **options)
    return generate_sd15(home, prompt, output, seed, strength, use_lora, **options)


def load_flux_pipeline(home, use_lora=True, *, model_id=None, revision=None):
    import torch
    from diffusers import BitsAndBytesConfig, Flux2KleinPipeline, Flux2Transformer2DModel
    from transformers import BitsAndBytesConfig as EncoderBitsAndBytesConfig, Qwen3ForCausalLM
    torch.set_num_threads(2)
    settings = model_settings(home)
    model_id = model_id or settings['model_id']
    source = {'revision': revision or settings['revision'], 'local_files_only': True, 'torch_dtype': torch.bfloat16}
    quantization = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                    'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
    # Compress both large components so inference can share the 3090 with henad.
    print('Loading FLUX transformer', flush=True)
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder='transformer', quantization_config=BitsAndBytesConfig(**quantization), **source)
    print('Loading FLUX text encoder', flush=True)
    encoder = Qwen3ForCausalLM.from_pretrained(
        model_id, subfolder='text_encoder', quantization_config=EncoderBitsAndBytesConfig(**quantization), **source)
    pipe = Flux2KleinPipeline.from_pretrained(model_id, transformer=transformer, text_encoder=encoder, **source)
    if use_lora:
        pipe.load_lora_weights(str(home / 'lora'), adapter_name='coruna')
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    return pipe


def generation_reporter(path, total_steps):
    def report(phase, completed_steps=0):
        if path is not None:
            write_json(path, {'phase': phase, 'completed_steps': completed_steps, 'total_steps': total_steps})
    def step_end(pipe, index, timestep, values):
        completed = index + 1
        report('finishing' if completed == total_steps else 'generating', completed)
        return values
    return report, step_end


def generate_flux(home, prompt, output, seed=19182, strength=.8, use_lora=True, progress=None, *, pipe=None, settings=None):
    settings = settings or model_settings(home)
    report, step_end = generation_reporter(progress, settings['steps'])
    import torch
    if pipe is None:
        report('loading')
        pipe = load_flux_pipeline(home, use_lora)
    if use_lora:
        pipe.set_adapters(['coruna'], adapter_weights=[strength])
    print('Generating image', flush=True)
    report('encoding')
    image = pipe(prompt=prompt,
                 height=512, width=512, num_inference_steps=settings['steps'], guidance_scale=settings['guidance_scale'],
                 max_sequence_length=settings['max_sequence_length'],
                 callback_on_step_end=step_end, callback_on_step_end_tensor_inputs=[],
                 generator=torch.Generator('cpu').manual_seed(seed)).images[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def generate_sd15(home, prompt, output, seed=19182, strength=.8, use_lora=True, progress=None):
    report, step_end = generation_reporter(progress, 24)
    report('loading')
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    torch.set_num_threads(2)
    pipe=StableDiffusionPipeline.from_pretrained(str(model_path(home)),torch_dtype=torch.float16,variant='fp16',local_files_only=True)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if use_lora:
        pipe.load_lora_weights(str(home/'lora'),adapter_name='coruna')
        pipe.set_adapters(['coruna'],adapter_weights=[strength])
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    report('encoding')
    image=pipe(prompt='a photograph of c0runa graffiti, painted street art on a wall, '+prompt,negative_prompt='blurry, low quality, watermark, signature, people',height=512,width=512,num_inference_steps=24,guidance_scale=7,callback_on_step_end=step_end,callback_on_step_end_tensor_inputs=[],generator=torch.Generator('cpu').manual_seed(seed)).images[0]
    output.parent.mkdir(parents=True,exist_ok=True)
    image.save(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','generate'])
    parser.add_argument('--home',type=Path,default=DEFAULT_HOME)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--prompt',default='abstract blue and coral lettering, weathered concrete, daylight')
    parser.add_argument('--seed',type=int,default=19182)
    parser.add_argument('--strength',type=float,default=.8)
    parser.add_argument('--base',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--progress',type=Path,help='Write generation phases and completed drawing steps atomically')
    parser.add_argument('--publish',action='store_true',help='Upload the completed adapter and reviewed dataset to private Hugging Face repos')
    parser.add_argument('--namespace',help='Hugging Face namespace; defaults to the authenticated account')
    args=parser.parse_args()
    if args.publish and args.action!='train':
        parser.error('--publish applies to training')
    if args.action=='train':
        try: train(args.home,args.steps,args.seed)
        except Exception as error:
            path=args.home/'training.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            write_json(path,{**state,'stage':'failed','error':str(error),'updated':time.time()})
            raise
        if args.publish:
            from publish_huggingface import publish_run
            result=publish_run(args.home,args.home/f'hub-{int(time.time())}',args.namespace,upload=True)
            print(json.dumps(result,indent=2))
    else:
        generate(args.home,args.prompt,args.output or args.home/'sample.png',args.seed,args.strength,not args.base,progress=args.progress)

if __name__=='__main__': main()
