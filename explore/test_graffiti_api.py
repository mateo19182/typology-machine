import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from graffiti_api import Jobs, validate_request
from graffiti_model import FLUX_MODEL, MODEL, generate, generate_flux


class GraffitiTests(unittest.TestCase):
    def test_public_request_bounds(self):
        self.assertEqual(validate_request({'prompt':'painted blue letters','seed':42})['seed'],42)
        self.assertEqual(validate_request({'prompt':'painted blue letters'})['strength'],1.2)
        for value in [None,[],{}, {'prompt':'x'}, {'prompt':'x'*501}, {'prompt':'abc','seed':-1}, {'prompt':'abc','strength':float('nan')}, {'prompt':'abc','strength':True}]:
            with self.assertRaises(ValueError): validate_request(value)
    def test_training_gate_and_global_job_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs=Jobs(Path(temp)); params=validate_request({'prompt':'painted blue letters'})
            self.assertEqual(jobs.start(params)[0],503)
            (Path(temp)/'training.json').write_text(json.dumps({'stage':'trained'}))
            with patch('graffiti_api.threading.Thread'):
                self.assertEqual(jobs.start(params)[0],202)
                self.assertEqual(jobs.start(params)[0],429)
                jobs.active=None
                self.assertEqual(jobs.start(params)[0],202)
    def test_status_never_exposes_paths_or_tracebacks(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'training.json').write_text(json.dumps({'stage':'failed','error':'/private/path','step':3}))
            self.assertEqual(Jobs(Path(temp)).training(),{'stage':'failed','step':3})
    def test_flux_status_and_generation_use_training_record(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)
            (home/'training.json').write_text(json.dumps({'stage':'trained','model':FLUX_MODEL,
                'steps':2000,'train_count':971,'validation_count':92,'recipe':{'path':'/private/weights'}}))
            status=Jobs(home).status()
            self.assertEqual(status['model_id'],FLUX_MODEL)
            self.assertEqual(status['steps'],30)
            self.assertEqual(status['training']['train_count'],971)
            self.assertNotIn('/private',json.dumps(status))
            with patch('graffiti_model.generate_flux') as flux, patch('graffiti_model.generate_sd15') as sd:
                generate(home,'blue letters',home/'image.png',42,0)
                flux.assert_called_once_with(home,'blue letters',home/'image.png',42,0,True)
                sd.assert_not_called()
    def test_legacy_sd_and_unsupported_model(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)
            self.assertEqual(Jobs(home).status()['model_id'],MODEL)
            with patch('graffiti_model.generate_sd15') as sd:
                generate(home,'blue letters',home/'image.png',use_lora=False)
                sd.assert_called_once_with(home,'blue letters',home/'image.png',19182,.8,False)
            (home/'training.json').write_text(json.dumps({'model':'unknown/model'}))
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                generate(home,'blue letters',home/'image.png')
    def test_flux_sends_raw_prompt_without_inserting_a_city_name(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)
            (home/'training.json').write_text(json.dumps({'model':FLUX_MODEL,'recipe':{'text_max_sequence_length':128}}))
            pipe=Mock();pipe.return_value.images=[Mock()]
            with patch('graffiti_model.load_flux_pipeline',return_value=pipe),patch.dict('sys.modules',{'torch':Mock()}):
                generate_flux(home,'wall',home/'image.png')
            self.assertEqual(pipe.call_args.kwargs['prompt'],'wall')
            self.assertEqual(pipe.call_args.kwargs['max_sequence_length'],128)
    def test_maintenance_keeps_status_available_without_starting_a_gpu_job(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict('os.environ',{'TYPOLOGY_GRAFFITI_PAUSED':'1'}):
            home=Path(temp);(home/'training.json').write_text(json.dumps({'stage':'trained'}))
            jobs=Jobs(home)
            self.assertTrue(jobs.status()['paused'])
            self.assertEqual(jobs.start(validate_request({'prompt':'wall'}))[0],503)
            self.assertFalse(jobs.jobs)
    def test_training_progress_is_bounded_and_redacted(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'training.json').write_text(json.dumps({'stage':'training','steps':2000,'step':0}))
            (home/'progress.json').write_text(json.dumps({'step':1200,'private':'/private/path'}))
            self.assertEqual(Jobs(home).training(),{'stage':'training','steps':2000,'step':1200})
            (home/'progress.json').write_text(json.dumps({'step':2001}))
            self.assertEqual(Jobs(home).training()['step'],0)
    def test_worker_failure_releases_gpu_slot_and_removes_partial_image(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs=Jobs(Path(temp));params=validate_request({'prompt':'blue letters'})
            (Path(temp)/'training.json').write_text(json.dumps({'stage':'trained'}))
            with patch('graffiti_api.threading.Thread'):
                _,job=jobs.start(params)
            partial=Path(temp)/'generated'/f"{job['id']}.png"
            partial.write_bytes(b'partial')
            with patch('graffiti_api.subprocess.run',side_effect=TimeoutError('/private/weights')):
                jobs.run(job['id'],params)
            self.assertEqual(jobs.jobs[job['id']]['stage'],'failed')
            self.assertNotIn('/private',json.dumps(jobs.jobs[job['id']]))
            self.assertIsNone(jobs.active)
            self.assertFalse(partial.exists())
    def test_generation_progress_is_bounded_and_failure_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'training.json').write_text(json.dumps({'stage':'trained','model':FLUX_MODEL}))
            jobs=Jobs(home)
            with patch('graffiti_api.threading.Thread'),patch('graffiti_api.time.time',return_value=100):
                _,job=jobs.start(validate_request({'prompt':'blue letters'}))
            progress=home/'generated'/f"{job['id']}.progress.json"
            progress.write_text(json.dumps({'phase':'generating','completed_steps':12,'total_steps':500,'private':'/weights'}))
            with patch('graffiti_api.time.time',return_value=115): value=jobs.get(job['id'])
            self.assertEqual(value['progress'],{'phase':'generating','completed_steps':12,'total_steps':30})
            self.assertEqual(value['elapsed_seconds'],15)
            self.assertNotIn('/weights',json.dumps(value))
            for bad in [None,{'phase':'generating','completed_steps':31},{'phase':'loading','completed_steps':True}]:
                progress.write_text(json.dumps(bad))
                self.assertEqual(jobs.get(job['id'])['progress']['completed_steps'],0)
            jobs.jobs[job['id']].update(stage='failed',duration_seconds=18.25)
            self.assertEqual(jobs.get(job['id'])['progress']['phase'],'failed')
            jobs.jobs[job['id']]['stage']='complete'
            self.assertEqual(jobs.get(job['id'])['progress']['completed_steps'],30)
            self.assertEqual(jobs.get(job['id'])['elapsed_seconds'],18.2)
    def test_only_successful_durations_are_persisted_and_estimates_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'training.json').write_text(json.dumps({'stage':'trained','model':FLUX_MODEL}))
            jobs=Jobs(home);params=validate_request({'prompt':'blue letters'})
            for duration in [20,30,100]:
                with patch('graffiti_api.threading.Thread'): _,job=jobs.start(params)
                with patch('graffiti_api.subprocess.run'),patch('graffiti_api.time.monotonic',side_effect=[0,duration]):
                    jobs.run(job['id'],params)
            timing=Jobs(home).status()['generation_timing']
            self.assertEqual(timing,{'sample_count':3,'typical_seconds':30,'min_seconds':20,'max_seconds':100})
            with patch('graffiti_api.threading.Thread'): _,job=jobs.start(params)
            with patch('graffiti_api.subprocess.run',side_effect=TimeoutError()),patch('graffiti_api.time.monotonic',side_effect=[0,600]):
                jobs.run(job['id'],params)
            self.assertEqual(Jobs(home).status()['generation_timing'],timing)
            (home/'training.json').write_text(json.dumps({'stage':'trained','model':MODEL}))
            self.assertEqual(Jobs(home).status()['generation_timing']['sample_count'],0)
    def test_flux_step_callback_reports_real_steps_and_finishing_before_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'training.json').write_text(json.dumps({'model':FLUX_MODEL,'recipe':{'text_max_sequence_length':128}}))
            progress=home/'progress.json';pipe=Mock();observed=[]
            def draw(**kwargs):
                observed.append(json.loads(progress.read_text()))
                callback=kwargs['callback_on_step_end']
                for index in [0,14,29]:
                    self.assertEqual(callback(pipe,index,None,{}),{})
                    observed.append(json.loads(progress.read_text()))
                return Mock(images=[Mock()])
            pipe.side_effect=draw
            with patch('graffiti_model.load_flux_pipeline',return_value=pipe),patch.dict('sys.modules',{'torch':Mock()}):
                generate_flux(home,'red letters',home/'image.png',progress=progress)
            self.assertEqual([value['completed_steps'] for value in observed],[0,1,15,30])
            self.assertEqual(observed[0]['phase'],'encoding')
            self.assertEqual(observed[-1]['phase'],'finishing')
    def test_library_survives_restart_and_hides_active_or_invalid_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);jobs=Jobs(home);saved=home/'generated'
            older='a'*32;newer='b'*32;active='c'*32
            for index,job in enumerate([older,newer,active]):
                file=saved/f'{job}.png';file.write_bytes(b'png');os.utime(file,(index+100,index+100))
            (saved/f'{newer}.json').write_text(json.dumps({'prompt':'red graffiti','seed':42,'strength':1.2,'private':'/weights','error':'traceback'}))
            (saved/f'{older}.json').write_text('{broken')
            (saved/'other.png').write_bytes(b'not a generation')
            jobs.active=active
            images=jobs.library()['images']
            self.assertEqual([item['id'] for item in images],[newer,older])
            self.assertEqual(images[0]['prompt'],'red graffiti')
            self.assertNotIn('prompt',images[1])
            self.assertNotIn('/weights',json.dumps(images))
            restarted=Jobs(home).library()['images']
            self.assertEqual([item['id'] for item in restarted],[active,newer,older])
            self.assertEqual(restarted[1]['seed'],42)
    def test_library_retains_200_and_thumbnails_are_cached_without_gpu_work(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);jobs=Jobs(home);saved=home/'generated'
            for index in range(205):
                file=saved/f'{index:032x}.png';file.write_bytes(b'png');os.utime(file,(index+1,index+1))
            images=jobs.library()['images']
            self.assertEqual(len(images),200)
            self.assertEqual(images[0]['id'],f'{204:032x}')
            self.assertEqual(images[-1]['id'],f'{5:032x}')
            job='f'*32;Image.new('RGB',(512,512),'red').save(saved/f'{job}.png')
            raw=jobs.thumbnail(job)
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.format,'WEBP');self.assertEqual(image.size,(256,256))
            with patch('PIL.Image.open',side_effect=AssertionError('Thumbnail should be cached')):
                self.assertEqual(jobs.thumbnail(job),raw)
            self.assertIsNone(jobs.thumbnail('e'*32))
            jobs.active=job;self.assertIsNone(jobs.thumbnail(job))
    def test_completed_generation_saves_settings_for_the_library(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'training.json').write_text(json.dumps({'stage':'trained'}))
            jobs=Jobs(home);params=validate_request({'prompt':'white graffiti on a shutter','seed':123,'strength':.4})
            with patch('graffiti_api.threading.Thread'): _,job=jobs.start(params)
            def generate(command,**kwargs): Image.new('RGB',(512,512),'white').save(command[command.index('--output')+1])
            with patch('graffiti_api.subprocess.run',side_effect=generate): jobs.run(job['id'],params)
            item=Jobs(home).library()['images'][0]
            self.assertEqual(item['id'],job['id']);self.assertEqual(item['prompt'],params['prompt'])
            self.assertEqual(item['seed'],123);self.assertEqual(item['strength'],.4)

if __name__=='__main__': unittest.main()
