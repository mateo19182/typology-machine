from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from graffiti_api import Jobs, validate_request
from graffiti_model import FLUX_MODEL, FLUX_DISTILLED_MODEL, FLUX_DISTILLED_REVISION, generate_flux, model_settings
from graffiti_worker import PersistentWorker


FAKE_CHILD = '''
import json, sys, time
from pathlib import Path
home=Path(sys.argv[1])
settings=json.loads(sys.argv[2])
print(json.dumps({'event':'ready',**settings}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request['prompt']=='fail':
        print(json.dumps({'event':'failed','private':'/private/weights'}),flush=True)
        break
    if request['prompt']=='partial':
        sys.stdout.write('unfinished')
        sys.stdout.flush()
        time.sleep(30)
    output=home/'generated'/f"{request['id']}.png"
    output.write_bytes(b'png')
    print(json.dumps({'event':'complete','id':request['id']}),flush=True)
'''


def prepare(home):
    (home/'training.json').write_text(json.dumps({'stage':'trained','model':FLUX_MODEL,
        'revision':'a'*40,'recipe':{'text_max_sequence_length':128}}))
    (home/'inference.json').write_text(json.dumps({'profile':'distilled','revision':FLUX_DISTILLED_REVISION}))
    (home/'generated').mkdir()
    return model_settings(home)


@contextmanager
def fake_worker(home, settings):
    handshake={key:settings[key] for key in ('model_id','revision','steps')}
    worker=PersistentWorker(home,settings,[sys.executable,'-u','-c',FAKE_CHILD,str(home),json.dumps(handshake)])
    try:
        yield worker
    finally:
        worker.close()


class PersistentGraffitiTests(unittest.TestCase):
    def test_reuses_one_process_and_reloads_after_a_failed_request(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);settings=prepare(home)
            with fake_worker(home,settings) as worker:
                worker.warm()
                process=worker.process
                worker.generate('a'*32,validate_request({'prompt':'red graffiti','strength':0}))
                worker.generate('b'*32,validate_request({'prompt':'red graffiti','strength':1.2}))
                self.assertIs(worker.process,process)
                self.assertEqual(worker.status(),{'stage':'ready','loads':1,'generations':2})
                with self.assertRaisesRegex(RuntimeError,'failed its image request'):
                    worker.generate('c'*32,validate_request({'prompt':'fail'}))
                self.assertIsNone(worker.process)
                self.assertIsNotNone(process.poll())
                worker.generate('d'*32,validate_request({'prompt':'blue graffiti'}))
                self.assertIsNot(worker.process,process)
                self.assertEqual(worker.status(),{'stage':'ready','loads':2,'generations':3})

    def test_timeout_is_bounded_even_for_an_unterminated_response(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);settings=prepare(home)
            with fake_worker(home,settings) as worker:
                worker.warm()
                process=worker.process
                start=time.monotonic()
                with self.assertRaises(TimeoutError):
                    worker.generate('a'*32,validate_request({'prompt':'partial'}),timeout=.1)
                self.assertLess(time.monotonic()-start,2)
                self.assertIsNotNone(process.poll())
                self.assertEqual(worker.status()['stage'],'stopped')

    def test_persistent_api_saves_model_version_and_four_step_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);settings=prepare(home);jobs=Jobs(home)
            with fake_worker(home,settings) as worker:
                jobs.worker=worker
                params=validate_request({'prompt':'blue graffiti','seed':42,'strength':1.2})
                with patch('graffiti_api.threading.Thread'):
                    _,job=jobs.start(params)
                jobs.run(job['id'],params)
                self.assertEqual(jobs.get(job['id'])['progress'],{'phase':'complete','completed_steps':4,'total_steps':4})
                image=jobs.library()['images'][0]
                self.assertEqual(image['inference'],{'model_id':FLUX_DISTILLED_MODEL,'revision':FLUX_DISTILLED_REVISION,'steps':4})
                self.assertEqual(image['prompt'],params['prompt'])
                self.assertEqual(jobs.status()['inference_worker']['loads'],1)
                metadata=home/'generated'/f"{job['id']}.json"
                metadata.write_text(json.dumps({'prompt':params['prompt'],'seed':42,'strength':1.2,
                    'inference':{'model_id':FLUX_DISTILLED_MODEL,'revision':'/private/weights','steps':4}}))
                self.assertNotIn('inference',jobs.library()['images'][0])
                metadata.write_text('[]')
                self.assertEqual(len(jobs.library()['images']),1)

    def test_cached_pipeline_uses_four_steps_and_resets_adapter_strength(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);settings=prepare(home);pipe=Mock();steps=[]
            def draw(**kwargs):
                self.assertEqual(kwargs['prompt'],'wall')
                self.assertEqual(kwargs['num_inference_steps'],4)
                self.assertEqual(kwargs['guidance_scale'],1.0)
                self.assertEqual(kwargs['max_sequence_length'],128)
                for index in range(4):
                    kwargs['callback_on_step_end'](pipe,index,None,{})
                    steps.append(json.loads((home/'progress.json').read_text())['completed_steps'])
                return Mock(images=[Mock()])
            pipe.side_effect=draw
            with patch('graffiti_model.load_flux_pipeline') as load,patch.dict('sys.modules',{'torch':Mock()}):
                for strength in (0,1.2):
                    generate_flux(home,'wall',home/'output.png',strength=strength,
                                  progress=home/'progress.json',pipe=pipe,settings=settings)
                load.assert_not_called()
            self.assertEqual(steps,[1,2,3,4,1,2,3,4])
            self.assertEqual([call.kwargs['adapter_weights'] for call in pipe.set_adapters.call_args_list],[[0],[1.2]])
            self.assertEqual(json.loads((home/'training.json').read_text())['model'],FLUX_MODEL)


if __name__=='__main__':
    unittest.main()
