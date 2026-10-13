import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from graffiti_api import Jobs, validate_request


class GraffitiTests(unittest.TestCase):
    def test_public_request_bounds(self):
        self.assertEqual(validate_request({'prompt':'painted blue letters','seed':42})['seed'],42)
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
                self.assertEqual(jobs.start(params)[0],429)
    def test_status_never_exposes_paths_or_tracebacks(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'training.json').write_text(json.dumps({'stage':'failed','error':'/private/path','step':3}))
            self.assertEqual(Jobs(Path(temp)).training(),{'stage':'failed','step':3})

if __name__=='__main__': unittest.main()
