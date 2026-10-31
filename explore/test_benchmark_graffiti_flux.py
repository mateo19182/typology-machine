import tempfile
from pathlib import Path
import unittest
from unittest.mock import call, patch

from benchmark_graffiti_flux import pause_generator


class BenchmarkMaintenanceTests(unittest.TestCase):
    def test_failed_benchmark_restores_generator_and_preserves_other_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            directory = home / '.config/systemd/user/typology-graffiti.service.d'
            directory.mkdir(parents=True)
            existing = directory / 'other.conf'
            existing.write_text('[Service]\nNice=10\n')
            own = directory / '90-inference-benchmark.conf'
            with patch('benchmark_graffiti_flux.Path.home', return_value=home), \
                 patch('benchmark_graffiti_flux.generator_status', side_effect=[{'busy': False, 'paused': False}, {'paused': True}]), \
                 patch('benchmark_graffiti_flux.service') as service:
                with self.assertRaisesRegex(RuntimeError, 'GPU worker failed'):
                    with pause_generator():
                        self.assertTrue(own.is_file())
                        raise RuntimeError('GPU worker failed')
            self.assertFalse(own.exists())
            self.assertEqual(existing.read_text(), '[Service]\nNice=10\n')
            self.assertEqual(service.call_args_list, [call(['daemon-reload']),
                call(['restart', 'typology-graffiti.service']), call(['daemon-reload']),
                call(['restart', 'typology-graffiti.service'])])

    def test_busy_or_paused_generator_is_not_interrupted(self):
        for status in [{'busy': True, 'paused': False}, {'busy': False, 'paused': True}]:
            with tempfile.TemporaryDirectory() as temp:
                home = Path(temp)
                with patch('benchmark_graffiti_flux.Path.home', return_value=home), \
                     patch('benchmark_graffiti_flux.generator_status', return_value=status), \
                     patch('benchmark_graffiti_flux.service') as service:
                    with self.assertRaisesRegex(RuntimeError, 'Wait for the current'):
                        with pause_generator():
                            self.fail('An occupied generator must not enter maintenance')
                    service.assert_not_called()
                self.assertFalse((home / '.config').exists())


if __name__ == '__main__':
    unittest.main()
