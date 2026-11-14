import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starlette.exceptions import HTTPException

from app.services import results


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.patcher = patch.object(results, 'ROOT', self.root)
        self.patcher.start()
        self.output = self.root / 'outputs/current'
        self.output.mkdir(parents=True)
        self.image = self.output / 'plot.png'
        self.image.write_bytes(b'plot contents')
        self.report = dict(checkpoint_sha256='current', assets={
            'curves': dict(path='outputs/current/plot.png', sha256=hashlib.sha256(self.image.read_bytes()).hexdigest())})
        self.report_path = self.output / 'results.json'
        self.report_path.write_text(json.dumps(self.report))
        self.detector = SimpleNamespace(model=object(), metadata={
            'checkpoint_sha256': 'current', 'results_report_path': 'outputs/current/results.json'})

    def tearDown(self):
        self.patcher.stop()
        self.directory.cleanup()

    def test_current_results_have_asset_urls_not_filesystem_paths(self):
        info = results.results_info(self.detector)
        self.assertTrue(info['available'])
        self.assertEqual(info['assets'], {'curves': '/api/results/assets/curves'})
        self.assertEqual(Path(results.results_asset(self.detector, 'curves').path), self.image)

    def test_stale_results_and_assets_are_suppressed(self):
        self.detector.metadata['checkpoint_sha256'] = 'another'
        self.assertFalse(results.results_info(self.detector)['available'])
        with self.assertRaises(HTTPException):
            results.results_asset(self.detector, 'curves')

    def test_tampered_plot_and_nonallowlisted_files_are_rejected(self):
        self.image.write_bytes(b'changed')
        for name in ['curves', '../raw-webcam.jpg', 'not-a-plot']:
            with self.assertRaises(HTTPException):
                results.results_asset(self.detector, name)

    def test_no_path_outside_outputs_can_be_served(self):
        self.detector.metadata['results_report_path'] = 'private.json'
        (self.root / 'private.json').write_text(json.dumps(self.report))
        self.assertFalse(results.results_info(self.detector)['available'])
        self.detector.metadata['results_report_path'] = 'outputs/current/results.json'
        self.report['assets']['curves']['path'] = 'private.png'
        (self.root / 'private.png').write_bytes(b'plot contents')
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaises(HTTPException):
            results.results_asset(self.detector, 'curves')

    def test_unloaded_model_has_no_results(self):
        self.detector.model = None
        self.assertFalse(results.results_info(self.detector)['available'])


if __name__ == '__main__':
    unittest.main()
