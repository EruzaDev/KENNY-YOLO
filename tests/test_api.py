"""Isolated API unit tests use mocks; deployed application never does."""
import base64
import importlib.util
import io
import unittest
from unittest.mock import patch

from PIL import Image

AVAILABLE = importlib.util.find_spec('fastapi') is not None


@unittest.skipUnless(AVAILABLE, 'FastAPI dependency unavailable')
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app.main import app
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.detector = app.state.detector
        self.detector.model = object()
        self.output = dict(task='classify', predictions=[], scores={'open_palm': .5, 'thumbs_up': .5},
                           message='No gesture detected', width=24, height=24, latency_ms=1)
        buffer = io.BytesIO(); Image.new('RGB', (24, 24), 'white').save(buffer, format='PNG')
        self.image = buffer.getvalue()

    def test_health_and_model_info(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        self.assertEqual(self.client.get('/api/model').json()['task'], 'classify')
        self.assertEqual(self.client.get('/').status_code, 200)

    def test_real_upload_route_with_isolated_mock(self):
        with patch.object(self.detector, 'predict', return_value=self.output) as predict:
            response = self.client.post('/api/predict', files={'file': ('frame.png', self.image, 'image/png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['predictions'], [])
        predict.assert_called_once()

    def test_invalid_upload_and_settings(self):
        self.assertEqual(self.client.post('/api/predict', files={'file': ('x.txt', b'x', 'text/plain')}).status_code, 415)
        self.assertEqual(self.client.post('/api/predict', files={'file': ('x.png', b'x', 'image/png')}).status_code, 422)
        self.assertEqual(self.client.post('/api/predict?confidence=2', files={'file': ('x.png', self.image, 'image/png')}).status_code, 422)
        self.assertEqual(self.client.post('/api/predict', files={'file': ('x.png', b'x' * (5*1024*1024+1), 'image/png')}).status_code, 413)

    def test_missing_model(self):
        self.detector.model = None
        response = self.client.post('/api/predict', files={'file': ('x.png', self.image, 'image/png')})
        self.assertEqual(response.status_code, 503)

    def test_webcam_frame_and_invalid_payload(self):
        with patch.object(self.detector, 'predict', return_value=self.output):
            with self.client.websocket_connect('/api/webcam') as socket:
                socket.send_text('not json')
                self.assertIn('error', socket.receive_json())
                socket.send_json({'image': base64.b64encode(self.image).decode(), 'smoothing': 3})
                result = socket.receive_json()
                self.assertEqual(result['predictions'], [])
                self.assertIsNone(result['stable_gesture'])

    def test_actual_model_upload(self):
        from pathlib import Path
        from app.services.detector import Detector
        from app.core.config import MODEL_PATH
        if not MODEL_PATH.exists():
            self.skipTest('Actual trained checkpoint unavailable')
        detector = Detector(MODEL_PATH); detector.load()
        from app.main import app
        app.state.detector = detector
        path = next(Path('data/processed/val').rglob('*.png'))
        response = self.client.post('/api/predict', files={'file': (path.name, path.read_bytes(), 'image/png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['task'], 'classify')
        results = self.client.get('/api/results').json()
        self.assertTrue(results['available'])
        self.assertEqual(results['checkpoint_sha256'], detector.metadata['checkpoint_sha256'])
        self.assertEqual(self.client.get(results['assets']['training-curves']).status_code, 200)
