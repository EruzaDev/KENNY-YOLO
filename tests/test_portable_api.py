"""Exercise the local runner in process, including actual weights and ASGI frames."""
import asyncio
import base64
import importlib.util
import json
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ['starlette', 'httpx', 'ultralytics'])


@unittest.skipUnless(AVAILABLE, 'Local ASGI/ML dependencies unavailable')
class PortableApiTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from app.core.config import MODEL_PATH
        if not MODEL_PATH.is_file():
            raise unittest.SkipTest('Trained checkpoint unavailable')
        from app.services.detector import Detector
        cls.detector = Detector(MODEL_PATH)
        cls.detector.load()
        cls.data_root = Path(cls.detector.metadata['config']['data'])
        cls.image = next((cls.data_root / 'val/open_palm').glob('*')).read_bytes()

    async def asyncSetUp(self):
        import httpx
        from app.portable import app
        self.app = app
        app.state.detector = self.detector
        app.state.inference_gate = asyncio.Semaphore(1)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://testserver')
        # The execution sandbox suppresses socket-based cross-thread wakeups.
        # A timer keeps this in-process ASGI test loop responsive without sockets.
        async def tick():
            while True:
                await asyncio.sleep(.02)
        self.ticker = asyncio.create_task(tick())

    async def asyncTearDown(self):
        await self.client.aclose()
        self.ticker.cancel()
        await asyncio.gather(self.ticker, return_exceptions=True)

    async def test_dashboard_assets_and_ready_model(self):
        for path in ['/', '/static/js/app.js', '/static/js/camera.js', '/static/css/style.css']:
            self.assertEqual((await self.client.get(path)).status_code, 200)
        self.assertEqual((await self.client.get('/api/health')).json()['status'], 'ready')
        model = (await self.client.get('/api/model')).json()
        self.assertTrue(model['ready'])
        self.assertEqual(model['missing_classes'], [])
        self.assertFalse(model['restart_required'])
        self.assertEqual(model['checkpoint_sha256'], self.detector.metadata['checkpoint_sha256'])
        self.assertEqual(model['backend'], 'starlette-local')
        self.assertFalse(model['robot_controls_enabled'])
        if model['metrics'] is not None:
            self.assertEqual(model['metrics']['checkpoint_sha256'], self.detector.metadata['checkpoint_sha256'])
        else:
            self.assertIsNotNone(model['validation_metrics'])

    async def test_actual_upload_prediction(self):
        response = await self.client.post('/api/predict', files={'file': ('palm.png', self.image, 'image/png')})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['task'], 'classify')
        self.assertEqual(result['predictions'][0]['class_name'], 'open_palm')
        self.assertIsNone(result['predictions'][0]['bbox'])

    async def test_checkpoint_linked_results_and_allowlisted_images(self):
        response = await self.client.get('/api/results')
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['available'])
        self.assertEqual(result['checkpoint_sha256'], self.detector.metadata['checkpoint_sha256'])
        for url in result['assets'].values():
            image = await self.client.get(url)
            self.assertEqual(image.status_code, 200)
            self.assertTrue(image.headers['content-type'].startswith('image/'))
        self.assertEqual((await self.client.get('/api/results/assets/raw-webcam')).status_code, 404)

    async def test_three_gestures_and_display_functions(self):
        from app.core.gestures import GESTURE_FUNCTIONS, CLASSES
        self.assertEqual(tuple(self.detector.classes), CLASSES)
        for label, function in GESTURE_FUNCTIONS.items():
            path = next((self.data_root / 'val' / label).glob('*'))
            response = await self.client.post('/api/predict', files={'file':(path.name, path.read_bytes(), 'image/png')})
            self.assertEqual(response.status_code, 200)
            prediction = response.json()['predictions'][0]
            self.assertEqual(prediction['class_name'], label)
            self.assertEqual(prediction['function'], function)
            self.assertGreaterEqual(prediction['confidence'], .8)

    async def test_actual_pair_contradiction_abstains(self):
        for label, hint in [('thumbs_up', 'thumb_folded')]:
            path = next((self.data_root / 'val' / label).glob('*'))
            response = await self.client.post('/api/predict?pose_hint=' + hint,
                files={'file':(path.name, path.read_bytes(), 'image/png')})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['predictions'], [])
            self.assertIn('rejection_reason', response.json())

    async def test_invalid_uploads(self):
        cases = [('/api/predict', ('x.txt', b'bad', 'text/plain'), 415),
                 ('/api/predict', ('x.png', b'bad', 'image/png'), 422),
                 ('/api/predict?pose_hint=invalid', ('x.png', self.image, 'image/png'), 422),
                 ('/api/predict?confidence=2', ('x.png', self.image, 'image/png'), 422),
                 ('/api/predict', ('x.png', b'x' * (5 * 1024 * 1024 + 1), 'image/png'), 413)]
        for url, upload, expected in cases:
            response = await self.client.post(url, files={'file': upload})
            self.assertEqual(response.status_code, expected)
            self.assertIn('detail', response.json())
        self.assertEqual((await self.client.post('/api/predict')).status_code, 422)

    async def test_webcam_frame_uses_actual_model(self):
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        scope = {'type': 'websocket', 'asgi': {'version': '3.0'}, 'scheme': 'ws',
                 'path': '/api/webcam', 'raw_path': b'/api/webcam', 'query_string': b'',
                 'headers': [(b'host', b'testserver'), (b'origin', b'http://testserver')],
                 'client': ('testclient', 123), 'server': ('testserver', 80), 'subprotocols': []}
        async def send(message):
            await outgoing.put(message)
        task = asyncio.create_task(self.app(scope, incoming.get, send))
        try:
            await incoming.put({'type': 'websocket.connect'})
            accepted = await asyncio.wait_for(outgoing.get(), 10)
            self.assertEqual(accepted['type'], 'websocket.accept')
            await incoming.put({'type': 'websocket.receive', 'text': 'invalid json'})
            malformed = await asyncio.wait_for(outgoing.get(), 10)
            self.assertIn('error', json.loads(malformed['text']))
            await incoming.put({'type': 'websocket.receive', 'text': json.dumps(
                {'image': base64.b64encode(self.image).decode(), 'confidence': .8, 'smoothing': 1})})
            message = await asyncio.wait_for(outgoing.get(), 20)
            result = json.loads(message['text'])
            self.assertEqual(result['predictions'][0]['class_name'], 'open_palm')
            self.assertEqual(result['stable_gesture'], 'open_palm')
            await asyncio.sleep(.09)
            await incoming.put({'type': 'websocket.receive', 'text': json.dumps(
                {'type':'no_hand', 'confidence':.8, 'smoothing':3})})
            reset = json.loads((await asyncio.wait_for(outgoing.get(), 10))['text'])
            self.assertEqual(reset['predictions'], [])
            self.assertIsNone(reset['stable_gesture'])
            self.assertEqual(reset['message'], 'No gesture detected')
            from app.core.gestures import GESTURE_FUNCTIONS
            for label in ['ok', 'thumbs_up']:
                await asyncio.sleep(.09)
                path = next((self.data_root / 'val' / label).glob('*'))
                await incoming.put({'type':'websocket.receive', 'text':json.dumps(
                    {'image':base64.b64encode(path.read_bytes()).decode(), 'confidence':.8, 'smoothing':1})})
                frame = json.loads((await asyncio.wait_for(outgoing.get(), 20))['text'])
                self.assertEqual(frame['stable_gesture'], label)
                self.assertEqual(frame['predictions'][0]['function'], GESTURE_FUNCTIONS[label])
            # Contradictory evidence clears an established gesture immediately.
            await asyncio.sleep(.09)
            await incoming.put({'type':'websocket.receive', 'text':json.dumps(
                {'image':base64.b64encode(path.read_bytes()).decode(), 'confidence':.8,
                 'smoothing':1, 'pose_hint':'thumb_folded'})})
            rejected = json.loads((await asyncio.wait_for(outgoing.get(), 20))['text'])
            self.assertEqual(rejected['predictions'], [])
            self.assertIsNone(rejected['stable_gesture'])
            await asyncio.sleep(.09)
            await incoming.put({'type':'websocket.receive', 'text':json.dumps(
                {'image':base64.b64encode(path.read_bytes()).decode(), 'confidence':.8, 'smoothing':3})})
            warming = json.loads((await asyncio.wait_for(outgoing.get(), 20))['text'])
            self.assertIsNone(warming['stable_gesture'])
            await incoming.put({'type': 'websocket.disconnect', 'code': 1000})
            await asyncio.wait_for(task, 5)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
