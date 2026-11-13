import io
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from app.services.detector import Detector
from app.services.inference import Stabilizer
from app.utils.image_utils import decode_image


class InferenceTests(unittest.TestCase):
    def test_missing_model_has_no_fallback(self):
        detector = Detector(Path('/tmp/nonexistent-gesture-model.pt'))
        with self.assertRaises(FileNotFoundError):
            detector.load()
        self.assertIsNone(detector.model)

    def test_loaded_checkpoint_identity_detects_stale_server_and_legacy_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            path.write_bytes(b'original checkpoint fixture')
            detector = Detector(path)
            detector.model = object()  # Unit test only; no prediction is mocked.
            detector.metadata = {'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertFalse(detector.deployment_status()['restart_required'])
            self.assertEqual(detector.deployment_status()['missing_classes'], [])
            path.write_bytes(b'updated checkpoint fixture')
            self.assertTrue(detector.deployment_status()['restart_required'])
            detector.classes = ['open_palm', 'thumbs_up']
            self.assertEqual(detector.deployment_status()['missing_classes'], ['closed_fist'])
            path.unlink()
            self.assertTrue(detector.deployment_status()['restart_required'])

    def test_decode_rgb(self):
        stream = io.BytesIO()
        Image.new('RGBA', (50, 30), 'white').save(stream, format='PNG')
        image = decode_image(stream.getvalue())
        self.assertEqual(image.mode, 'RGB')
        self.assertEqual(image.size, (50, 30))

    def test_invalid_images(self):
        for data in [b'', b'not an image', b'x' * (5 * 1024 * 1024 + 1)]:
            with self.assertRaises(ValueError):
                decode_image(data)

    def test_stabilization_and_immediate_abstention(self):
        stabilizer = Stabilizer()
        def frame(label):
            return dict(predictions=[dict(class_name=label, confidence=.9)] if label else [])
        self.assertIsNone(stabilizer.apply(frame('open_palm'), 3)['stable_gesture'])
        self.assertIsNone(stabilizer.apply(frame('open_palm'), 3)['stable_gesture'])
        self.assertEqual(stabilizer.apply(frame('open_palm'), 3)['stable_gesture'], 'open_palm')
        self.assertIsNone(stabilizer.apply(frame('thumbs_up'), 3)['stable_gesture'])
        self.assertIsNone(stabilizer.apply(frame(None), 3)['stable_gesture'])
        self.assertIsNone(stabilizer.apply(frame('thumbs_up'), 3)['stable_gesture'])
        with self.assertRaises(ValueError):
            stabilizer.apply(frame(None), 0)

    def test_blank_images_never_receive_a_gesture(self):
        if not Path('models/best.pt').exists():
            self.skipTest('Trained checkpoint unavailable')
        detector = Detector(Path('models/best.pt'))
        detector.load()
        for color in ['white', 'black', 'gray', (80, 110, 160)]:
            result = detector.predict(Image.new('RGB', (640, 480), color))
            self.assertEqual(result['predictions'], [])
            self.assertEqual(result['message'], 'No gesture detected')
            self.assertEqual(result['rejection_reason'], 'No visual content')

    def test_actual_trained_model(self):
        if not Path('models/best.pt').exists():
            self.skipTest('Training has not completed; real-model integration is unverified.')
        detector = Detector(Path('models/best.pt'))
        detector.load()
        image_path = next(Path('data/processed/val').rglob('*.png'))
        with Image.open(image_path) as image:
            output = detector.predict(image, .8)
        self.assertEqual(output['task'], 'classify')
        self.assertEqual(set(output['scores']), set(detector.classes))
        self.assertLessEqual(len(output['predictions']), 1)
        self.assertGreaterEqual(output['latency_ms'], 0)
        for prediction in output['predictions']:
            self.assertIsNone(prediction['bbox'])
            self.assertTrue(0 <= prediction['confidence'] <= 1)
