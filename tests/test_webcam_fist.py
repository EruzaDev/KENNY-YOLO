import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training.prepare_webcam_fist import prepare, pixel_hash


class WebcamFistPreparationTests(unittest.TestCase):
    def test_webcam_sessions_are_train_only_and_inherited_test_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.full((128, 128, 3), 255, dtype=np.uint8)
            pixels[20:, 40:90] = [130, 85, 55]
            hand = Image.fromarray(pixels)
            source = root / 'hand.png'
            hand.save(source)
            bg = root / 'background.jpg'
            Image.new('RGB', (100, 160), 'darkgreen').save(bg)
            rows = [dict(label='closed_fist', split=split, group=index, sha256=pixel_hash(hand),
                         view='isolated', processed=str(source))
                    for index, split in enumerate(['train', 'val', 'test'])]
            base = root / 'manifest.json'
            base.write_text(json.dumps(dict(manifest=rows, classes=['closed_fist'],
                background_photos=[dict(path=str(bg), sha256=hashlib.sha256(bg.read_bytes()).hexdigest())])))
            examples = []
            for index in range(2):
                path = root / f'closed_fist__session_same-session__{index}__hand.jpg'
                Image.new('RGB', (80, 90), (90 + index, 65, 50)).save(path)
                examples.append(path)
            result = prepare(base, examples, root / 'prepared', root / 'preview.jpg')
            webcam = [r for r in result['manifest'] if r['view'] == 'webcam_hand']
            self.assertEqual(len(webcam), 2)
            self.assertEqual({r['split'] for r in webcam}, {'train'})
            self.assertEqual(len({r['group'] for r in webcam}), 1)
            self.assertNotIn(webcam[0]['group'], {r['group'] for r in result['manifest'] if r['split'] != 'train'})
            test = [r for r in result['manifest'] if r['split'] == 'test']
            self.assertEqual(len(test), 1)
            with Image.open(test[0]['processed']) as image:
                np.testing.assert_array_equal(np.asarray(image), pixels)
            self.assertTrue(any(r['view'] == 'hand_crop' and r['split'] == 'val' for r in result['manifest']))
            for row in result['manifest']:
                with Image.open(row['processed']) as image:
                    self.assertEqual(row['processed_sha256'], pixel_hash(image))
            with self.assertRaises(ValueError):
                prepare(base, examples, root / 'prepared', root / 'preview.jpg')


if __name__ == '__main__':
    unittest.main()
