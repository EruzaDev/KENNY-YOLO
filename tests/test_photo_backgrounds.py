import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training.prepare_photo_backgrounds import prepare


class PhotoBackgroundTests(unittest.TestCase):
    def test_only_training_changes_and_foreground_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.full((64, 64, 3), 255, dtype=np.uint8)
            pixels[20:44, 20:44] = [140, 85, 50]
            source = root / 'hand.png'
            Image.fromarray(pixels).save(source)
            background = root / 'background.jpg'
            Image.new('RGB', (100, 140), (20, 40, 70)).save(background)
            rows = [dict(processed=str(source), split=split, label='open_palm',
                         view='isolated', group=i, sha256='example')
                    for i, split in enumerate(['train', 'val', 'test'])]
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(classes=['open_palm'], manifest=rows)))
            result = prepare(manifest, [background], root / 'output', root / 'preview.jpg')
            self.assertEqual([r['split'] for r in result['manifest']].count('train'), 2)
            for record in result['manifest']:
                with Image.open(record['processed']) as image:
                    actual = np.asarray(image)
                np.testing.assert_array_equal(actual[20:44, 20:44], pixels[20:44, 20:44])
                if record['split'] != 'train':
                    np.testing.assert_array_equal(actual, pixels)
            self.assertEqual({r['group'] for r in result['manifest'] if r['split'] == 'train'}, {0})
            with self.assertRaises(ValueError):
                prepare(manifest, [background], root / 'output', root / 'preview.jpg')


if __name__ == '__main__':
    unittest.main()
