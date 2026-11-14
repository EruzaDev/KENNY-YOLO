import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.image_sources import resolve_source, source_images
from training.organize_images import organize


class ImageOrganizationTests(unittest.TestCase):
    def test_moves_are_lossless_cataloged_and_historical_paths_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['IMG_background.jpg', 'closed_fist__session_example__123__hand.jpg',
                     'closed_fist__session_example__124__full.jpg']
            for name in names:
                (root / name).write_bytes(name.encode())
            unrelated = root / 'unrelated.jpg'
            unrelated.write_bytes(b'preserve me')
            catalog = organize(root)
            self.assertEqual(len(catalog['images']), 3)
            for row in catalog['images']:
                self.assertFalse((root / row['original_path']).exists())
                moved = resolve_source(row['original_path'], root)
                self.assertEqual(moved, root / row['path'])
                self.assertEqual(hashlib.sha256(moved.read_bytes()).hexdigest(), row['sha256'])
                self.assertEqual(resolve_source(root / row['original_path'], root), moved)
            self.assertEqual(len(source_images('background', root)), 1)
            self.assertEqual(len(source_images('webcam_hand', root)), 1)
            self.assertEqual(unrelated.read_bytes(), b'preserve me')
            self.assertEqual(len(organize(root)['images']), 3)
            self.assertEqual(json.loads((root / 'assets/images/catalog.json').read_text()), catalog)

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'IMG_photo.jpg'
            original.write_bytes(b'new')
            target = root / 'assets/images/backgrounds/IMG_photo.jpg'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'old')
            with self.assertRaises(ValueError):
                organize(root)
            self.assertEqual(original.read_bytes(), b'new')
            self.assertEqual(target.read_bytes(), b'old')


if __name__ == '__main__':
    unittest.main()
