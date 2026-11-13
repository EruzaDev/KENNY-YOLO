import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from training.prepare_dataset import CLASSES, audit, fingerprint, prepare


class DatasetTests(unittest.TestCase):
    def test_class_mapping(self):
        self.assertEqual(CLASSES, ('open_palm', 'thumbs_up'))

    def test_saved_dataset_integrity_and_leakage(self):
        manifest = Path('data/processed/manifest.json')
        if not manifest.exists():
            self.skipTest('Prepare supplied dataset first.')
        report = json.loads(manifest.read_text())
        records = report['manifest']
        self.assertEqual(len(records), len({r['sha256'] for r in records}))
        groups = {}
        for record in records:
            groups.setdefault(record['group'], set()).add(record['split'])
            digest, _ = fingerprint(Path(record['processed']))
            self.assertEqual(digest, record['sha256'])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        for i, a in enumerate(records):
            for b in records[:i]:
                if a['split'] != b['split']:
                    self.assertGreater((a['phash'] ^ b['phash']).bit_count(), report['near_duplicate_distance'])
        for split in ['train', 'val', 'test']:
            self.assertEqual({r['label'] for r in records if r['split'] == split}, set(CLASSES))

    def test_corruption_and_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for label in CLASSES:
                (raw / label).mkdir()
                Image.new('RGB', (40, 40), 'white').save(raw / label / 'same.png')
            (raw / CLASSES[0] / 'broken.png').write_bytes(b'broken')
            report = audit(raw)
            self.assertEqual(len(report['errors']), 1)
            self.assertEqual(len(report['label_conflicts']), 1)
            with self.assertRaises(ValueError):
                prepare(raw, raw.parent / 'unused-output', raw / 'report.json')
