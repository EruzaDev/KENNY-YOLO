"""Verify inherited capture separation and validation-only model selection."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from training.compare_models import is_improvement, score
from training.prepare_dataset import fingerprint
from training.prepare_originals import original_number


class ImprovementTests(unittest.TestCase):
    def test_source_number_mapping(self):
        self.assertIsNone(original_number(1,'thumbs_up'))
        self.assertEqual(original_number(12,'thumbs_up'),11)
        self.assertEqual(original_number(13,'thumbs_up'),300)
        self.assertEqual(original_number(13,'open_palm'),12)
        self.assertEqual(original_number(14,'open_palm'),300)

    def test_originals_preserve_existing_groups(self):
        path=Path('data/processed_with_originals/manifest.json')
        if not path.exists():self.skipTest('Expanded data has not been prepared.')
        report=json.loads(path.read_text())
        base=json.loads(Path(report['source_manifest']).read_text())
        inherited={r['sha256']:r for r in base['manifest']}
        groups={}
        rows=report['manifest']
        for row in rows:
            groups.setdefault(row['group'],set()).add(row['split'])
            capture=row.get('capture_sha256',row['sha256'])
            self.assertEqual(row['split'],inherited[capture]['split'])
            self.assertEqual(row['group'],inherited[capture]['group'])
            self.assertEqual(fingerprint(Path(row['processed']))[0],row['processed_sha256'])
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))
        for i,a in enumerate(rows):
            for b in rows[:i]:
                if a['split'] != b['split']:
                    self.assertGreater((a['phash']^b['phash']).bit_count(),8)
        self.assertEqual(len(rows),2*len(inherited))

    def test_rejects_candidate_with_worse_accuracy(self):
        def metrics(accuracy,nll):
            return dict(accuracy=accuracy,all_views_correct_group_rate=accuracy,group_nll=nll,
                        views={'original':{'accuracy':accuracy}},
                        per_class={'open_palm':{'precision':accuracy,'recall':accuracy}})
        self.assertTrue(is_improvement(metrics(.8,.3),metrics(.9,.2)))
        self.assertFalse(is_improvement(metrics(.9,.3),metrics(.8,.1)))
        self.assertFalse(is_improvement(metrics(.9,.1),metrics(.9,.2)))

    def test_test_record_cannot_enter_selection(self):
        with patch('training.compare_models.Detector') as detector:
            with self.assertRaises(ValueError):
                score(Path('unused.pt'),[{'split':'test'}],.8)
            detector.return_value.predict.assert_not_called()
