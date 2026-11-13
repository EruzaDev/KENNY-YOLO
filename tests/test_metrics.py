import unittest
from training.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_abstentions_and_class_errors(self):
        labels = ['open_palm', 'thumbs_up', 'unknown']
        matrix, report = classification_metrics(
            ['open_palm', 'open_palm', 'thumbs_up', 'thumbs_up'],
            ['open_palm', 'unknown', 'open_palm', 'thumbs_up'], labels)
        self.assertEqual(matrix.tolist(), [[1,0,1], [1,1,0], [0,0,0]])
        self.assertEqual(report['accuracy'], .5)
        self.assertEqual(report['open_palm']['precision'], .5)
        self.assertEqual(report['open_palm']['recall'], .5)
        self.assertAlmostEqual(report['thumbs_up']['f1-score'], 2/3)
        self.assertEqual(report['unknown']['support'], 0)
        with self.assertRaises(ValueError):
            classification_metrics([], [], labels)
