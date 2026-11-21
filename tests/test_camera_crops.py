"""Sampling and selection contracts, not claims about camera accuracy."""
import unittest
from types import SimpleNamespace

import torch
from ultralytics.utils.metrics import ClassifyMetrics

from training.camera_crops import CameraCropTrainer, MacroGestureValidator
from training.prepare_hand_crops import hand_bounds


class CameraCropTests(unittest.TestCase):
    def test_macro_selection_does_not_hide_minority_class_failures(self):
        validator = MacroGestureValidator.__new__(MacroGestureValidator)
        validator.metrics = ClassifyMetrics()
        validator.targets = [torch.tensor([0] * 100 + [1, 2])]
        validator.pred = [torch.tensor([[0, 1, 2]] * 102)]
        stats = validator.get_stats()
        self.assertGreater(stats['metrics/accuracy_top1'], .98)
        self.assertAlmostEqual(stats['fitness'], 1 / 3)

    def test_sampling_gives_equal_class_mass_and_changes_draws(self):
        class Dataset:
            samples = [(None, label) for label in [0] + [1] * 2 + [2] * 10]
            def __len__(self): return len(self.samples)
            def __getitem__(self, index): return self.samples[index][1]
        dataset = Dataset()
        trainer = SimpleNamespace(args=SimpleNamespace(seed=42), build_dataset=lambda *args: dataset)
        loader = CameraCropTrainer.get_dataloader(trainer, 'unused', 16)
        self.assertEqual(len(loader), 24)
        labels = torch.tensor([sample[1] for sample in dataset.samples])
        for label in range(3):
            self.assertAlmostEqual(float(loader.sampler.weights[labels == label].sum()), 1.)
        self.assertNotEqual(list(loader.sampler), list(loader.sampler))

    def test_crop_geometry_matches_browser_padding_and_clamping(self):
        points = [SimpleNamespace(x=.2 + i * .02, y=.3 + i * .02) for i in range(21)]
        # IEEE floating-point boundaries match the browser's floor/ceil behavior.
        self.assertEqual(hand_bounds(points, 640, 480), (63, 79, 449, 400))
        self.assertIsNone(hand_bounds([], 640, 480))


if __name__ == '__main__':
    unittest.main()
