"""Protect hand pixels, deterministic validation, and no-regression model selection."""
import copy
import importlib.util
import random
import unittest

import numpy as np
from PIL import Image, ImageDraw

from training.background_augmentation import patterned_background, replace_background, white_background_mask
from training.robustness import qualifies


class BackgroundTests(unittest.TestCase):
    def test_background_replacement_preserves_foreground_and_highlights(self):
        source = Image.new('RGB', (100, 100), 'white')
        draw = ImageDraw.Draw(source)
        draw.rectangle((20, 20, 80, 80), fill=(120, 80, 55))
        draw.rectangle((45, 45, 55, 55), fill='white')
        mask = white_background_mask(source)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[50, 50])  # White highlight inside the foreground is retained.
        output = np.array(replace_background(source, Image.new('RGB', source.size, 'blue')))
        np.testing.assert_array_equal(output[~mask], np.array(source)[~mask])
        np.testing.assert_array_equal(output[0, 0], [0, 0, 255])

    def test_real_background_is_not_removed(self):
        source = Image.new('RGB', (100, 100), (180, 180, 180))
        self.assertIsNone(white_background_mask(source))
        self.assertEqual(source.tobytes(), replace_background(source, Image.new('RGB', source.size, 'red')).tobytes())

    def test_pattern_generation_is_reproducible(self):
        first = patterned_background((100, 100), random.Random(42), 'clutter')
        second = patterned_background((100, 100), random.Random(42), 'clutter')
        self.assertEqual(first.tobytes(), second.tobytes())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_validation_preprocessing_is_unchanged(self):
        from training.transforms import background_transform, transform
        source = Image.new('RGB', (60, 100), (100, 80, 55))
        np.testing.assert_array_equal(transform(320, False)(source).numpy(),
                                      background_transform(320, False)(source).numpy())

    def test_promotion_rejects_thumb_regression(self):
        def report(palm, thumb):
            metrics = {'per_class': {'open_palm': {'precision':1., 'recall':palm},
                                     'thumbs_up': {'precision':1., 'recall':thumb}}}
            return dict(manifest_sha256='same', confidence=.8, split='val',
                        conditions={'background_clutter':metrics}, clean_views={'original':{'accuracy':1.}},
                        stability={'mean_raw_label_changes':0., 'stable_correct_after_warmup':1.})
        baseline = report(.5, 1.)
        self.assertTrue(qualifies(baseline, report(1., 1.))[0])
        self.assertFalse(qualifies(baseline, report(1., .9))[0])
        self.assertFalse(qualifies(baseline, baseline)[0])
        invalid = copy.deepcopy(report(1., 1.)); invalid['split'] = 'test'
        self.assertFalse(qualifies(baseline, invalid)[0])
