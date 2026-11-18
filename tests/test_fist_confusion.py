"""Targeted fist/thumb training and abstention contracts; no accuracy claims from fixtures."""
import importlib.util
import unittest

from PIL import Image, ImageDraw
from app.services.inference import Stabilizer, check_thumb_evidence
from training.hand_crop import hand_focused_crop


class FistConfusionTests(unittest.TestCase):
    def test_contradictions_abstain_without_relabeling_and_clear_history(self):
        stabilizer = Stabilizer()
        for label, conflict in [('thumbs_up', 'thumb_folded')]:
            frame = dict(predictions=[dict(class_name=label, confidence=.99)], scores={label:.99})
            self.assertEqual(stabilizer.apply(frame, 1)['stable_gesture'], label)
            rejected = check_thumb_evidence(frame, conflict)
            self.assertEqual(rejected['predictions'], [])
            self.assertEqual(rejected['scores'], frame['scores'])
            self.assertIsNone(stabilizer.apply(rejected, 3)['stable_gesture'])
            self.assertEqual(list(stabilizer.history), [])
            self.assertIs(check_thumb_evidence(frame), frame)
        palm = dict(predictions=[dict(class_name='open_palm')])
        self.assertIs(check_thumb_evidence(palm, 'thumb_folded'), palm)
        ok = dict(predictions=[dict(class_name='ok')])
        for hint in ['thumb_folded', 'thumb_extended']:
            self.assertIs(check_thumb_evidence(ok, hint), ok)
        with self.assertRaises(ValueError):
            check_thumb_evidence(palm, 'invalid')

    def test_selection_rejects_unproven_improvement_and_palm_regression(self):
        import copy
        import json
        from pathlib import Path
        from training.select_three_class import fist_thumb_failures
        report_path = Path('outputs/fist_confusion/baseline_val.json')
        if not report_path.exists():
            self.skipTest('Validation comparison unavailable')
        baseline = json.loads(report_path.read_text())
        self.assertTrue(any('No measured reduction' in x for x in fist_thumb_failures(baseline, baseline)))
        candidate = copy.deepcopy(baseline)
        candidate['conditions']['clean']['per_class']['open_palm']['recall'] = .5
        self.assertTrue(any('open_palm recall decreased' in x for x in fist_thumb_failures(baseline, candidate)))

    def test_crop_preserves_distal_hand_and_skips_nonwhite_background(self):
        image = Image.new('RGB', (200, 300), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((85, 90, 115, 299), fill=(120, 70, 40))
        draw.rectangle((60, 40, 140, 140), fill=(120, 70, 40))
        crop = hand_focused_crop(image)
        self.assertLess(crop.height, image.height)
        self.assertEqual(crop.getbbox(), (0, 0, crop.width, crop.height))
        # Distal silhouette is retained with padding and the wrist shortened.
        self.assertIn((120, 70, 40), crop.getdata())
        colored = Image.new('RGB', (120, 80), (40, 70, 90))
        self.assertEqual(hand_focused_crop(colored).tobytes(), colored.tobytes())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_pair_loss_targets_fist_thumb_and_leaves_palm_supervision_unchanged(self):
        import torch
        from training.transforms import FistThumbDiscriminationLoss, PalmWeightedLoss
        logits = torch.tensor([[.2, -.4, .3], [.1, .9, .2], [.4, .2, .3]], requires_grad=True)
        batch = {'cls':torch.tensor([0, 1, 2])}
        base, _ = PalmWeightedLoss(2.)(logits, batch)
        total, _ = FistThumbDiscriminationLoss(2.)(logits, batch)
        self.assertGreater(total.item(), base.item())
        extra = torch.autograd.grad(total-base, logits)[0]
        self.assertEqual(extra[1].abs().sum().item(), 0.)
        self.assertEqual(extra[:, 1].abs().sum().item(), 0.)
        self.assertGreater(extra[:, [0, 2]].abs().sum().item(), 0.)
