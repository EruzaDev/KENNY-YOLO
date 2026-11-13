"""Full-frame classification transforms; training augmentation never touches val/test."""
import random
from copy import copy

import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class Letterbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), color='white',
                            method=Image.Resampling.BILINEAR)


class MildNoise:
    def __call__(self, tensor):
        if random.random() < .15:
            return (tensor + torch.randn_like(tensor) * .01).clamp(0, 1)
        return tensor


def transform(size: int, augment: bool = False):
    steps = [Letterbox(size)]
    if augment:
        steps += [T.RandomHorizontalFlip(.5),
                  T.RandomAffine(8, translate=(.04, .04), scale=(.94, 1.04), fill=255),
                  T.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
                  T.RandomApply([T.GaussianBlur(3, sigma=(.1, .6))], p=.1)]
    steps += [T.ToTensor()]
    if augment:
        steps += [MildNoise()]
    steps += [T.Normalize(mean=(0, 0, 0), std=(1, 1, 1))]
    return T.Compose(steps)


class GestureDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        super().__init__(root=root, args=args, augment=augment, prefix=prefix)
        self.torch_transforms = transform(args.imgsz, augment)


class GestureValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return GestureDataset(root=img_path, args=self.args, augment=False, prefix='val')


class GestureTrainer(ClassificationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        self.transferred_classes = []
        if weights is not None:
            self.transferred_classes = transfer_class_rows(model, weights, self.data['names'])
        return model

    def build_dataset(self, img_path, mode='train', batch=None):
        return GestureDataset(root=img_path, args=self.args, augment=mode == 'train', prefix=mode)

    def get_validator(self):
        self.loss_names = ['loss']
        return GestureValidator(self.test_loader, self.save_dir, args=copy(self.args),
                                _callbacks=self.callbacks)


def background_transform(size: int, augment: bool = False):
    """Same deterministic validation preprocessing, stronger realistic training diversity."""
    if not augment:
        return transform(size, False)
    from training.background_augmentation import BackgroundVariation
    return T.Compose([
        Letterbox(size), BackgroundVariation(), T.RandomHorizontalFlip(.5),
        T.RandomAffine(12, translate=(.03, .03), scale=(.86, 1.0), fill=255),
        T.ColorJitter(brightness=.3, contrast=.25, saturation=.15, hue=.015),
        T.RandomApply([T.GaussianBlur(3, sigma=(.1, .8))], p=.15),
        T.ToTensor(), MildNoise(),
        T.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ])


class BackgroundGestureDataset(GestureDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        super().__init__(root, args, augment, prefix)
        self.torch_transforms = background_transform(args.imgsz, augment)


class BackgroundGestureTrainer(GestureTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return BackgroundGestureDataset(img_path, self.args, augment=mode == 'train', prefix=mode)


def transfer_class_rows(model, previous, names):
    """Retain existing class rows by name when the classification head grows."""
    old_head, new_head = previous.model[-1].linear, model.model[-1].linear
    if old_head.out_features == new_head.out_features:
        return []
    old_names = previous.names
    copied = []
    with torch.no_grad():
        for old_index, label in old_names.items():
            for new_index, new_label in names.items():
                if label == new_label:
                    new_head.weight[new_index].copy_(old_head.weight[old_index])
                    new_head.bias[new_index].copy_(old_head.bias[old_index])
                    copied.append(label)
    return copied


class BalancedBackgroundGestureDataset(BackgroundGestureDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        super().__init__(root, args, augment, prefix)
        if augment:
            from collections import defaultdict
            groups = defaultdict(list)
            for index, sample in enumerate(self.samples):
                groups[sample[1]].append(index)
            target = max(map(len, groups.values()))
            # Training-only oversampling; no files or extra independent captures.
            indices = [samples[i % len(samples)]
                       for samples in groups.values() for i in range(target)]
            self.samples = [self.samples[i] for i in indices]
            if self.cache_ram:
                # Cache positions must follow the repeated sample positions too.
                self.img_cache = [self.img_cache[i] for i in indices]


class BalancedBackgroundGestureTrainer(BackgroundGestureTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return BalancedBackgroundGestureDataset(img_path, self.args, augment=mode == 'train', prefix=mode)


class PalmWeightedLoss:
    """Modest Open Palm emphasis after class-balanced sampling, with standard CE metrics."""
    def __init__(self, palm_weight=1.25):
        self.palm_weight = palm_weight

    def __call__(self, predictions, batch):
        import torch.nn.functional as functional
        predictions = predictions[1] if isinstance(predictions, (tuple, list)) else predictions
        weights = predictions.new_tensor([1., self.palm_weight, 1.])
        loss = functional.cross_entropy(predictions, batch['cls'], weight=weights)
        return loss, {'loss':loss.detach()}


class ResolutionBalancedDataset(BalancedBackgroundGestureDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        super().__init__(root, args, augment, prefix)
        if augment:
            from training.background_augmentation import ResolutionVariation
            # Insert before conversion to tensor; all existing augmentations remain.
            self.torch_transforms.transforms.insert(5, ResolutionVariation())


class ResolutionBalancedTrainer(BalancedBackgroundGestureTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return ResolutionBalancedDataset(img_path, self.args, augment=mode == 'train', prefix=mode)

    def set_model_attributes(self):
        super().set_model_attributes()
        if list(self.data['names'].values()) != ['closed_fist','open_palm','thumbs_up']:
            raise ValueError('Palm-weighted profile requires the canonical three-class order.')
        self.model.criterion = PalmWeightedLoss()


class StableBackbonePalmTrainer(ResolutionBalancedTrainer):
    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.criterion = PalmWeightedLoss(2.)

    def preprocess_batch(self, batch):
        # Freezing parameters alone does not freeze BatchNorm running statistics.
        # Keep the inherited extractor's statistics fixed as well as its weights.
        for block in self.model.model[:9]:
            block.eval()
        return super().preprocess_batch(batch)


class FistThumbDiscriminationLoss(PalmWeightedLoss):
    """Retain three-class supervision and emphasize the difficult fist/thumb pair."""
    def __call__(self, predictions, batch):
        import torch.nn.functional as functional
        logits = predictions[1] if isinstance(predictions, (tuple, list)) else predictions
        loss, _ = super().__call__(predictions, batch)
        labels = batch['cls']
        selected = (labels == 0) | (labels == 2)
        if selected.any():
            pair_logits = logits[selected][:, [0, 2]]
            pair_targets = (labels[selected] == 2).long()
            loss = loss + .3*functional.cross_entropy(pair_logits, pair_targets)
        return loss, {'loss':loss.detach()}


class HandFocusedDataset(ResolutionBalancedDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        super().__init__(root, args, augment, prefix)
        if augment:
            from training.hand_crop import HandCropVariation
            self.torch_transforms.transforms.insert(0, HandCropVariation())


class FistThumbTrainer(StableBackbonePalmTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return HandFocusedDataset(img_path, self.args, augment=mode == 'train', prefix=mode)

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.criterion = FistThumbDiscriminationLoss(2.)
