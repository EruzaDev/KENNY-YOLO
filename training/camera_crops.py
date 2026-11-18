"""Class-balanced crop training with real photo backgrounds and macro validation."""
import random
from copy import copy
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from ultralytics.data.build import InfiniteDataLoader

from training.background_augmentation import replace_background, white_background_mask
from training.transforms import GestureDataset, GestureTrainer, GestureValidator, transform


class CameraBackground:
    def __init__(self):
        self.photos = []
        for path in sorted(Path('assets/images/backgrounds').glob('*.jpg')):
            with Image.open(path) as im:
                photo = im.convert('RGB')
                photo.thumbnail((640, 640))
                self.photos.append(photo.copy())
        if not self.photos:
            raise ValueError('Real background photos are required for crop training.')

    def __call__(self, image):
        if random.random() < .1 or white_background_mask(image) is None:
            return image
        background = ImageOps.fit(random.choice(self.photos), image.size,
                                  centering=(random.random(), random.random()))
        return replace_background(image, background)


class CameraCropDataset(GestureDataset):
    def __init__(self, root, args, augment=False, prefix=''):
        super().__init__(root, args, augment, prefix)
        if augment:
            self.torch_transforms = T.Compose([CameraBackground(), transform(args.imgsz, True)])


class MacroGestureValidator(GestureValidator):
    def get_stats(self):
        stats = super().get_stats()
        target = torch.cat(self.targets).reshape(-1)
        predicted = torch.cat(self.pred)[:, 0]
        recall = [(predicted[target == label] == label).float().mean().item()
                  for label in target.unique()]
        stats['metrics/balanced_accuracy'] = sum(recall) / len(recall)
        stats['fitness'] = stats['metrics/balanced_accuracy']
        return stats


class CameraCropTrainer(GestureTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return CameraCropDataset(img_path, self.args, augment=mode == 'train', prefix=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        dataset = self.build_dataset(dataset_path, mode)
        counts = torch.bincount(torch.tensor([sample[1] for sample in dataset.samples]))
        weights = torch.tensor([1. / counts[sample[1]] for sample in dataset.samples])
        # Draw a fresh balanced sample each epoch, rather than repeating tiny classes 58 times.
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=384, replacement=True,
            generator=torch.Generator().manual_seed(self.args.seed))
        return InfiniteDataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=False)

    def get_validator(self):
        self.loss_names = ['loss']
        return MacroGestureValidator(self.test_loader, self.save_dir, args=copy(self.args),
                                     _callbacks=self.callbacks)
