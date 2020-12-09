import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str


class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, img_size):
        w, h = img_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if max_size is not None:
            min_orig = float(min((w, h)))
            max_orig = float(max((w, h)))

            if max_orig / min_orig * size > max_size:
                size = int(round(max_size * min_orig / max_orig))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)

        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, img, target):
        size = self.get_size(img.size)
        img = F.resize(img, size)
        target = target.resize(img.size)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = target.transpose(0)

        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target


def preset_transform(min_size = 800, max_size= 1333):

    min_size = min_size
    max_size = max_size
    flip = 0.5

    normalize = Normalize(mean=[0.40789654, 0.44719302, 0.47026115], std=[0.28863828, 0.27408164, 0.27809835])

    transform = Compose(
        [Resize(min_size, max_size), RandomHorizontalFlip(flip), ToTensor(), normalize]
    )

    return transform


def preset_transform_voc(min_size = 448, max_size= 448, train=True):
    if train:
        min_size = min_size
        max_size = max_size
        flip = 0.5

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = Compose(
            [Resize(min_size, max_size), RandomHorizontalFlip(flip), ToTensor(), normalize]
        )
    
    else: 
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = Compose(
            [Resize(min_size, max_size), ToTensor(), normalize]
        )

    return transform