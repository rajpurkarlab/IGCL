"""
Preprocessing utilities for image data.
"""

from PIL import ImageFilter
from PIL.ImageOps import equalize
import random
from torchvision import transforms
from torchvision.transforms.functional import pad

class EqualizeHist():
    """
    Apply histogram equalization to the input.
    """
    def __call__(self, x):
        return equalize(x)
    
    def __repr__(self):
        return self.__class__.__name__

class GaussianBlur():
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class PadToSquare():
    """
    Apply zero padding to the shorter edge of the image such that the output is a square.
    """
    def __call__(self, x):
        w, h = x.size
        diff = abs(w - h)
        p1 = diff // 2
        p2 = p1
        if diff % 2 == 1:
            p2 += 1
        if w > h:
            return pad(x, (0, p1, 0, p2))
        else:
            return pad(x, (p1, 0, p2, 0))

    def __repr__(self):
        return self.__class__.__name__

class RandomCropWithScale():
    """
    Apply a random crop to the input image, with cropping scale sampled from the given range.
    """
    def __init__(self, scale=(0.8, 1.0)):
        assert len(scale) == 2, "Min and max of scale must be both provided."
        assert scale[0] >= 0.0
        assert scale[1] <= 1.0
        self.scale = scale
    
    def __call__(self, x):
        w, h = x.size
        _scale = random.uniform(*self.scale)
        w_out = int(w * _scale)
        h_out = int(h * _scale)
        left = random.randrange(w - w_out)
        top = random.randrange(h - h_out)
        return transforms.functional.crop(x, top, left, h_out, w_out)
    
    def __repr__(self):
        return self.__class__.__name__

def augmentations(p=0.5):
    return transforms.Compose([
        transforms.Resize(320),
        transforms.RandomApply([RandomCropWithScale(scale=(0.9,1.0))], p=p),
        transforms.RandomHorizontalFlip(p=p),
        transforms.RandomApply([transforms.RandomAffine(
            degrees=(-15, 15),
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fillcolor=0)],
            p=p),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=p),
        transforms.RandomApply([GaussianBlur(sigma=(0.1,2.0))], p=p)
    ])

def resize_and_normalize(imsize=224):
    """
    Pad and resize the input image into [imsize, imsize]; convert input into tensor, and do ImageNet-based normalization.
    Note that:
        1. if input is np.array image, the order of channels has to be [H,W,C], and after conversion
            the channels will be [C,H,W] in the output tensor;
        2. Input must have range of [0,255], and will be converted to [0,1];
        3. This normalization works for DenseNet and ResNet, but not for some other models including
            inception-v3, etc.
    """
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def pretrain_augmentations(imsize=224, p=0.8):
    return transforms.Compose([
        transforms.Resize(int(imsize/0.6)),
        # apply more aggressive cropping for pretraining
        transforms.RandomApply([RandomCropWithScale(scale=(0.6,1.0))], p=p),
        transforms.RandomHorizontalFlip(),
        # slightly more aggressive transformation
        transforms.RandomApply([transforms.RandomAffine(
            degrees=(-20, 20),
            translate=(0.1, 0.1),
            scale=(0.95, 1.05),
            fillcolor=0)],
            p=p),
        # more aggressive color jittering
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=p),
        # a very small amount of blur for pretraining
        transforms.RandomApply([GaussianBlur(sigma=(0.1,3.0))], p=p)
    ])