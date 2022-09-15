import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

_mean = torch.tensor(list(IMAGENET_DEFAULT_MEAN))
_std = torch.tensor(list(IMAGENET_DEFAULT_STD))
unnormalize = transforms.Normalize((-_mean / _std).tolist(), (1.0 / _std).tolist())
