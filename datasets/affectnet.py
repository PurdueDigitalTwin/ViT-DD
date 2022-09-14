from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from timm.data import ImageDataset, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from pathlib import Path
from lib.augment import new_data_aug_generator


class AffectNetDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 data_root: str = "./datasets",
                 input_size: int = 224,
                 color_jitter: float = 0.3,
                 three_augment: bool = True,
                 src: bool = False,  # simple random crop
                 ):
        super(AffectNetDataModule, self).__init__()
        self.save_hyperparameters()
        self.data_path = Path(self.hparams.data_root) / "AffectNet" / "imgs"
        self.train_transforms = self.build_transform(is_train=True)
        self.eval_transforms = self.build_transform(is_train=False)
        if self.hparams.three_augment:
            self.train_transforms = new_data_aug_generator(self.hparams)
        self.class_map = {str(i): i for i in range(8)}

    def build_transform(self, is_train):
        resize_im = self.hparams.input_size > 32
        if is_train:
            transform = create_transform(input_size=self.hparams.input_size,
                                         is_training=True,
                                         color_jitter=self.hparams.color_jitter)
            if not resize_im:  # replace RandomResizedCropAndInterpolation with RandomCrop
                transform.transforms[0] = transforms.RandomCrop(self.hparams.input_size, padding=4)
            return transform
        t = []
        if resize_im:
            # int((256 / 224) * args.input_size) (deit crop ratio (256 / 224), deit III crop ratio 1.0)
            size = int((1.0) * self.hparams.input_size)
            t.append(transforms.Resize(size,
                                       interpolation=InterpolationMode.BICUBIC, ))  # to maintain same ratio w.r.t. 224 images
            t.append(transforms.CenterCrop(self.hparams.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = ImageDataset(str(self.data_path / "train"), transform=self.train_transforms,
                                     class_map=self.class_map)
        train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset,
                          sampler=train_sampler,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset = ImageDataset(str(self.data_path / "val"), transform=self.eval_transforms,
                                   class_map=self.class_map)
        val_sampler = SequentialSampler(val_dataset)
        return DataLoader(val_dataset,
                          sampler=val_sampler,
                          batch_size=int(1.5 * self.hparams.batch_size),
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=False)
