import torch

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from timm.data import ImageDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets.sfddd import SFDDDParser, SFDDDMTLParser
from datasets.aucdd import AUCDDParser, AUCDDMTLParser
from torchvision.transforms import InterpolationMode
from timm.data.transforms import RandomResizedCropAndInterpolation

from pathlib import Path
from PIL import Image

from lib.augment import GaussianBlur, Solarization, gray_scale


class DistractedDriverLDM(LightningDataModule):
    def __init__(self,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 dataset: str = "SFDDD",
                 data_root: str = "./datasets",
                 annotation_path: str = "./annotations",
                 train_list: str = "train_list_01.csv",
                 val_list: str = "val_list_01.csv",
                 pred_list: str = "driver_imgs_list.csv",
                 input_size: int = 224,
                 color_jitter: float = 0.3,
                 three_augment: bool = True,
                 src: bool = False,  # simple random crop
                 ):
        super(DistractedDriverLDM, self).__init__()
        self.save_hyperparameters()
        self.data_path = Path(self.hparams.data_root) / self.hparams.dataset
        if self.hparams.dataset == "AUCDD":
            self.data_path = self.data_path / "v2_cam1_cam2_split_by_driver" / "cam1"
        self.annotation_path = Path(self.hparams.annotation_path) / self.hparams.dataset
        self.parser_class = eval(f"{self.hparams.dataset}Parser")
        self.train_transforms = self.build_transform(is_train=True)
        self.eval_transforms = self.build_transform(is_train=False)

    def build_transform(self, is_train, input_size: int = 0):
        img_size = self.hparams.input_size if not input_size else input_size
        interpolation = InterpolationMode.BICUBIC
        remove_random_resized_crop = self.hparams.src
        color_jitter = self.hparams.color_jitter

        scale = (0.08, 1.0)

        if is_train:
            if remove_random_resized_crop:
                primary_tfl = [
                    transforms.Resize(img_size, interpolation=interpolation),
                    transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
                    transforms.RandomHorizontalFlip()
                ]
            else:
                primary_tfl = [
                    RandomResizedCropAndInterpolation(img_size, scale=scale, interpolation=interpolation),
                    transforms.RandomHorizontalFlip()
                ]
            secondary_tfl = []
            if self.hparams.three_augment:
                secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)])]
            if color_jitter is not None and not color_jitter == 0:
                secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
            final_tfl = [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
            return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

        else:
            t = []
            t.append(transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC))
            t.append(transforms.CenterCrop(img_size))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            return transforms.Compose(t)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_list = str(self.annotation_path / self.hparams.train_list)
        train_dataset = ImageDataset(str(self.data_path),
                                     parser=self.parser_class(self.data_path, train_list),
                                     transform=self.train_transforms)
        train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset,
                          sampler=train_sampler,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_list = str(self.annotation_path / self.hparams.val_list)
        val_dataset = ImageDataset(str(self.data_path),
                                   parser=self.parser_class(self.data_path, val_list),
                                   transform=self.eval_transforms)
        val_sampler = SequentialSampler(val_dataset)
        return DataLoader(val_dataset,
                          sampler=val_sampler,
                          batch_size=int(1.5 * self.hparams.batch_size),
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=False)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pred_list = str(self.annotation_path / self.hparams.pred_list)
        pred_dataset = ImageDataset(str(self.data_path),
                                    parser=self.parser_class(self.data_path, pred_list),
                                    transform=self.eval_transforms)
        return DataLoader(pred_dataset,
                          sampler=SequentialSampler(pred_dataset),
                          batch_size=int(1.5 * self.hparams.batch_size),
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()


class MTLDistractedDriverLDM(DistractedDriverLDM):
    def __init__(self,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 dataset: str = "SFDDD",
                 data_root: str = "./datasets",
                 annotation_path: str = "./annotations",
                 pseudo_label_path: str = "./pseudo_emo_label",
                 train_list: str = "train_list_01.csv",
                 val_list: str = "val_list_01.csv",
                 pred_list: str = "driver_imgs_list.csv",
                 input_size_1: int = 224,
                 input_size_2: int = 32,
                 color_jitter: float = 0.3,
                 three_augment: bool = True,
                 src: bool = True,
                 ):
        super(MTLDistractedDriverLDM, self).__init__(batch_size, num_workers, dataset, data_root, annotation_path,
                                                     train_list, val_list, pred_list, input_size_1, color_jitter,
                                                     three_augment, src)
        self.input_size_1, self.input_size_2 = input_size_1, input_size_2
        self.train_transforms_1, self.eval_transforms_1 = self.train_transforms, self.eval_transforms
        self.train_transforms_2 = self.build_transform(is_train=True, input_size=input_size_2)
        self.eval_transforms_2 = self.build_transform(is_train=False, input_size=input_size_2)
        self.pseudo_label_path = Path(pseudo_label_path)
        self.parser_class = eval(f"{self.hparams.dataset}MTLParser")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_list = str(self.annotation_path / self.hparams.train_list)
        parser = self.parser_class(root=self.data_path, img_list=train_list, emo_path=self.pseudo_label_path)
        train_dataset = MTLDataset(self.data_path, parser,
                                   transform=[self.train_transforms, self.train_transforms_2],
                                   input_sizes=[self.hparams.input_size_1, self.hparams.input_size_2])
        train_sampler = RandomSampler(train_dataset)
        return DataLoader(train_dataset,
                          sampler=train_sampler,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_list = str(self.annotation_path / self.hparams.val_list)
        parser = self.parser_class(root=self.data_path, img_list=val_list, emo_path=self.pseudo_label_path)
        val_dataset = MTLDataset(self.data_path, parser,
                                 transform=[self.eval_transforms, self.eval_transforms_2],
                                 input_sizes=[self.hparams.input_size_1, self.hparams.input_size_2])
        val_sampler = SequentialSampler(val_dataset)
        return DataLoader(val_dataset,
                          sampler=val_sampler,
                          batch_size=int(1.5 * self.hparams.batch_size),
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pred_list = str(self.annotation_path / self.hparams.pred_list)
        parser = self.parser_class(root=self.data_path, img_list=pred_list, emo_path=self.pseudo_label_path)
        predict_dataset = MTLDataset(self.data_path, parser,
                                     transform=[self.eval_transforms, self.eval_transforms_2],
                                     input_sizes=[self.hparams.input_size_1, self.hparams.input_size_2])
        val_sampler = SequentialSampler(predict_dataset)
        return DataLoader(predict_dataset,
                          sampler=val_sampler,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=False)


class MTLDataset(ImageDataset):
    NUM_IMG_CHANNELS = 3
    NON_FACE_LABEL = 7

    def __init__(self, root, parser, transform, input_sizes):
        super(MTLDataset, self).__init__(root, parser=parser, transform=transform)
        self.t1, self.t2 = self.transform
        self.input_sizes = input_sizes

    def __getitem__(self, index):
        img_1, target1, img_2, target2 = self.parser[index]
        img_1 = Image.open(img_1).convert('RGB')
        img_1 = self.t1(img_1)
        if img_2 is not None:
            img_2 = Image.open(img_2).convert('RGB')
            img_2 = self.t2(img_2)
            return img_1, target1, img_2, target2
        else:
            img_2 = torch.zeros([MTLDataset.NUM_IMG_CHANNELS, self.input_sizes[1], self.input_sizes[1]])
            return img_1, target1, img_2, MTLDataset.NON_FACE_LABEL
