import sys

import timm.data

sys.path.append('/home/ym2382/projects/DriverEmo')

import argparse
import csv
import tqdm
import torch
import types

# from datasets.distracted_driver import MMDDDataModule
# from lib.easyface.detect_align import FaceDetectAlign
# from mm_train import MMModel
from datasets import MTLDistractedDriverLDM, MTLDataset
from train import ViTDDLM
from torchvision import transforms as T
from pathlib import Path
from torchvision.utils import save_image
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchshow
from timm.data import ImageDataset
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import functools
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import matplotlib
from torchvision import transforms


def main(hparams):
    hparams.output_root = Path(hparams.output_root)
    data_module = MTLDistractedDriverLDM(
        batch_size=1,
        dataset=hparams.dataset,
        pred_list=hparams.pred_list,
        pseudo_label_path=hparams.pseudo_label_path)
    model: ViTDDLM = ViTDDLM.load_from_checkpoint(hparams.ckpt_path)
    model.eval()
    loader = data_module.predict_dataloader()
    dataset: MTLDataset = loader.dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default=r"~/runs/vis")
    parser.add_argument('--ckpt_path', type=str, default=r"~/runs/checkpoints/epoch=17-step=738.ckpt")
    parser.add_argument('--dataset', type=str, default="AUCDD")
    parser.add_argument('--pseudo_label_path', type=str, default="~/runs/pseudo_emo_label")
    parser.add_argument('--pred_list', type=str, default="train_list.csv")
    parser.add_argument('--num_layers', type=int, default=12)
    args = vars(parser.parse_args())
    main(args)
