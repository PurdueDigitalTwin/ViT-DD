import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchshow
import types

from pathlib import Path
from tqdm import tqdm

from datasets import MTLDistractedDriverLDM, MTLDataset
from lib.utils import unnormalize
from train import ViTDDLM


def forward(self, x, attn_maps: list):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Save attention maps here
    tensor_attn = attn[0, 0, :, :]
    attn_maps.append(torch.clone(tensor_attn).cpu().numpy())
    ########

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def vis_attn(mat: np.ndarray, output_path: Path, img=None, heatmap_kwargs=None, savefig_kwargs=None):
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    if savefig_kwargs is None:
        savefig_kwargs = {}
    # if not output_path.exists():
    #     output_path.mkdir(parents=True)

    fig, ax = plt.subplots(1)
    heatmap = sns.heatmap(mat, ax=ax, **heatmap_kwargs)
    if img is not None:
        ax.imshow(img, aspect=heatmap.get_aspect(), extent=heatmap.get_xlim() + heatmap.get_ylim(), zorder=1)
    fig.savefig(output_path, dpi=200, **savefig_kwargs)
    ax.clear()


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
    attn_maps = []
    for i in range(hparams.num_layers):
        attn_layer: torch.nn.Module = model.model.backbone.blocks[i].attn
        forward_fn = functools.partial(forward, attn_maps=attn_maps)
        attn_layer.forward = types.MethodType(forward_fn, attn_layer)

    image_path = 'train/c0/1316.jpg'
    it = iter(loader)
    for image_idx in tqdm(range(len(dataset))):
        data_item = next(it)
        cur_path = str(dataset.filename(image_idx, absolute=False)).strip()
        if cur_path == image_path:
            split, cls, number = cur_path.split('/')
            number = number.split('.')[0]
            output_path = hparams.output_root / image_path.split('.')[0]
            # if not output_path.exists():
            #     output_path.mkdir(parents=True)
            for subdir in ['driver', 'face', 'classes']:
                subdir = output_path / subdir
                if not subdir.exists():
                    subdir.mkdir(parents=True)

            driver_img, distraction_target, face_img, emotion_target = data_item
            torchshow.save(driver_img, str(output_path / f'{cls}_{number}_driver.jpg'))
            torchshow.save(face_img, str(output_path / f'{cls}_{number}_face.jpg'))
            with torch.no_grad():
                distraction_pred, emotion_pred = model(driver_img, face_img)
                distraction_pred = F.softmax(distraction_pred[0], dim=0)
                emotion_pred = F.softmax(emotion_pred[0], dim=0)
                print(distraction_pred, emotion_pred)
            driver_img = unnormalize(driver_img[0]).permute(1, 2, 0).cpu().numpy()
            face_img = unnormalize(face_img[0]).permute(1, 2, 0).cpu().numpy()

            for mp in attn_maps:
                # Attentions between the distraction token and visual tokens
                for i in range(hparams.num_layers):
                    mat = mp[0, :]
                    heatmap_kwargs = dict(cmap="jet", zorder=2, cbar=False, xticklabels=False, yticklabels=False,
                                          alpha=0.5)
                    savefig_kwargs = dict(bbox_inches='tight', pad_inches=0.01)
                    mat_driver = mat[2: 2 + 196].reshape([14, 14])
                    mat_face = mat[198:198 + 4].reshape([2, 2])
                    vis_attn(mat_driver, output_path / 'driver' / f'attn_driver_{i}.png',
                             driver_img, heatmap_kwargs, savefig_kwargs)
                    vis_attn(mat_face, output_path / 'face' / f'attn_face_{i}.png',
                             face_img, heatmap_kwargs, savefig_kwargs)

                    # Attention maps between class tokens
                    mat = mp[:2, :2]
                    output_path = output_path / 'classes' / f'attn_classes_{i}.png'
                    categories = ['Distraction', 'Emotion']
                    heatmap_kwargs = dict(annot=False, xticklabels=categories, yticklabels=categories, cmap='mako')
                    vis_attn(mat, output_path, heatmap_kwargs=heatmap_kwargs)

            attn_maps.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default=r"~/runs/vis")
    parser.add_argument('--ckpt_path', type=str, default=r"~/runs/checkpoints/epoch=17-step=738.ckpt")
    parser.add_argument('--dataset', type=str, default="AUCDD", choices=["AUCDD"])
    parser.add_argument('--pseudo_label_path', type=str, default="~/runs/pseudo_emo_label")
    parser.add_argument('--pred_list', type=str, default="train_list.csv")
    parser.add_argument('--num_layers', type=int, default=12)
    args = vars(parser.parse_args())
    main(args)
