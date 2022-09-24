import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import AffectNetDataModule, DistractedDriverLDM, MTLDistractedDriverLDM
from models import ViTDD

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from torchmetrics import Accuracy, ConfusionMatrix
from torch.optim import Optimizer

from timm.data import Mixup
from timm.models import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.scheduler.scheduler import Scheduler

from pathlib import Path
from typing import Optional


class VisionTransformerLM(LightningModule):
    def __init__(self,
                 batch_size: int = 256,
                 num_classes: int = 10,
                 epochs: int = 100,
                 attn_only: bool = True,
                 smoothing: float = 0.0,  # Label smoothing
                 vis_path: str = "./runs/vis",

                 # Model parameters
                 model: str = "deit3_small_patch16_224",  # Name of model to train
                 input_size: int = 224,  # images input size
                 drop: float = 0.0,  # Dropout rate
                 drop_path: float = 0.05,  # Drop path rate

                 # Optimizer parameters
                 opt: str = "adamw",
                 weight_decay: float = 0.05,

                 # Learning rate schedule parameters
                 sched: str = "cosine",
                 lr: float = 4e-3,
                 warmup_lr: float = 1e-6,
                 min_lr: float = 1e-5,
                 warmup_epochs: int = 5,  # epochs to warmup LR, if scheduler supports
                 cooldown_epochs: int = 0,  # epochs to cooldown LR at min_lr, after cyclic schedule ends

                 # Mixup parameters
                 mixup: float = 0.8,  # mixup alpha, mixup enabled if > 0
                 cutmix: float = 1.0,  # cutmix alpha, cutmix enabled if > 0.
                 mixup_prob: float = 1.0,  # Prob of performing mixup or cutmix when either/both is enabled
                 mixup_switch_prob: float = 0.5,  # Prob of switching to cutmix when both mixup and cutmix enabled
                 mixup_mode: str = "batch",  # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
                 ):
        super(VisionTransformerLM, self).__init__()
        self.save_hyperparameters()

        self.model: torch.nn.Module = create_model(
            self.hparams.model,
            pretrained=True,
            num_classes=self.hparams.num_classes,
            drop_rate=self.hparams.drop,
            drop_path_rate=self.hparams.drop_path,
            drop_block_rate=None,
            img_size=self.hparams.input_size
        )

        self._init_mixup()
        self._init_frozen_params()
        self.train_criterion = torch.nn.CrossEntropyLoss()
        self.valid_criterion = torch.nn.CrossEntropyLoss()
        self.valid_acc = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes, normalize='true')

    def _init_mixup(self):
        self.mixup_fn = None
        mixup_active = self.hparams.mixup > 0 or self.hparams.cutmix > 0.
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=self.hparams.mixup,
                cutmix_alpha=self.hparams.cutmix,
                cutmix_minmax=None,
                prob=self.hparams.mixup_prob,
                switch_prob=self.hparams.mixup_switch_prob,
                mode=self.hparams.mixup_mode,
                label_smoothing=self.hparams.smoothing,
                num_classes=self.hparams.num_classes
            )

    def _init_frozen_params(self):
        if self.hparams.attn_only:
            for name_p, p in self.model.named_parameters():
                if '.attn.' in name_p:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            self.model.head.weight.requires_grad = True
            self.model.head.bias.requires_grad = True
            self.model.pos_embed.requires_grad = True
            for p in self.model.patch_embed.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        samples, targets = batch
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
        outputs = self.forward(samples)
        loss = self.train_criterion(outputs, targets)
        loss_value = loss.item()
        self.log('Loss/train', loss_value, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.forward(samples)
        loss = self.valid_criterion(outputs, targets)
        loss_value = loss.item()
        self.valid_acc.update(outputs, targets)
        self.log("Accuracy/val_driver", self.valid_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("Loss/val", loss_value, sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        samples, targets = batch
        outputs = self.forward(samples)
        self.confusion_matrix.update(outputs, targets)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        opt: Optimizer = self.optimizers()
        self.log("LR", opt.param_groups[0]["lr"], on_epoch=True, sync_dist=True)

    def on_test_end(self) -> None:
        self.visualize_confusion_matrix()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.hparams.opt,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler, _ = create_scheduler(self.hparams, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: Scheduler, optimizer_idx, metric) -> None:
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

    def visualize_confusion_matrix(self):
        cf_matrix = self.confusion_matrix.compute().cpu()
        categories = [f'C{i}' for i in range(self.hparams.num_classes)]
        fig, ax = plt.subplots(1)
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.2f', xticklabels=categories, yticklabels=categories)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True Label')
        vis_path = Path(self.hparams.vis_path)
        fig.savefig(str(vis_path / f"cf_matrix.png"), dpi=200)


class ViTDDLM(LightningModule):
    def __init__(self,
                 batch_size: int = 256,
                 num_classes_1: int = 10,  # Distraction Detection
                 num_classes_2: int = 8,  # Emotion Recognition
                 epochs: int = 100,
                 attn_only: bool = True,
                 smoothing: float = 0.0,  # Label smoothing
                 lambda1: float = 0.5,
                 lambda2: float = 0.5,
                 pretrained_ckpt_path: str = "",
                 vis_path: str = "./runs/vis",

                 # Model parameters
                 model: str = "deit3_small_patch16_224",  # Name of model to train
                 input_size_1: int = 224,  # images input size
                 input_size_2: int = 32,  # images input size
                 drop: float = 0.0,  # Dropout rate
                 drop_path: float = 0.05,  # Drop path rate

                 # Optimizer parameters
                 opt: str = "adamw",
                 weight_decay: float = 0.05,

                 # Learning rate schedule parameters
                 sched: str = "cosine",
                 lr: float = 4e-3,
                 warmup_lr: float = 1e-6,
                 min_lr: float = 1e-5,
                 warmup_epochs: int = 5,  # epochs to warmup LR, if scheduler supports
                 cooldown_epochs: int = 0,  # epochs to cooldown LR at min_lr, after cyclic schedule ends
                 ):
        super(ViTDDLM, self).__init__()
        self.save_hyperparameters()

        self.model = ViTDD(
            self.hparams.model,
            self.hparams.num_classes_1,
            self.hparams.num_classes_2,
            drop_rate=self.hparams.drop,
            drop_path_rate=self.hparams.drop_path,
            input_size_1=self.hparams.input_size_1,
            input_size_2=self.hparams.input_size_2
        )

        self._init_frozen_params()
        self._init_load_ckpt()

        self.train_criterion = torch.nn.CrossEntropyLoss()
        self.valid_criterion = torch.nn.CrossEntropyLoss()
        self.valid_acc_1 = Accuracy()
        self.valid_acc_2 = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes_1, normalize='true')
        self.test_acc_per_class = Accuracy(average=None, num_classes=self.hparams.num_classes_1)

    def _init_frozen_params(self):
        if self.hparams.attn_only:
            for name_p, p in self.model.named_parameters():
                if '.attn.' in name_p:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            self.model.backbone.head.weight.requires_grad = True
            self.model.backbone.head.bias.requires_grad = True
            self.model.head2.weight.requires_grad = True
            self.model.head2.bias.requires_grad = True
            self.model.backbone.pos_embed.requires_grad = True
            self.model.pos_embed2.requires_grad = True
            for p in self.model.patch_embed2.parameters():
                p.requires_grad = True

    def _init_load_ckpt(self):
        if self.hparams.pretrained_ckpt_path:
            ckpt = torch.load(self.hparams.pretrained_ckpt_path)
            state_dict = {k.partition('model.')[2]: ckpt['state_dict'][k] for k in ckpt['state_dict'].keys()}
            self.model.backbone.load_state_dict(state_dict)

    def forward(self, x1, x2):
        return self.model.forward(x1, x2)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        d_img, d_target, face_img, emo_target = batch
        d_pred, emo_pred = self.forward(d_img, face_img)
        loss = self.hparams.lambda1 * self.train_criterion(d_pred, d_target) + \
               self.hparams.lambda2 * self.train_criterion(emo_pred, emo_target)
        loss_value = loss.item()
        self.log('Loss/train', loss_value, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        d_img, d_target, face_img, emo_target = batch
        d_pred, emo_pred = self.forward(d_img, face_img)
        loss = self.hparams.lambda1 * self.valid_criterion(d_pred, d_target) + \
               self.hparams.lambda2 * self.valid_criterion(emo_pred, emo_target)
        loss_value = loss.item()
        self.valid_acc_1.update(d_pred, d_target)
        self.valid_acc_2.update(emo_pred, emo_target)
        self.log("Accuracy/val_driver", self.valid_acc_1, on_step=True, on_epoch=True, sync_dist=True)
        self.log("Accuracy/val_emotion", self.valid_acc_2, on_step=True, on_epoch=True, sync_dist=True)
        self.log("Loss/val", loss_value, sync_dist=True)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        opt: Optimizer = self.optimizers()
        self.log("LR", opt.param_groups[0]["lr"], on_epoch=True)

    def test_step(self, batch, batch_idx):
        d_img, d_target, face_img, emo_target = batch
        d_pred, emo_pred = self.forward(d_img, face_img)
        self.confusion_matrix.update(d_pred, d_target)
        self.test_acc_per_class.update(d_pred, d_target)

    def on_test_end(self) -> None:
        self.visualize_confusion_matrix()
        acc_per_class = self.test_acc_per_class.compute().cpu().numpy()
        print(acc_per_class)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.hparams.opt,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler, _ = create_scheduler(self.hparams, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: Scheduler, optimizer_idx, metric) -> None:
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

    def visualize_confusion_matrix(self):
        cf_matrix = self.confusion_matrix.compute().cpu()
        categories = [f'C{i}' for i in range(self.hparams.num_classes_1)]
        fig, ax = plt.subplots(1)
        sns.heatmap(cf_matrix, annot=True, cmap='Greens', fmt='.2f', xticklabels=categories, yticklabels=categories)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True Label')
        vis_path = Path(self.hparams.vis_path)
        fig.savefig(str(vis_path / f"cf_matrix_vitdd.png"), dpi=200)


def cli_main():
    cli = LightningCLI(seed_everything_default=42,
                       trainer_defaults=dict(accelerator='gpu', devices=1),
                       save_config_overwrite=True)


if __name__ == "__main__":
    cli_main()
