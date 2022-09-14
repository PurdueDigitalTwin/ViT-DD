import torch
import torch.nn as nn

from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed


class ViTDD(nn.Module):
    def __init__(self, backbone: str, num_classes_1, num_classes2, drop_rate, drop_path_rate, input_size_1,
                 input_size_2):
        super(ViTDD, self).__init__()
        self.backbone: VisionTransformer = create_model(
            backbone,
            pretrained=True,
            num_classes=num_classes_1,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=None,
            img_size=input_size_1
        )
        self.num_classes2 = num_classes2
        self.input_size_2 = input_size_2
        self.embed_dim = self.backbone.embed_dim
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.patch_embed2 = PatchEmbed(img_size=input_size_2, patch_size=16, in_chans=3, embed_dim=self.embed_dim)
        self.pos_embed2 = nn.Parameter(torch.randn(1, self.patch_embed2.num_patches, self.embed_dim) * .02)
        self.head2 = nn.Linear(self.embed_dim, self.num_classes2)

    def forward(self, x1, x2):
        x1 = self.backbone.patch_embed(x1)
        x1 = x1 + self.backbone.pos_embed
        x2 = self.patch_embed2(x2)
        x2 = x2 + self.pos_embed2
        x = torch.cat([self.backbone.cls_token.expand(x1.shape[0], -1, -1),
                       self.cls_token2.expand(x2.shape[0], -1, -1),
                       x1,
                       x2], dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1 = self.backbone.head(x1)
        x2 = self.head2(x2)
        return x1, x2
