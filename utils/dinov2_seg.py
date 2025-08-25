import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import timm
import sys

sys.path.append('/data0/3dg/splatam/segmentation/facebookresearch_dinov2_main')
from dinov2.models import vision_transformer as vits

# we use dinov2_vitb14 model
dino_backbones = {
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
}

def make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: bool = False,
    **kwargs,
):
    from dinov2.models import vision_transformer as vits

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    return model

class DINO2SEG(nn.Module):
    def __init__(self, img_h, img_w, num_cls, backbone='dinov2_b', mode='get_semantic', edge=10, dim=16):
        super(DINO2SEG, self).__init__()
        self.backbones = dino_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        # Create the model
        if backbone == 'dinov2_b':
            self.backbone = make_dinov2_model(arch_name="vit_base")
        elif backbone == 'dinov2_l':
            self.backbone = make_dinov2_model(arch_name="vit_large")

        self.num_class = num_cls
        self.mode = mode
        self.img_h = img_h
        self.img_w = img_w

        switch = False
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                if 'blocks.4.' in name:
                    switch = True
                if switch:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        if edge == 10:
            self.segmentation_conv = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(self.embedding_size, dim, (3, 3), padding=(1, 1)),
                nn.Upsample((self.img_h - 2 * edge, self.img_w - 2 * edge)),
                nn.Conv2d(dim, self.num_class, (3, 3), padding=(1, 1))
            )
            self.upsample = nn.Upsample((448, 616))

        elif edge == 8:
            self.img_h = 368
            self.img_w = 496
            self.segmentation_conv = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(self.embedding_size, dim, (3, 3), padding=(1, 1)),
                nn.Upsample((self.img_h, self.img_w)),
                nn.Conv2d(dim, self.num_class, (3, 3), padding=(1, 1))
            )
            self.upsample = nn.Upsample((364, 490))

        else:
            self.segmentation_conv = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(self.embedding_size, dim, (3, 3), padding=(1, 1)),
                nn.Upsample((self.img_h, self.img_w)),
                nn.Conv2d(dim, self.num_class, (3, 3), padding=(1, 1))
            )
            self.upsample = nn.Upsample((672, 1190))

    def forward(self, x):
        if self.mode == 'classification':
            outputs = self.segmentation_conv[-1](x)
            #out = torch.max(outputs, 1).indices.squeeze()
            return outputs

        x = self.upsample(x)
        bs = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)

        out = self.backbone.forward_features(x.float())

        out = out["x_norm_patchtokens"] # [1,2880,768]

        out = out.reshape(bs, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))  # [1,768,40,72]

        if self.mode == 'get_feature':
            for i in range(3):
                out = self.segmentation_conv[i](out)
            return out
        elif self.mode == 'get_semantic':
            outputs = self.segmentation_conv(out)
            #out = torch.max(outputs, 1).indices
            return outputs



class Segmentation:
    def __init__(self, config):
        self.dim = config['model']['c_dim']

        self.pretrained_model_path = config['model']['pretrained_model_path']
        print("pretrained_model_path: ", self.pretrained_model_path)
        self.n_classes = config['model']['n_classes']

        self.img_h = config['model']['H']
        self.img_w = config['model']['W']

        self.crop_edge = config['model']['crop_edge']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = self.get_dinov2().cuda()

    def get_dinov2(self):
        model = DINO2SEG(img_h=self.img_h, img_w=self.img_w, num_cls=self.n_classes, edge=self.crop_edge, dim=self.dim)
        model.load_state_dict(torch.load(self.pretrained_model_path, map_location=self.device))
        print("get_semantic_dinov2")
        return model

    def set_mode_get_feature(self):
        self.cnn.mode = 'get_feature'

    def set_mode_get_semantic(self):
        self.cnn.mode = 'get_semantic'

    def set_mode_classification(self):
        self.cnn.mode = 'classification'