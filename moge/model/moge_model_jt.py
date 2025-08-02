from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import importlib
import warnings
import json

import jittor as jt
import jittor.nn as nn

import utils3d
from huggingface_hub import hf_hub_download

from ..utils.geometry_jt import normalized_view_plane_uv, recover_focal_shift, gaussian_blur_2d
from .utils_jt import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from ..utils.tools import timeit
from .dinov2.models.vision_transformer_jt import (
    vit_small,     
    vit_base,
    vit_large,
    vit_giant2
)


def normalized_view_plane_uv(*, width: int, height: int,
                             aspect_ratio: float,
                             dtype=jt.float32, device=None):
    ys = jt.linspace(-1., 1., height).cast(dtype)   # H
    xs = jt.linspace(-1., 1., width ).cast(dtype) 
    yy, xx = jt.meshgrid(ys, xs)                     # (H,W)
    uv  = jt.stack([xx / aspect_ratio, yy], dim=-1)  # (H,W,2)
    return uv


def conv2d_pad(in_ch, out_ch, k=3, stride=1, **_):
    pad = (k - 1) // 2
    return nn.Conv(in_ch, out_ch, k, stride, padding=pad)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
                 padding_mode='replicate',
                 activation='relu',
                 norm='group_norm'):
        super().__init__()
        out_channels    = out_channels    or in_channels
        hidden_channels = hidden_channels or in_channels

        act_layer = {
            'relu'      : lambda : nn.ReLU(),
            'leaky_relu': lambda : nn.LeakyReLU(0.2),
            'silu'      : lambda : nn.SiLU(),
            'elu'       : lambda : nn.ELU(),
        }[activation]

        norm1 = nn.GroupNorm(1, in_channels)
        norm2 = nn.GroupNorm(hidden_channels//32 if norm=='group_norm' else 1,
                             hidden_channels)

        self.layers = nn.Sequential(
            norm1, act_layer(),
            conv2d_pad(in_channels, hidden_channels, 3, padding_mode=padding_mode),
            norm2, act_layer(),
            conv2d_pad(hidden_channels, out_channels, 3, padding_mode=padding_mode),
        )
        self.skip = (nn.Conv(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def execute(self, x):
        return self.layers(x) + self.skip(x)

class Head(nn.Module):
    def __init__(self,
                 num_features: int,
                 dim_in: int,
                 dim_out,                       
                 dim_proj: int = 512,
                 dim_upsample=(256, 128, 128),
                 dim_times_res_block_hidden: int = 1,
                 num_res_blocks: int = 1,
                 res_block_norm='group_norm',
                 last_res_blocks: int = 0,
                 last_conv_channels: int = 32,
                 last_conv_size: int = 1):
        super().__init__()

        self.projects = nn.ModuleList(
            [nn.Conv(dim_in, dim_proj, 1) for _ in range(num_features)]
        )

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch),
                *[ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks)]
            ) for in_ch, out_ch in zip((dim_proj,) + dim_upsample[:-1], dim_upsample)
        ])
        if isinstance(dim_out, int):         
            dim_out = [dim_out]
        elif isinstance(dim_out, (list, tuple)):      
            dim_out = list(dim_out)
        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2,
                dim_out_,
                dim_times_res_block_hidden,
                last_res_blocks,
                last_conv_channels,
                last_conv_size,
                res_block_norm,
            ) for dim_out_ in dim_out
        ])

    def _make_upsampler(self, in_c: int, out_c: int):
        up = nn.ConvTranspose(in_c, out_c, kernel_size=2, stride=2, bias=True)
        conv = nn.Conv(out_c, out_c, kernel_size=3, stride=1, padding=1)   # ← 去掉 padding_mode
        # up.weight.assign(up.weight[:, :, :1, :1])
        return nn.Sequential(up, conv)

    def _make_output_block( 
        self,
        dim_in: int,
        dim_out: int,
        dim_times_res_block_hidden: int,
        last_res_blocks: int,
        last_conv_channels: int,
        last_conv_size: int,
        res_block_norm: str
        ):
        layers = [
            nn.Conv(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1),  # ← same
            *[ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)],
            nn.ReLU(),
            nn.Conv(last_conv_channels, dim_out,
                    kernel_size=last_conv_size,
                    stride=1,
                    padding=last_conv_size // 2),
        ]
        return nn.Sequential(*layers)

    def _add_uv(self, feat, img_h, img_w):
        uv = normalized_view_plane_uv(width=feat.shape[-1],
                                        height=feat.shape[-2],
                                        aspect_ratio=img_w/img_h,
                                        dtype=feat.dtype).to(feat)
        uv = uv.permute(2,0,1).unsqueeze(0).expand(feat.shape[0], -1, -1, -1)
        return jt.concat([feat, uv], dim=1)

    def execute(self, hidden_states, image):
        img_h, img_w = image.shape[-2:]
        ph, pw = img_h // 14, img_w // 14         

        feats = []
        for proj, (feat, _cls) in zip(self.projects, hidden_states):
            feat = feat.permute(0, 2, 1).reshape(feat.shape[0], ph, pw, -1).permute(0, 3, 1, 2)
            feats.append(proj(feat))
        x = jt.stack(feats, dim=1).sum(dim=1)      

        for block in self.upsample_blocks:
            x = self._add_uv(x, img_h, img_w)
            x = block(x)                           

        x = nn.interpolate(x, size=(img_h, img_w), mode='bilinear')
        x = self._add_uv(x, img_h, img_w)

        outputs = [blk(x) for blk in self.output_block]
        return outputs if len(outputs) > 1 else outputs[0]

_BACKBONES = {
    "dinov2_vits14": vit_small,
    "dinov2_vitb14": vit_base,
    "dinov2_vitl14": vit_large,
    "dinov2_vitg14": vit_giant2,
}

def _get_backbone(name, **kw):
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone {name}")
    return _BACKBONES[name](patch_size=14, **kw)

# --------- MoGeModel ----------------------------------------- #
class MoGeModel(nn.Module):
    def __init__(self,
                 encoder: str = "dinov2_vitl14",
                 intermediate_layers: Union[int, List[int]] = 4,
                 dim_proj=512,
                 dim_upsample=(256,128,64),
                 dim_times_res_block_hidden=2,
                 num_res_blocks=2,
                 output_mask=True,
                 split_head=True,
                 remap_output="exp",
                 res_block_norm="group_norm",
                 trained_diagonal_size_range=(600,900),
                 trained_area_range=(500*500, 700*700),
                 last_res_blocks=0,
                 last_conv_channels=32,
                 last_conv_size=1,
                 **deprecated):
        super().__init__()
        if deprecated:
            warnings.warn(f"Ignored deprecated args: {deprecated}")

        # ------------ backbone ------------
        self.encoder_name = encoder
        self.backbone = _get_backbone(encoder, block_chunks=0) 
        dim_feature = self.backbone.num_features                 # 768 / 384 …

        # ------------ head ------------
        head_out = 3 if not output_mask else 4 if output_mask and not split_head else [3, 1]
        self.head = Head(
            num_features=intermediate_layers if isinstance(intermediate_layers,int) else len(intermediate_layers),
            dim_in=dim_feature,
            dim_out=head_out,
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size,
        )

        self.register_buffer("image_mean", jt.array([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("image_std",  jt.array([0.229,0.224,0.225]).view(1,3,1,1))

        # meta
        self.intermediate_layers = intermediate_layers
        self.remap_output        = remap_output
        self.output_mask         = output_mask
        self.split_head          = split_head
        self.trained_area_range  = trained_area_range
        self.trained_diagonal_size_range = trained_diagonal_size_range

    def enable_backbone_gradient_checkpointing(self):
        for i,b in enumerate(self.backbone.blocks):
            self.backbone.blocks[i] = wrap_module_with_gradient_checkpointing(b)

    def enable_jittor_sdpa(self):
        for blk in self.backbone.blocks:
            blk.attn = wrap_dinov2_attention_with_sdpa(blk.attn)

    # ---------- 前向 ----------
    def execute(self, image: jt.Var, mixed_precision=False):
        raw_h, raw_w = image.shape[-2:]
        ph, pw = raw_h // 14, raw_w // 14

        # 归一化
        image = (image - self.image_mean) / self.image_std
        image_14 = nn.interpolate(image, size=(ph*14, pw*14), mode="bilinear")

        # backbone
        feats = self.backbone.get_intermediate_layers(
            image_14, self.intermediate_layers, return_class_token=True
        )

        # head
        out = self.head(feats, image)
        if self.output_mask:
            if self.split_head:
                pts, msk = out
            else:
                pts, msk = jt.split(out, [3,1], dim=1)
            pts = pts.permute(0,2,3,1)
            msk = msk.squeeze(1)
        else:
            pts = out.permute(0,2,3,1)

        # remap
        if self.remap_output in (False, "linear"):
            pass
        elif self.remap_output in ("sinh", True):
            pts = jt.sinh(pts)
        elif self.remap_output=="exp":
            xy,z = jt.split(pts,[2,1],dim=-1)
            z = jt.exp(z)
            pts = jt.concat([xy*z, z], dim=-1)
        elif self.remap_output=="sinh_exp":
            xy,z = jt.split(pts,[2,1],dim=-1)
            pts = jt.concat([jt.sinh(xy), jt.exp(z)], dim=-1)
        else:
            raise ValueError(f"invalid remap {self.remap_output}")

        ret = {"points": pts}
        if self.output_mask:
            ret["mask"] = msk
        return ret

    # ---------- 推理接口 (TODO: utils3d) ----------
    def infer(self, image: jt.Var, **kwargs):
        raise NotImplementedError("infer() 里的 utils3d 依赖尚未迁移，请按需求补充")
