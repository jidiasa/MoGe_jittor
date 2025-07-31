# jt_patch_embed.py

import jittor as jt
import jittor.nn as nn
from typing import Optional, Tuple, Union

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten_embedding: bool = True,
    ):
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_HW,
            stride=patch_HW
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def execute(self, x: jt.Var) -> jt.Var:
        B, C, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width {patch_W}"

        x = self.proj(x)  # (B, embed_dim, H_patch, W_patch)
        H, W = x.shape[2], x.shape[3]
        x = x.reshape((B, self.embed_dim, -1)).transpose((0, 2, 1))  # (B, HW, C)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape((B, H, W, self.embed_dim))  # (B, H, W, C)
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        flops += Ho * Wo * self.embed_dim  # norm layer
        return flops
