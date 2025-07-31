# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import jittor as jt
import jittor.nn as nn
from typing import Optional

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x: jt.Var, attn_bias: Optional[jt.Var] = None) -> jt.Var:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]
        q = q * self.scale

        attn = jt.matmul(q, k.transpose(2, 3))  # [B, heads, N, N]
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = jt.matmul(attn, v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, dim]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out