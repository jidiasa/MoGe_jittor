# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# jt_dino_head.py

import jittor as jt
import jittor.nn as nn
from jittor.nn import init
from typing import List


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)

def l2_normalize(x: jt.Var, dim: int = -1, eps: float = 1e-12) -> jt.Var:
    return x / (jt.norm(x, dim=dim, keepdims=True) + eps)

class WeightNormLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.weight_v = jt.randn((out_dim, in_dim))
        self.weight_g = jt.ones(out_dim)
        self.bias = None
        if bias:
            self.bias = jt.zeros(out_dim)

    def execute(self, x):
        weight = l2_normalize(self.weight_v, dim=1) * self.weight_g.unsqueeze(1)
        x = jt.matmul(x, weight.t())
        if self.bias is not None:
            x += self.bias
        return x


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)

        self.last_layer = WeightNormLinear(bottleneck_dim, out_dim, bias=False)
        self.last_layer.weight_g.assign(jt.ones(out_dim))

        # Init all linear weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.mlp(x)
        x = l2_normalize(x, dim=-1, eps=1e-12)
        x = self.last_layer(x)
        return x
