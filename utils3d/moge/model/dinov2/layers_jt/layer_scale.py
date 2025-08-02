# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

# jt_layer_scale.py

import jittor as jt
import jittor.nn as nn
from typing import Union

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, jt.Var] = 1e-5,
        inplace: bool = False,
    ):
        super().__init__()
        self.inplace = inplace
        if isinstance(init_values, (float, int)):
            init_tensor = jt.ones(dim) * init_values
        else:
            init_tensor = init_values

        self.gamma = init_tensor

    def execute(self, x: jt.Var) -> jt.Var:
        return x.multiply(self.gamma) if self.inplace else x * self.gamma
