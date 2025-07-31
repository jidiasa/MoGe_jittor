#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert the pretrained MoGe-2 PyTorch weights to a Jittor-friendly .npz file.
The resulting file will be saved as `moge_weights_jittor.npz` in the
current working directory.

If you are running on a CPU-only machine, the script will fall back to CPU
automatically.
"""

import numpy as np
import torch
from moge.model.moge_model import MoGeModel  # MoGe-2

# ------------------------------------------------------------------
# 1. Load the PyTorch model (local path or HF Hub) and move to device
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
model.to(device).eval()   # evaluation mode is safer when exporting
torch.set_grad_enabled(False)

# ------------------------------------------------------------------
# 2. Extract state_dict and convert every tensor to NumPy
# ------------------------------------------------------------------
state_dict = model.state_dict()
np_state_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}

# ------------------------------------------------------------------
# 3. Save as .npz (use savez_compressed if you prefer smaller size)
# ------------------------------------------------------------------
np.savez("moge_weights_jittor.npz", **np_state_dict)

print("✅  Weights exported to moge_weights_jittor.npz")
