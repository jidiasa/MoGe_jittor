import os, urllib.request, torch, numpy as np

URL = ("https://dl.fbaipublicfiles.com/dinov2/"
       "dinov2_vitb14/dinov2_vitb14_pretrain.pth")
CKPT = "dinov2_vitb14_pretrain.pth"
NPZ  = "dinov2_vitb14.npz"

# 1. 下载官方权重
if not os.path.exists(CKPT):
    urllib.request.urlretrieve(URL, CKPT)

# 2. 读取 state_dict（全是 torch.Tensor）
sd_pt = torch.load(CKPT, map_location="cpu")

# 3. 转成 numpy 并保存为 npz
sd_np = {k: v.numpy() for k, v in sd_pt.items()}
np.savez(NPZ, **sd_np)
print("Saved", NPZ, "with", len(sd_np), "tensors")