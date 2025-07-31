# 在仅安装 jittor 的环境运行
import numpy as np, jittor as jt
from models.vision_transformer_jt import vit_base   # 你的 Jittor ViT 路径

NPZ = "dinov2_vitb14.npz"
sd_np = np.load(NPZ)

# 1. 构造模型
model = vit_base(patch_size=14, drop_path_rate=0, img_size=518, init_values=1.0, block_chunks=0).eval()

# 2. 把 numpy → jt.Var 并加载
sd_jt = {k: jt.array(sd_np[k]) for k in sd_np.files}
ret = model.load_parameters(sd_jt) 

model_keys = {p.name() for p in model.parameters()}      
state_keys = set(sd_jt.keys())                           

missing    = model_keys   - state_keys   
unexpected = state_keys - model_keys     

print("missing keys   :", missing)
print("unexpected keys:", unexpected)

# 3. 简单前向验证
x = jt.randn((1,3,224,224))
with jt.no_grad():
    y = model(x)
print("CLS embedding shape:", y.shape, "norm:", y.norm())

