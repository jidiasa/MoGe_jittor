# test_moge_jittor.py
import cv2, numpy as np, jittor as jt
import jittor.nn as nn
from moge.model.moge_model_jt import MoGeModel    
jt.flags.use_cuda = 1

model = MoGeModel(encoder="dinov2_vitl14",
                  output_mask=False).cuda()        

npz = np.load("moge_weights_jittor.npz")
weights = {k: jt.array(v) for k, v in npz.items()}   
model.load_parameters(weights)                    

img_bgr = cv2.imread("example_images/BooksCorridor.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img     = jt.array(img_rgb / 255., dtype=jt.float32)  
img     = img.permute(2,0,1).unsqueeze(0).cuda()      

with jt.no_grad():
    out = model(img)         

pts  = out["points"]
print("points:", pts.shape, pts.dtype, pts.min().item(), pts.max().item())

if "mask" in out:
    print("mask :", out["mask"].shape, out["mask"].dtype)