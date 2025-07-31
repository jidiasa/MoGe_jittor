import jittor as jt
from model.moge_model_jt import MoGeModel      
jt.flags.use_cuda = 1   
model = MoGeModel(encoder="dinov2_vitb14").cuda()
img = jt.randn((1,3,224,224))
out = model(img)
print("points:", out["points"].shape)