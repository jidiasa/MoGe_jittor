
import jittor as jt
from models.vision_transformer_jt import vit_base   

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    model = vit_base(
        patch_size=14,
        drop_path_rate=0.0,     
    ).eval()                   

    print(f"Model params: {count_params(model)/1e6:.2f} M")

    dummy = jt.randn((1, 3, 224, 224))

    with jt.no_grad():
        out = model(dummy)     

    print("Output shape :", out.shape)   

    feats = model(dummy, is_training=True)
    print("x_norm_clstoken :", feats["x_norm_clstoken"].shape)   # [1, 768]
    print("x_norm_patchtokens :", feats["x_norm_patchtokens"].shape)  # [1, 256, 768]

if __name__ == "__main__":
    main()
