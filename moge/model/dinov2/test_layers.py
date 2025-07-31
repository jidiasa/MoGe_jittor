from layers_jt.block import Block
import jittor as jt

blk = Block(dim=768, num_heads=12, drop_path=0.1)
blk.eval()  # or .train()
x = jt.randn((1, 256, 768))
y = blk(x)
print(y.shape)  # [1, 256, 768]
