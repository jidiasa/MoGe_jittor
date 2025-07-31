# test_moge_blocks.py
import jittor as jt
from model.moge_model_jt import ResidualConvBlock, Head      
jt.flags.use_cuda = 1              
def test_residual_block():
    print("=== ResidualConvBlock ===")
    block = ResidualConvBlock(64, hidden_channels=128).cuda()

    x = jt.randn((2, 64, 32, 32))
    y = block(x)
    print("input ", x.shape, "→ output", y.shape)

    loss = y.mean()
    grads = jt.grad(loss, block.parameters())        # 取所有参数梯度
    has_grad = any(g is not None for g in grads)
    print("grad ok:", has_grad)

def test_head():
    print("\n=== Head ===")
    head = Head(
        num_features=4,
        dim_in=768,
        dim_out=[3,1],             #
    ).cuda()

    # mock ViT hidden states: list of 4 tuples (feat, cls)
    B, P, C = 2, 16*16, 768        # 224/14 = 16
    hidden = [
        (jt.randn((B, C, P)), jt.randn((B, C))) for _ in range(4)
    ]
    img = jt.randn((B, 3, 224, 224))
    outs = head(hidden, img)       # list([B,3,H,W], [B,1,H,W])
    print("output 0:", outs[0].shape, " output 1:", outs[1].shape)
    loss = sum([o.abs().mean() for o in outs])

    # ① 最简：直接用 jt.grad 检查梯度
    grads = jt.grad(loss, head.parameters())
    print("grad ok:", any(g is not None for g in grads))

if __name__ == "__main__":
    test_residual_block()
    test_head()
