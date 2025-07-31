from typing import *
import math
import jittor as jt
import jittor.nn as nn

def _recompute(fn, *args, **kwargs):
    if hasattr(jt, "recompute"):
        return jt.recompute(fn, *args, **kwargs)
    else:
        return fn(*args, **kwargs)
    
def wrap_module_with_gradient_checkpointing(module: nn.Module):

    class _CheckpointWrapper(module.__class__):
        _restore_cls = module.__class__       

        def execute(self, *args, **kwargs):
            def _forward(*a, **kw):
                return super(_CheckpointWrapper, self).execute(*a, **kw)

            return _recompute(_forward, *args, **kwargs)

    module.__class__ = _CheckpointWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    if hasattr(module.__class__, "_restore_cls"):
        module.__class__ = module.__class__._restore_cls
    return module


def _scaled_dot_attn(q, k, v, attn_bias=None):
   
    Dh = q.shape[-1]
    logits = jt.bmm(q, k.transpose(2, 3)) / math.sqrt(Dh)  
    if attn_bias is not None:         
        logits += attn_bias 
    weights = nn.softmax(logits, dim=-1)
    out = jt.bmm(weights, v)                               
    return out


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    
    class _AttentionSDPA(module.__class__):
        def execute(self, x: jt.Var, attn_bias=None):
            B, N, C = x.shape
            H = self.num_heads
            qkv = self.qkv(x)                     # (B,N,3C)
            qkv = qkv.reshape(B, N, 3, H, C//H)   # (B,N,3,H,Dh)
            q, k, v = [t.transpose(1,2) for t in jt.unbind(qkv, dim=2)]
            # 现在 (B,H,N,Dh)
            out = _scaled_dot_attn(q, k, v, attn_bias)
            out = out.transpose(1,2).reshape(B, N, C)   # 回 (B,N,C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

    module.__class__ = _AttentionSDPA
    return module

# if __name__ == "__main__":
#     jt.flags.use_cuda = 1
#     from dinov2.models.vision_transformer_jt import vit_base

#     blk = vit_base(patch_size=14).blocks[0][0]  
#     blk.attn = wrap_dinov2_attention_with_sdpa(blk.attn)
#     blk = wrap_module_with_gradient_checkpointing(blk)

#     x = jt.randn((2, 17*17+1, 768))
#     y = blk(x)
#     print("out:", y.shape)                        
#     print("mean =", y.mean().item())             

#     loss = y.mean()
#     grads = jt.grad(loss, blk.parameters())
#     has_grad = any([g is not None for g in grads])
#     print("grad ok:", has_grad)