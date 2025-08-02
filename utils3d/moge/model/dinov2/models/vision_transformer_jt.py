# jt_vision_transformer.py  ── Jittor 版本 DinoViT
import math
from functools import partial
from typing import Callable, Sequence, Tuple, Union, List

import jittor as jt
import jittor.nn as nn

from ..layers_jt import Mlp, PatchEmbed, SwiGLUFFNFused, Attention, Block

def _named_apply(fn: Callable, module: nn.Module, name=""):
    for child_name, child_module in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        fn(child_module, full_name)          
        _named_apply(fn, child_module, full_name)

def _init_weights_linear(m: nn.Module, name=""):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class BlockChunk(nn.ModuleList):
    def execute(self, x):
        for b in self:
            x = b(x)
        return x

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: float = 1e-5,        
        act_layer = nn.GELU,
        ffn_layer: str = "mlp",             
        block_chunks: int = 1,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.embed_dim = embed_dim
        self.num_features = embed_dim 
        self.num_tokens = 1
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros(1, 1, embed_dim)
        self.pos_embed = jt.zeros(1, num_patches + self.num_tokens, embed_dim)
        self.register_tokens = (
            jt.zeros(1, num_register_tokens, embed_dim) if num_register_tokens else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [float(x) for x in jt.linspace(0, drop_path_rate, depth)]

        if ffn_layer.lower() == "mlp":
            ffn_cls = Mlp
        elif ffn_layer.lower() in ("swiglufused", "swiglu"):
            ffn_cls = SwiGLUFFNFused
        elif ffn_layer.lower() == "identity":
            ffn_cls = lambda *a, **kw: nn.Identity()
        else:
            raise NotImplementedError(ffn_layer)

        blocks: List[nn.Module] = []
        for i in range(depth):
            blk = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_cls,
                init_values=init_values,
                attn_class=Attention,        
            )
            blocks.append(blk)

        if block_chunks > 0:
            self.chunked_blocks = True
            chunksize = depth // block_chunks
            chunked = []
            for i in range(0, depth, chunksize):

                chunked.append([nn.Identity()] * i + blocks[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()           

        self.mask_token = jt.zeros(1, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.gauss_(self.cls_token, 0.0, 1e-6)          
        if self.register_tokens is not None:
            nn.init.gauss_(self.register_tokens, 0.0, 1e-6)
        _named_apply(_init_weights_linear, self)


    def _interpolate_pos_encoding(self, x, w, h):
        w, h = int(w), int(h)
        prev_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        cls_pe, patch_pe = pos_embed[:, 0], pos_embed[:, 1:]
        dim = x.shape[-1]


        w0, h0 = int(w // self.patch_size), int(h // self.patch_size)
        M = int(math.sqrt(N))
        assert M * M == N, "pos_embed wrong"

        if self.interpolate_offset:
            h0 += self.interpolate_offset
            w0 += self.interpolate_offset

        target_size = (int(h0), int(w0))  

        patch_pe = jt.nn.interpolate(
            patch_pe.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=target_size,
            mode="bicubic"
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return jt.concat([cls_pe.unsqueeze(0), patch_pe], dim=1).cast(prev_dtype)

    def _prepare_tokens(self, x: jt.Var, masks=None):
        B, C, W, H = x.shape
        x = self.patch_embed(x)                       

        if masks is not None:
            x = jt.where(masks.unsqueeze(-1), self.mask_token.cast(x.dtype), x)

        x = jt.concat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self._interpolate_pos_encoding(x, W, H)

        if self.register_tokens is not None:
            x = jt.concat([
                x[:, :1],
                self.register_tokens.expand(B, -1, -1),
                x[:, 1:],
            ], dim=1)
        return x
    
    def prepare_tokens_with_masks(self, x: jt.Var, masks: jt.Var = None):
        B, C_img, H_img, W_img = x.shape
        x = self.patch_embed(x)                   

        if masks is not None:
            mask_tok = self.mask_token.cast(x.dtype).expand(B, -1, -1)
            x = jt.where(masks.unsqueeze(-1), mask_tok, x)

        cls_tok = self.cls_token.expand(B, -1, -1)      
        x = jt.concat([cls_tok, x], dim=1)              

        x = x + self._interpolate_pos_encoding(x, H_img, W_img)

        if self.register_tokens is not None:
            reg_tok = self.register_tokens.expand(B, -1, -1)  # (B,R,C)
            x = jt.concat([x[:, :1], reg_tok, x[:, 1:]], dim=1)

        return x

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        outputs = []
        total_blocks = len(self.blocks)
        blocks_to_take = range(total_blocks - n, total_blocks) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                outputs.append(x)
        assert len(outputs) == len(blocks_to_take), \
            f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
        return outputs

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        outputs, i = [], 0
        total_block_len = len(self.blocks[-1])          
        blocks_to_take  = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for chunk in self.blocks:                       
            for blk in chunk[i:]:                       
                x = blk(x)
                if i in blocks_to_take:
                    outputs.append(x)
                i += 1
        assert len(outputs) == len(blocks_to_take), \
            f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
        return outputs
    
    def get_intermediate_layers(
        self,
        x: jt.Var,
        n: Union[int, Sequence[int]] = 1,    
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = (self._get_intermediate_layers_chunked if self.chunked_blocks
                else self._get_intermediate_layers_not_chunked)(x, n)

        if norm:
            outputs = [self.norm(o) for o in outputs]


        cls_tokens = [o[:, 0] for o in outputs]
        outputs    = [o[:, 1 + self.num_register_tokens:] for o in outputs]  

        if reshape:
            B, _, H_img, W_img = x.shape
            H_patch, W_patch   = H_img // self.patch_size, W_img // self.patch_size
            outputs = [
                o.reshape(B, H_patch, W_patch, -1)
                .permute(0, 3, 1, 2).contiguous()
                for o in outputs
            ]

        return tuple(zip(outputs, cls_tokens)) if return_class_token else tuple(outputs)

    def execute(self, x, masks=None, is_training=False):
        feats = self._forward_features(x, masks)
        if is_training:
            return feats
        return self.head(feats["x_norm_clstoken"])

    def _forward_features(self, x, masks=None):
        x = self._prepare_tokens(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken":  x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : 1 + self.num_register_tokens],
            "x_norm_patchtokens": x_norm[:, 1 + self.num_register_tokens :],
            "x_prenorm": x,
            "masks": masks,
        }

def vit_small(patch_size=14, **kw):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, **kw
    )

def vit_base(patch_size=14, **kw):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, **kw
    )

def vit_large(patch_size=14, **kw):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, **kw
    )

def vit_giant2(patch_size=14, **kw):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24, **kw
    )
