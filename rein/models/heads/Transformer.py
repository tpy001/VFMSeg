from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import os
import warnings
from mmseg.registry import MODELS


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def _forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class MemEffAttention(CrossAttention):

    def forward(self, x, context=None, mask=None):
        if not XFORMERS_AVAILABLE:
            return self._forward(x,context)

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v))
        x = memory_efficient_attention(q, k, v)
        x = rearrange(x, 'b n h d -> b n (h d)', h=h)

        return self.to_out(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = MemEffAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(query_dim, dropout=dropout, glu=gated_ff)
        self.attn2 = MemEffAttention(query_dim=query_dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)


    def forward(self, x, context=None):
        return self._forward(x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        if context is not None:
            # context = self.proj_in_context(context)
            context = rearrange(context, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
    
@MODELS.register_module()
class TransformerDecoder(nn.Module):
    
    def __init__(self, query_dim, img_feat_dim, n_heads, d_head,
                 depth=1, dropout=0., ):
        super().__init__()
        self.in_channels = query_dim
        self.norm = Normalize(query_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, n_heads, d_head, dropout=dropout, context_dim=img_feat_dim)
                for _ in range(depth)]
        )


    def forward(self, query,img_feats):
        b, c, h, w = img_feats.shape
        
        x = self.norm(query)
        x = rearrange(x, 'b c h w -> b (h w) c')
        img_feats = rearrange(img_feats, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, img_feats)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
    
@MODELS.register_module()
class MaskTransformerDecoder(TransformerDecoder):
    
    def __init__(self, mask_ratio,**kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(1,self.in_channels,1,1))
        self.mask_enable = True

    def mask_feat(self,feats):
        b, c, h, w = feats.shape
        mask = torch.rand(b,1,h,w,device=feats.device) > self.mask_ratio
        expanded_mask_token = self.mask_token.expand(b, -1, h, w)
        mask_feat = torch.where(mask, feats, expanded_mask_token)
        return mask_feat

    def forward(self, query,img_feats):
        # Mask features
        if self.mask_enable:
            query = self.mask_feat(query)

        # Transformer Decoder
        b, c, h, w = img_feats.shape
        x = self.norm(query)
        x = rearrange(x, 'b c h w -> b (h w) c')
        img_feats = rearrange(img_feats, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, img_feats)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x