import torch
import einx
import logging
from einops import rearrange, einsum
from torch import nn, Tensor
from jaxtyping import Float, Int
# For NTK-aware RoPE, please refer to https://spaces.ac.cn/archives/9675
# or https://zhuanlan.zhihu.com/p/8306958113

logger = logging.getLogger(__name__)

def rotate_half(x, original=True):
    if not original:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x = torch.concat([-x2, x1], dim=-1)
    else:
        x1 = x[..., ::2]  # even, [x0, x2, x4, x6, ...]
        x2 = x[..., 1::2]  # odd, [x1, x3, x5, x7, ...]
        # [-x1, x0, -x3, x2, ...]
        x = torch.stack([-x2, x1], dim=-1).view(*x.shape)
    return x


def apply_rope(q, k, cos, sin, unsqueeze_dim=1, original=True):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q, original=original) * sin)
    k_embed = (k * cos) + (rotate_half(k, original=original) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 d_model: int,
                 theta: float,
                 context_length: int,
                 context_scale: int=1,  # expand context length when inference using NTK-aware RoPE
                 device: torch.device | None=None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.context_length = context_length
        self.context_scale = context_scale
        self.theta = theta
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(
                context_length, d_model, theta, context_scale).to(device), persistent=False
        )
        
    
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float, context_scale: int = 1) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        if context_scale > 1:  
            # NTK-aware RoPE: theta = theta * lambda, lambda = context_scale ** (dim / (dim - 2)) or context_scale ** (2 / (dim - 2))
            logger.info(f"Expanding context length with {context_scale} scales. NTK-aware RoPE applied.")
            lmbd = context_scale ** (dim / (dim - 2))
            theta = theta * lmbd
        freqs = theta ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))
    
    def get_cosin(self, T, pos_ids: Int[Tensor, " ... seq"] | None=None, original: bool = True):
        if pos_ids is None:
            pos_ids = torch.arange(T, device=self.device).unsqueeze(0)
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)
        if not original:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        else:
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        return cos, sin

    def forward(self, x: Float[Tensor, " ... seq d"],
                pos_ids: Int[Tensor, " ... seq"] | None=None,
                original: bool = True) -> Float[Tensor, " ... seq d"]:
        
        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = self.get_cosin(x.shape[-2], pos_ids, original=original)
        x = (x * cos) + rotate_half(x, original=original) * sin
        return x
    
    def get_FLOPS(self, ctx_len):  # for single k
        return 3 * ctx_len * self.d_model
    
    def extra_repr(self):
        return f"context_length={self.context_length}, dim/2={self._freq_cis_cache.shape[1]}"

        
        
        