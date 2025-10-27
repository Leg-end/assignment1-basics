import torch
import einx
from einops import rearrange, einsum
from torch import nn, Tensor
from jaxtyping import Float, Int


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 d_model: int,
                 theta: float,
                 context_length: int,
                 device: torch.device | None=None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, d_model, theta).to(device), persistent=False
        )
    
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"] | None=None) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        if pos_ids is None:
            seq_len = x.shape[-2]
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def get_FLOPS(self, ctx_len):  # for single k
        return 3 * ctx_len * self.d_model
    
    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"



def rotate_half(x, original=True):
    if not original:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x = torch.concat([-x2, x1], dim=-1)
    else:
        x1 = x[..., 1::2]  # odd
        x2 = x[..., ::2]  # even
        # [-x1, x0, -x3, x2, ...]
        x = torch.stack([-x1, x2], dim=-1).view(*x.shape)
    return x
        
        
        