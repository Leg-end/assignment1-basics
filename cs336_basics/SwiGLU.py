import torch
from torch import nn
from .linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = 4 * d_model
        self.up_proj = Linear(d_model, d_ff)
        self.down_proj = Linear(d_ff, d_model)

    def forward(self, x):
        return self.down_proj(silu(self.up_proj(x)))


class SwiGLU(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu. approximately 8/3 * d_model.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w1 = Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
    def get_FLOPS(self, ctx_len):
        w1_flops = 2 * ctx_len * self.d_ff * self.d_model
        w2_flops = 2 * ctx_len * self.d_ff * self.d_model
        w3_flops = 2 * ctx_len * self.d_ff * self.d_model
        return w1_flops + w2_flops + w3_flops + 5 * ctx_len * self.d_ff