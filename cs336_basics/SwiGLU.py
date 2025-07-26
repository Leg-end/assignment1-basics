import torch
from torch import nn


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)



class SwiGLU(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
    def get_FLOPS(self, ctx_len):
        w1_flops = 2 * ctx_len * self.d_ff * self.d_model + 2 * ctx_len * self.d_ff
        w2_flops = 2 * ctx_len * self.d_ff * self.d_model
        w3_flops = ctx_len * self.d_ff + 2 * ctx_len * self.d_ff * self.d_model
        return w1_flops + w2_flops + w3_flops