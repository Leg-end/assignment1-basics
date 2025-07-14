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
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = silu(self.w1(x))
        o2 = self.w3(x)
        return self.w2(o1 * o2)