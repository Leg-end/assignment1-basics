import torch
from torch import nn


class RMSNorm(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 eps: float=1e-6,
                 device: torch.device | None = None):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        
    def forward(self, x):
        # NOTE: in practice, many implementations will
        # manually upcast the input to fp32 here to prevent overflow when you
        # square the input.
        # https://github.com/pytorch/pytorch/issues/66707
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).rsqrt().add(self.eps)
        return (x / rms * self.weight).to(in_dtype)