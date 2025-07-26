import torch
from torch import nn


class RMSNorm(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 eps: float=1e-5,
                 device: torch.device | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        
    def forward(self, x):
        # NOTE: in practice, many implementations will
        # manually upcast the input to fp32 here to prevent overflow when you
        # square the input.
        # https://github.com/pytorch/pytorch/issues/66707
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(in_dtype)
    
    def get_FLOPS(self, ctx_len):
        # x.pow.mean: 2LD + 1, +eps, rsqrt: +2
        rms_flops = 2 * ctx_len * self.d_model + 3
        # matrix multiply: 2LD, scalr multiply: + 1
        return rms_flops + 2 * ctx_len * self.d_model + 1