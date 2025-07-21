import torch
from torch import nn


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
    
    
class RoPE(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 theta: float,
                 max_seq_len: int,
                 device: torch.device | None=None):
        super().__init__()
        self.device = device
        cos, sin = self._init_cache(d_model, theta, max_seq_len)
        self.register_buffer("cos", cos.to(device), persistent=False)
        self.register_buffer("sin", sin.to(device), persistent=False)
    
    @staticmethod 
    def _init_cache(d_model: int, theta: float, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))  # [d/2]
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(1) @ inv_freq.unsqueeze(0)  # [L, 1] x [1, d/2]
        cos = torch.cos(angles).repeat_interleave(2, -1)
        sin = torch.sin(angles).repeat_interleave(2, -1)
        return cos, sin
        
    def forward(self, q: torch.Tensor, k: torch.Tensor | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = q.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=q.device).unsqueeze(0)
        cos = self.cos[token_positions]  # [B or 1, seq_len, d]
        sin = self.sin[token_positions]
        # 
        q_rotated = q * cos + rotate_half(q) * sin
        if k is not None:
            k_rotated = k * cos + rotate_half(k) * sin
            return q_rotated, k_rotated
        else:
            return q_rotated
        
        
        