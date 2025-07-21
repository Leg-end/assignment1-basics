import torch
import math
from torch import nn
from .RoPE import RoPE


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    Q = 1 / math.sqrt(d_k) * Q
    A = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        A = A.masked_fill(mask == 0, float('-inf'))
    output = torch.matmul(softmax(A, -1), V)
    return output


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 pos_encoder: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.pos_encoder = pos_encoder
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        qkv_proj_weight = torch.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        qkv = x @ qkv_proj_weight.T
        q, k, v = qkv.chunk(3, -1)
        
        q = q.view(q.shape[:-1] + (self.num_heads, -1)).permute(0, 2, 1, 3)
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).permute(0, 2, 1, 3)
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).permute(0, 2, 1, 3)
        
        if self.pos_encoder is not None:
            q, k = self.pos_encoder(q, k, token_positions)
        
        # TODO pass as arg
        casual_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()
        
        o = scaled_dot_product_attention(q, k, v, mask=casual_mask)
        o = o.permute(0, 2, 1, 3).contiguous()
        o = o.view(o.shape[:-2] + (self.d_model,))
        output = self.output_proj(o)
        return output
    