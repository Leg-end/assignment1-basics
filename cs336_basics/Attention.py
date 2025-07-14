import torch
import math
from torch import nn
from .RoPE import RoPE


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    return x_exp - torch.sum(x_exp, dim=dim, keepdim=True)
    


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
    if mask is not None:
        A = torch.baddbmm(mask, Q, K.transpose(-2, -1))
    else:
        A = torch.bmm(Q, K.transpose(-2, -1))
    output = torch.bmm(softmax(A, -1), V)
    return output


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 pos_encoder: RoPE | None = None):
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
        qkv = x @ qkv_proj_weight
        q, k, v = qkv.chunk(3, -1)
        
        q = q.transpose(0, 1).view(seq_len, -1, self.d_model // self.num_heads).transpose(0, 1)
        k = k.transpose(0, 1).view(seq_len, -1, self.d_model // self.num_heads).transpose(0, 1)
        v = v.transpose(0, 1).view(seq_len, -1, self.d_model // self.num_heads).transpose(0, 1)
        
        if self.pos_encoder is not None:
            q, k = self.pos_encoder(q, k, token_positions)
        
        # TODO pass as arg
        casual_mask = torch.empy(seq_len, seq_len)
        casual_mask.fill_(float('-inf'))
        casual_mask.triu_(1)
        
        o = scaled_dot_product_attention(q, k, v, mask=casual_mask)
        o = o.view(seq_len, -1, self.d_model).transpose(0, 1)
        output = self.output_proj(o)
        return output
    