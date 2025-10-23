import torch
import math
import einx
from einops import rearrange, einsum
from torch import nn
from .RoPE import RoPE
from .linear import Linear


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
    A = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        A = A.masked_fill(mask == 0, float('-inf'))
    output = einsum(softmax(A, -1), V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output


class CasualMultiHeadSelfAttention(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 pos_encoder: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.pos_encoder = pos_encoder
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, self.d_k * num_heads)
        self.k_proj = Linear(d_model, self.d_k * num_heads)
        self.v_proj = Linear(d_model, self.d_k * num_heads)
        self.output_proj = Linear(d_model, self.d_k * num_heads)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *b, seq_len, d_model = x.size()
        assert d_model == self.d_model, f"d_model of input ({d_model}) should be equal to {self.d_model}"
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q, K, V = (rearrange(X,  "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
                   for X in (Q, K, V))

        if self.pos_encoder is not None:
            if token_positions is None:
                token_positions = einx.rearrange(
                    "seq -> b... seq", torch.arange(seq_len, device=x.device), b = [1] * len(b))
            # duplicate for each head
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            Q, K = self.pos_encoder(Q, K, token_positions)
        
        seq = torch.arange(seq_len, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b = [1] * len(b))
        kj = einx.rearrange('key -> b... 1 1 key', seq, b = [1] * len(b))
        casual_mask = qi >= kj
        
        O = scaled_dot_product_attention(Q, K, V, mask=casual_mask)
        O = rearrange(O, "... heads seq d -> ... seq (heads d)").contiguous()
        output = self.output_proj(O)
        return output
    
    def get_FLOPS(self, ctx_len):
        qkv_flops = 3 * 2 * ctx_len * self.d_model * self.d_model
        if self.pos_encoder is not None:
            qkv_flops + self.pos_encoder.get_FLOPS(ctx_len)
        # num_head * each head's QK^T
        qk_flops = 2 * ctx_len * ctx_len * self.d_model
        # num_head * each head's softmax
        softmax_flops = 3 * ctx_len * ctx_len * self.num_heads
        # num_head * each head's AV
        v_flops = 2 * ctx_len * ctx_len * self.d_model
        o_flops = 2 * ctx_len * self.d_model * self.d_model
        return qkv_flops + qk_flops + softmax_flops + v_flops + o_flops
    