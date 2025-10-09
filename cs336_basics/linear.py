from torch import nn
from typing import Optional
from jaxtyping import Float, Int
from einops import einsum
import torch
import math

class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None):
        super(Linear, self).__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )
        
    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    def extra_repr(self):
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}"
    

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None):
        super(Embedding, self).__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )
        
    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_model"]:
        return self.weight[token_ids, :]
    
    def extra_repr(self):
        return f"num_embeddings={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]}"