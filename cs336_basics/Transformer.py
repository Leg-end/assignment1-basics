import torch
from torch import nn
from .RMSNorm import RMSNorm
from .Attention import MultiHeadSelfAttention
from .SwiGLU import SwiGLU
from .RoPE import RoPE


class Transformer(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 pos_encoder: RoPE | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            pos_encoder=pos_encoder
        )
        self.ln2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x))
        
        output = y + self.ffn(self.ln2(y))
        
        return output
    
    
class TransformerLM(nn.Module):
    
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.rope = RoPE(d_model // num_heads, rope_theta, context_length)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Transformer(
                d_model,
                num_heads,
                d_ff,
                pos_encoder=self.rope
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, 1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        output = self.lm_head(x)
        return output
    
    def generate(self,
                 x: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 eos_token_id: int | None = None) -> torch.LongTensor:
        pass