import torch
from torch import nn
from .RMSNorm import RMSNorm
from .Attention import MultiHeadSelfAttention, softmax
from .SwiGLU import SwiGLU
from .RoPE import RoPE

import os
import json


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
        # pre-norm: so that it doesn't affect the main residual signal path
        y = x + self.attn(self.ln1(x))
        
        output = y + self.ffn(self.ln2(y))
        
        return output
    
    def get_FLOPS(self, ctx_len):
        return self.ln1.get_FLOPS(ctx_len) + self.attn.get_FLOPS(ctx_len) \
            + self.ln2.get_FLOPS(ctx_len) + self.ffn.get_FLOPS(ctx_len)
    
    
class TransformerLM(nn.Module):
    
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.rope = RoPE(d_model // num_heads, rope_theta, context_length) if rope_theta is not None else None
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
        
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = x.to(torch.long)
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        output = self.lm_head(x)
        return output
    
    def get_num_params(self, non_embedding=True):
        n_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            non_embedding -= self.lm_head.weight.numel()
        return n_param
    
    @torch.no_grad()
    def generate(self,
                 x: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 eos_token_id: int | None = None) -> torch.LongTensor:
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        ori_seq_len = x.size(-1)
        for _ in range(max_new_tokens):
            # Always padding left, thus model see meaning token at right side
            x = x[:, -self.context_length:] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1]
            temp_scaled_next_token_logits = next_token_logits / temperature
            if top_k:
                topk_values, _ = torch.topk(temp_scaled_next_token_logits,
                                            k=min(top_k, temp_scaled_next_token_logits.size(-1)))
                threshold = topk_values[:, -1]
                topk_mask = temp_scaled_next_token_logits < threshold
                temp_scaled_next_token_logits.masked_fill_(topk_mask, float("-inf"))
            next_token_prob = softmax(temp_scaled_next_token_logits, dim=-1)
            # sample from a multinomial with model generated probability
            next_token_id = torch.multinomial(next_token_prob, 1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat([x, next_token_id], dim=-1)
        new_token_ids = x[:, ori_seq_len:]
        return new_token_ids
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        
        weight_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weight_path)
        
        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
    
    
    def get_FLOPS(self):
        flops = sum(layer.get_FLOPS(self.context_length) for layer in self.layers)
        flops += self.ln_final.get_FLOPS(self.context_length)
        flops += 2 * self.context_length * self.vocab_size * self.d_model
        return flops
    
    
    def get_mem(self, dtype=torch.float16):
        unit = torch.finfo(dtype).bits // 8
        return self.get_num_params() * unit