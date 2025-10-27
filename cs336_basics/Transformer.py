import torch
import logging
from torch import nn
from .linear import Linear, Embedding
from .RMSNorm import RMSNorm
from .Attention import CasualMultiHeadSelfAttention, softmax
from .SwiGLU import SwiGLU
from .RoPE import RotaryEmbedding
from .Loss import cross_entropy_loss
from torch.nn import functional as F

import os
import json

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 pos_encoder: RotaryEmbedding | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CasualMultiHeadSelfAttention(
            d_model,
            num_heads,
            pos_encoder=pos_encoder
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm: so that it doesn't affect the main residual signal path
        y = x + self.attn(self.ln1(x))
        
        output = y + self.ffn(self.ln2(y))
        
        return output
    
    def get_FLOPS(self, ctx_len):
        return self.ln1.get_FLOPS(ctx_len) + self.attn.get_FLOPS(ctx_len) \
            + self.ln2.get_FLOPS(ctx_len) + self.ffn.get_FLOPS(ctx_len)
    
    
class BasicsTransformerLM(nn.Module):
    """
    Memory allocation:
        Model Parameters (P)
            Output Embedding: vocab_size * d_model
            Transformer Block per layer (L layers):
                Self-Attention(Q, K, V, O): 4 * d_model^2
                Feed-Forward(w1, w2, w3, dff = 4 * d_model): 3 * d_ff * d_model = 12 * d_model^2
                RMSNorm: d_model
            Final RMSNorm: d_model
            P = (vocab_size + L + 1) * d_model + 16 * L * d_model^2
        Activation (A):
            Embeddings: batch_size * S * d_model
            Transformer Block per layer (L layers):
                Self-Attention(Q^T*K matrix): 
                    QKV Projection: 3 * batch_size * S * d_model
                    Q^T * K: batch_size * num_heads * S * S
                    Softmax: batch_size * num_heads * S * S
                    Attn * V: batch_size * num_heads * S * d_model
                    Outout Projection: batch_size * S * d_model
                    Total: (num_heads + 4) * batch_size * S * d_model + 2 * batch_size * num_heads * S^2
                Feed-Forward Network:
                    W1: batch_size * S * d_ff = batch_size * S * 4 * d_model
                    SiLU: batch_size * S * d_ff = batch_size * S * 4 * d_model
                    W3: batch_size * S * d_ff = batch_size * S * 4 * d_model
                    W2: batch_size * S * d_model
                    Total: 13 * batch_size * S * d_model
                RMSNorm:
                    input: batch_size * S * d_model
                    rms: batch_size * S * d_model
                    batch_size * S * d_model
                    Total: 2 * batch_size * S * d_model
            Final RMSForm: 2 * batch_size * S * d_model
            Output Layer:
                Output(logits): batch_size * S * vocab_size
                Cross-Entropy: batch_size * S * vocab_size
        Gradients = 4P bytes
        Optimizer State = 3 * 4P bytes
    Total Peak Memory: 20 PB + 4AB

    """
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float | None = None):
        super().__init__()
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.rope = RotaryEmbedding(d_model // num_heads, rope_theta, context_length) if rope_theta is not None else None
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                pos_encoder=self.rope
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")
        
    def forward(self, x: torch.IntTensor, y: torch.IntTensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        if y is not None:
            logits = self.lm_head(x)
            loss = cross_entropy_loss(logits.view(-1, self.vocab_size), y.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params
    
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
            logits, _ = self.forward(x)
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
            unwanted_prefix_start_idx = k.find(unwanted_prefix)
            if unwanted_prefix_start_idx != -1:
                state_dict[k[unwanted_prefix_start_idx + len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
    
    
    def get_FLOPS(self):
        """
        FLOPs: one multiplication or one addition is counted as 1 FLOP
        Matmul FLOPs: including m * n * p multiplication and m * n * p addition, thus 2 * (m * n * p) FLOPs
        calculate flops per token = 6 * N + 12 * L * H * Q * T
            6 * N: matmul per token (for weights)
                2 * N: forward matrix multiplication (one multiplication and one addition)
                4 * N: backward matrix multiplication (grads to input and weights)
                    e.g. Y = XW
                    2N: dX = dY @ W.T
                    2N: dW = X.T @ dY
            12 * L * H * Q * T: self-attention flops T tokens (for tensors)
                (Note that QKV and O projection already included in N params)
                per head per layer:
                    forward:
                        A = QK^T: L * H * 2 * T * Q * T = 2 * T^2 * Q
                        O = AV: L * H * 2 * T * T * Q = 2 * T^2 * Q
                    backward:
                        dQ = dA @ K; dK = Q.T @ dA: 4 * T^2 * Q
                        dA = dO @ V.T; dV = A.T @ dO: 4 * T^2 * Q
                    total: 12 * T^2 * Q
                Total: 12 * L * H * T^2 * Q
                Total per token = 12 * L * H * T^2 * Q / T = 12 * L * H * Q * T
        """
        N = self.get_num_params()
        L, H, Q, T = self.num_layers, self.num_heads, self.d_model // self.num_heads, self.context_length
        flops_per_token = 6 * N + 12 * L * H * Q * T
        return flops_per_token * T
    
    
    def get_mem(self, dtype=torch.float16):
        unit = torch.finfo(dtype).bits // 8
        return self.get_num_params() * unit
    
    def get_MFU(self, fwdbwd_per_iter: int, dt: float) -> float:
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS 
        Args:
            fwdbwd_per_iter: number of forward-backward passes per iteration
            dt: time per iteration in seconds
        Step 1:
            calculate flops per token = 6 * N + 12 * L * H * Q * T
        Step 2:
            calculate flops per forward-backword iteration = (12LHQT + 6N) * T
        Step 3:
            calculate flops per iteration
            Note that for gradient accumulation, we may have fwdbwd_per_iter = batch_size / accumulation_steps
        Step 4:
            calculate flops per second = flops per iteration / seconds per iteration
        Step 5:
            mfu = flops per second / A100 bfloat16 peak flops per seconds
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        flops_per_fwdbwd = self.get_FLOPS()
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu