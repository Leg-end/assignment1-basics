import torch
import logging
from typing import Callable, AsyncGenerator, Union, Optional
from torch import nn
from .Tokenizer import BPETokenizer
from .linear import Linear, Embedding
from .RMSNorm import RMSNorm
from .Attention import CasualMultiHeadSelfAttention, softmax
from .SwiGLU import SwiGLU
from .RoPE import RotaryEmbedding
from .Loss import cross_entropy_loss
import os
import json
import asyncio

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 pos_encoder: RotaryEmbedding | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.attn = CasualMultiHeadSelfAttention(
            d_model,
            num_heads,
            pos_encoder=pos_encoder
        )
        self.ln2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor,
                cos: torch.Tensor | None = None,
                sin: torch.Tensor | None = None) -> torch.Tensor:
        # pre-norm: so that it doesn't affect the main residual signal path
        y = x + self.attn(self.ln1(x), cos=cos, sin=sin)
        
        output = y + self.ffn(self.ln2(y))
        
        # post-norm: hard to train, but learn more robust representations
        # y = self.ln1(x + self.attn(x, cos=cos, sin=sin))
        
        # output = self.ln2(y + self.ffn(y))
        
        # parallel layers
        # output = x + self.attn(self.ln1(x), cos=cos, sin=sin) + self.ffn(self.ln2(x))
        
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
                 context_scale: int = 1,
                 tie_word_embeddings: bool = False,
                 rope_theta: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        if rope_theta is not None:
            self.pos_embeddings = RotaryEmbedding(d_model // num_heads, rope_theta, context_length,
                                        context_scale=context_scale)
        else:
            # self.pos_embeddings = Embedding(context_length, d_model)
            self.pos_embeddings = None
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        if tie_word_embeddings:
            self.token_embeddings.weight = self.lm_head.weight
        
        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")
        
    def forward(self, x: torch.IntTensor, y: torch.IntTensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        x = self.token_embeddings(x)
        if self.rope_theta is not None:
            cos, sin = self.pos_embeddings.get_cosin(T)
        else:
            cos, sin = None, None
            # pos_indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            # x += self.pos_embeddings(pos_indices)
        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)
        x = self.ln_final(x)
        if y is not None:
            logits = self.lm_head(x)
            loss = cross_entropy_loss(logits.view(-1, self.vocab_size), y.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # change to x to pass test
            loss = None
        return logits, loss
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            if not self.tie_word_embeddings:
                n_params -= self.lm_head.weight.numel()
        elif self.tie_word_embeddings:
            n_params -= self.lm_head.weight.numel()
        return n_params
    
    def sample(self,
               input_ids: torch.Tensor,
               full_ids: torch.Tensor,
               temperature: float = 1.0,
               top_k: int | None = 50,
               top_p: float | None = 0.9,
               repetition_penalty: float = 1.0) -> torch.LongTensor:
        logits, _ = self(input_ids)
        logits = logits[:, -1, :]
        
        if repetition_penalty != 1.0:
            score = logits.gather(1, full_ids)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits.scatter_(1, full_ids, score)
        
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        probs = softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next
    
    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = 50,
                 top_p: float | None = 0.9,
                 repetition_penalty: float = 1.0,
                 stop_token_ids: list[int] | None = None,
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
            top_p: float
                If provided, only sample from the smallest set of vocab items with cumulative probability >= top_p.
            repetition_penalty: float
                Penalty to apply to repeated tokens (1.0 = no penalty, >1.0 = penalty).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            idx_next = self.sample(input_ids=idx_cond,
                                   full_ids=idx,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_token_id is not None:
                if idx_next.item() == eos_token_id:
                    break
            if stop_token_ids is not None:
                if idx_next.item() in stop_token_ids:
                    break
        return idx
    
    # @torch.no_grad()
    # def generate_async(self,
    #                    prompt: str,
    #                    max_new_tokens: int,
    #                    tokenizer: BPETokenizer,
    #                    temperature: float = 1.0,
    #                    top_k: Optional[int] = None,
    #                    stop_tokens: Optional[list[str]] = None) -> AsyncGenerator[str, None]:
    #     input_ids = tokenizer.encode(prompt)
    #     x = torch.tensor([input_ids], device=self.device).to(torch.int64)
        
    #     ori_seq_len = x.size(-1)
    #     generated_tokens = []
    #     stop_token_ids = []
    #     if stop_tokens:
    #         for token in stop_tokens:
    #             token_ids = tokenizer.encode(token)
    #             stop_token_ids.append(token_ids)
        
    #     eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        
        
    #     # 开始生成
    #     for step in range(max_new_tokens):
    #         # 让出控制权，实现真正的异步
    #         await asyncio.sleep(0)
            
    #         logits, _ = self.forward(x[:, -self.context_length:] if x.size(-1) > self.context_length else x)
    #         next_token_logits = logits[:, -1]
    #         temp_scaled_next_token_logits = next_token_logits / temperature
    #         if top_k:
    #             topk_values, _ = torch.topk(temp_scaled_next_token_logits,
    #                                         k=min(top_k, temp_scaled_next_token_logits.size(-1)))
    #             threshold = topk_values[:, -1]
    #             topk_mask = temp_scaled_next_token_logits < threshold
    #             temp_scaled_next_token_logits.masked_fill_(topk_mask, float("-inf"))
    #         next_token_prob = softmax(temp_scaled_next_token_logits, dim=-1)
    #         # sample from a multinomial with model generated probability
    #         next_token_id = torch.multinomial(next_token_prob, 1)
    #         token_id = next_token_id.item()
            
    #         generated_tokens.append(token_id)
    #         new_text = self.tokenizer.decode([token_id])
    #         yield new_text
            
    #         if eos_token_id is not None and token_id == eos_token_id:
    #             break
    #         if stop_token_ids and token_id in stop_token_ids:
    #             break
            
    #         x = torch.cat((x, next_token_id), dim=-1)
    
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