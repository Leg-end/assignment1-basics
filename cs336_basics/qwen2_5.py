import torch
import os
import json
import einx
import logging
from einops import rearrange
from .RMSNorm import RMSNorm
from .Attention import scaled_dot_product_attention, softmax
from .SwiGLU import SwiGLU
from .RoPE import RotaryEmbedding, apply_rope
from .Loss import cross_entropy_loss
from torch import nn

logger = logging.getLogger(__name__)


class CasualGroupQueryAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 num_groups: int,
                 proj_bias: bool = False,
                 pos_encoder: RotaryEmbedding | None = None):
        super(CasualGroupQueryAttention, self).__init__()
        assert d_model % num_heads == 0, f"d_model({d_model}) should be divisible by num_heads({num_heads})"
        assert num_heads % num_groups == 0, f"num_heads({num_heads}) should be divisible by num_groups({num_groups})"
        self.d_model = d_model
        self.pos_encoder = pos_encoder
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, self.d_k * num_heads)
        self.kv_proj = nn.Linear(d_model, 2 * self.num_groups * self.d_k) # share within different groups
        self.o_proj = nn.Linear(self.d_k * num_heads, d_model, bias=proj_bias)
        
    def forward(self, x: torch.Tensor,
                token_positions: torch.Tensor | None = None,
                cos: torch.Tensor | None = None,
                sin: torch.Tensor | None = None) -> torch.Tensor:
        *b, T, d_model = x.size()
        assert d_model == self.d_model, f"d_model of input ({d_model}) should be equal to {self.d_model}"
        Q = self.q_proj(x)
        K, V = self.kv_proj(x).chunk(2, dim=-1)
        
        Q = rearrange(Q, '... seq (heads d) -> ... heads seq d', heads=self.num_heads)
        K = rearrange(K, '... seq (groups d) -> ... groups seq d', groups=self.num_groups)
        V = rearrange(V, '... seq (groups d) -> ... groups seq d', groups=self.num_groups)
        
        rep = self.num_heads // self.num_groups
        K = einx.rearrange('... groups seq d -> ... (groups rep) seq d', K, rep=rep)
        V = einx.rearrange('... groups seq d -> ... (groups rep) seq d', V, rep=rep)
        
        if self.pos_encoder is not None:
            if cos is not None:
                raise ValueError("Did you mean to pass both cos and sin? If so, you should pass cos=cos, sin=sin.")
            if token_positions is None:
                token_positions = einx.rearrange(
                    "seq -> b... seq", torch.arange(T, device=x.device), b = [1] * len(b))
            # duplicate for each head
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.pos_encoder(Q, token_positions)
            K = self.pos_encoder(K, token_positions)
        elif cos is not None and sin is not None:
            Q, K = apply_rope(Q, K, cos, sin, original=False)
        
        seq = torch.arange(T, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b = [1] * len(b))
        kj = einx.rearrange('key -> b... 1 1 key', seq, b = [1] * len(b))
        casual_mask = qi >= kj
        
        O = scaled_dot_product_attention(Q, K, V, casual_mask)
        O = rearrange(O, "batch heads seq d -> batch seq (heads d)").contiguous()
        output = self.o_proj(O)
        return output
    
class Block(nn.Module):
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 num_groups: int,
                 d_ff: int,
                 pos_encoder: RotaryEmbedding | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CasualGroupQueryAttention(
            d_model,
            num_heads,
            num_groups,
            pos_encoder=pos_encoder
        )
        self.ln2 = RMSNorm(d_model)
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
        
        return output
    

class Qwen2_5(nn.Module):
    
    def __init__(self,
                 vocab_size: int,
                 max_position_embeddings: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 intermediate_size: int,
                 context_scale: int = 1,
                 tie_word_embeddings: bool = True,
                 rope_theta: float | None = None,
                 **kwargs):
        super().__init__()
        self.d_model = hidden_size
        self.vocab_size = vocab_size
        self.num_layes = num_hidden_layers
        self.num_heads = num_attention_heads
        self.context_length = max_position_embeddings
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.rope = RotaryEmbedding(hidden_size // num_attention_heads, rope_theta, max_position_embeddings,
                                    context_scale=context_scale) if rope_theta is not None else None
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Block(
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                intermediate_size
            )
            for _ in range(num_hidden_layers)
        ])
        self.ln_final = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.token_embeddings.weight = self.lm_head.weight
        
        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")
        
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        B, T = x.shape
        if self.rope_theta is not None:
            cos, sin = self.rope.get_cosin(T, original=False)
        else:
            cos, sin = None, None
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)
        x = self.ln_final(x)
        if y is not None:
            logits = self.lm_head(x)
            loss = cross_entropy_loss(logits.view(-1, self.vocab_size), y.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = 50,
                 top_p: float | None = 0.9,
                 repetition_penalty: float = 1.0,
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
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if repetition_penalty != 1.0:
                score = logits.gather(1, idx)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, idx, score)
            
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
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_token_id is not None:
                if idx_next.item() == eos_token_id:
                    break
        return idx
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            if not self.tie_word_embeddings:
                n_params -= self.lm_head.weight.numel()
        elif self.tie_word_embeddings:
            n_params -= self.lm_head.weight.numel()
        return n_params
        
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
        L, H, Q, T = self.num_layes, self.num_heads, self.d_model // self.num_heads, self.context_length
        flops_per_token = 6 * N + 12 * L * H * Q * T
        return flops_per_token * T
    
    @classmethod
    def from_config(cls, config: dict):   # <class 'omegaconf.dictconfig.DictConfig'>
        model = cls(**config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        return model

    @classmethod
    def from_pretrained(cls, model_path: str, use_hf: bool = True):
        if not use_hf:
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            model = cls(**config)
            weight_path = os.path.join(model_path, "model.pt")
            state_dict = torch.load(weight_path)
            # Remove _orig_mod. prefix that comes from serializing a compiled model
            unwanted_prefix = "_orig_mod."
            for k, _ in list(state_dict.items()):
                unwanted_prefix_start_idx = k.find(unwanted_prefix)
                if unwanted_prefix_start_idx != -1:
                    state_dict[k[unwanted_prefix_start_idx + len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        else:
            from transformers import AutoModelForCausalLM, AutoConfig
            config = AutoConfig.from_pretrained(model_path).to_dict()
            model = cls(**config)
            sd = model.state_dict()
            sd_keys = sd.keys()

            model_hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto"
            )
            sd_hf = model_hf.state_dict()

            key_map = {'token_embeddings': 'embed_tokens', 'attn': 'self_attn', 'q_proj': 'q_proj',
                       'o_proj': 'o_proj', 'ffn': 'mlp', 'ln1': 'input_layernorm',
                       'ln2': 'post_attention_layernorm', 'ln_final': 'norm',
                       "w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

            def to_hf_key(key):
                components = key.split('.')
                for i, c in enumerate(components):
                    if c in key_map.keys():
                        components[i] = key_map[c]

                if not key == 'lm_head.weight':
                    key = 'model.' + '.'.join(components)
                
                return key
            # print("\n=======\n".join(sd_hf.keys()))
            cnt = 0
            for key in sd_keys:
                hf_key = to_hf_key(key)
                if 'kv_proj' in hf_key:
                    hf_key_k, hf_key_v = hf_key.replace('kv_proj', 'k_proj'), hf_key.replace('kv_proj', 'v_proj')
                    sd[key].copy_(torch.concat((sd_hf[hf_key_k], sd_hf[hf_key_v]), dim=0))
                    cnt += 2
                else:
                    # print("=" * 20)
                    # print(key, hf_key)
                    # print(sd[key].shape, sd_hf[hf_key].shape)
                    sd[key].copy_(sd_hf[hf_key])
                    cnt += 1
            print(f"Loaded {cnt} parameters from HF model. with {len(sd_hf)} parameters")
        return model
    
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