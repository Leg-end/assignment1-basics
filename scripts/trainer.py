import comet_ml
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import BPETokenizer, get_adamw_cls, run_get_lr_cosine_schedule,\
    run_load_checkpoint, run_save_checkpoint, run_get_batch, clip_grad_norm, BasicsTransformerLM
from cs336_basics.qwen2_5 import Qwen2_5
# from cs336_basics.model import BasicsTransformerLM
import torch
import math
import logging
import hydra
import hashlib
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm, trange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rich.pretty import pprint as pprint
from rich.traceback import install
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.inference import ChatBot
from typing import Union, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

install(show_locals=True)

# GPU 峰值 FLOPS 表 (bfloat16/float16)
GPU_PEAK_FLOPS = {
    "A100": 312e12,      # 312 TFLOPS
    "A800": 312e12,      # 与 A100 相同
    "H100": 989e12,      # 989 TFLOPS
    "V100": 125e12,      # 125 TFLOPS
    "RTX3090": 142e12,   # 142 TFLOPS
    "RTX4090": 330e12,   # 330 TFLOPS
    "T4": 65e12,         # 65 TFLOPS
}

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def val_batch_iter(memmap: np.ndarray, batch_size: int, context_length: int, device: str | torch.device):
    N = len(memmap)
    steps = (N-context_length-1) // batch_size
    for i in range(steps):
        start = i * batch_size
        end = start + batch_size
        x = np.stack([memmap[j: j + context_length] for j in range(start, end)])
        y = np.stack([memmap[j+1: j + context_length+1] for j in range(start, end)])
        yield torch.tensor(x).to(device).long(), torch.tensor(y).to(device).long()
        
def setup_hydra_output_for_distributed():
    """在Hydra初始化前调用此函数"""
    # 判断当前进程是否是rank 0
    if dist.is_available() and dist.is_initialized():
        is_rank_zero = dist.get_rank() == 0
    else:
        is_rank_zero = int(os.environ.get('RANK', 0)) == 0
    
    # 如果不是rank 0，设置环境变量禁用Hydra输出
    if not is_rank_zero:
        os.environ['HYDRA_FULL_ERROR'] = '0'  # 减少错误输出
        # 重定向输出到空设备
        os.environ['HYDRA_OUTPUT'] = 'null'
    else:
        logger.info("Rank 0 process, enabling Hydra output")
        

class Trainer:
    """统一的训练器类，支持自动 batch_size 推荐和时间预估"""
    
    def __init__(
        self,
        model: BasicsTransformerLM,
        args: Union[DictConfig, Dict],
        tokenizer: BPETokenizer,
    ):
        """
        Args:
            model: 模型实例
            args: 训练参数 (支持 DictConfig 或 dict)
            tokenizer: tokenizer 实例
            device: 设备类型
        """
        self.model = model
        self.args = args
        self.training_cfg = args.training
        self.model_cfg = args.model
        self.paths_cfg = args.paths
        self.infer_cfg = args.inference
        self.tokenizer = tokenizer
        self.device = self.training_cfg.device
        self.dtype = DTYPE_MAP[self.training_cfg.dtype]
    
    def setup_distributed(self):
        """设置分布式训练环境"""
        self.is_ddp = int(os.environ.get("RANK", -1)) != -1
        
        if self.is_ddp:
            init_process_group(backend='nccl')
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = "cuda"  # model will be moved to CUDA device in current GPU
            torch.cuda.set_device(f"cuda:{ddp_local_rank}")
            seed = self.training_cfg.seed + ddp_rank  # each process gets a different seed
            self.is_master_process = ddp_rank == 0
            
            # 同步所有进程，确保都准备好了
            dist.barrier()
            # Calculate free GPU memory of each GPU
            local_free_mem, _ = torch.cuda.mem_get_info()
            local_tensor = torch.tensor([local_free_mem]).cuda()
            all_tensors = [torch.zeros(1).cuda() for _ in range(ddp_world_size)]
            
            dist.all_gather(all_tensors, local_tensor)
            
            # 计算最小值
            self.min_free_mem = min([t.item() for t in all_tensors])

            if self.is_master_process:
                logger.info("Using DDP with world size: {}".format(ddp_world_size))
                all_tensors = all_tensors / (1024 ** 3)
                free_list = [f"GPU{i}: {t.item():.2f} GB" for i, t in enumerate(all_tensors)]
                logger.info(f"Available memory per GPU: {', '.join(free_list)}")
        else:
            seed = self.training_cfg.seed
            ddp_world_size = 1
            self.is_master_process = True
            self.min_free_mem, _ = torch.cuda.mem_get_info()
        
        # Seed each process differently so we can be sure that they
        # see different data batches.
        # NOTE: This assumes that you're using torch RNG, you may have
        # to seed numpy too as well if your code uses numpy random functions.
        torch.manual_seed(seed)
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        # compile the model, requires torch 2.0
        if self.training_cfg.compile:
            self.model = torch.compile(self.model)
        
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[ddp_local_rank])
    
    def setup_optimizer(self):
        """Create optimizer
        Set up the AdamW optimizer.
        First, we need to group the parameters that should be decayed and those that shouldn't.
        In particular, we do not apply decay on 1D parameters (e.g., biases and RMSNorms)
        filter out those that do not require grad
        """
        # 过滤需要梯度的参数
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
        params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {"params": params_to_decay, "weight_decay": self.training_cfg.weight_decay},
            {"params": params_to_not_decay, "weight_decay": 0.0},
        ]
        
        self.optimizer = get_adamw_cls()(
            optim_groups,
            lr=self.training_cfg.lr,
            betas=(self.training_cfg.beta1, self.training_cfg.beta2),
            eps=self.training_cfg.eps,
            weight_decay=self.training_cfg.weight_decay
        )
                                                                                                                                       
    def setup_comet_experiment(self):
        """设置 Comet ML 实验"""
        comet_cfg = self.args.comet
        api_key = comet_cfg.api_key
        project_name = comet_cfg.project
        workspace = comet_cfg.workspace
        resume = self.training_cfg.resume_checkpoint
        run_id = comet_cfg.run_id
        
        if self.is_master_process and not resume:  # always create a new experiment if not resuming
            experiment_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
            api = comet_ml.API(api_key=api_key)  # Assumes API key is set in config/env
            api_experiment = api.get_experiment_by_key(experiment_id)
            
            if api_experiment is not None:
                logger.warning(f"Experiment {run_id} already exists, using random key")
                experiment_id = comet_ml.get_experiment_key(None)
            
            os.environ["COMET_EXPERIMENT_KEY"] = experiment_id
            self.experiment = comet_ml.start(
                api_key=api_key,
                workspace=workspace,
                project_name=project_name,
                experiment_key=experiment_id,
                experiment_config=comet_cfg.get("exp_cfg", None)
            )
            self.experiment.log_parameters(self.args)
        else:
            experiment_id = os.environ["COMET_EXPERIMENT_KEY"]
            api = comet_ml.API(api_key=api_key)  # Assumes API key is set in config/env
            api_experiment = api.get_experiment_by_key(experiment_id)
            
            if api_experiment is not None:
                self.experiment = comet_ml.ExistingExperiment(
                    project_name=project_name,
                    api_key=api_key,
                )
            else:
                self.experiment = comet_ml.Experiment(
                    project_name=project_name,
                    workspace=workspace,
                    api_key=api_key
                )
    
    def load_checkpoint(self,
                        resume_checkpoint: Optional[int] = None,
                        model_output: Optional[str] = None):
        """加载检查点"""
        resume_checkpoint = resume_checkpoint or self.paths_cfg.resume_checkpoint
        if model_output is not None:
            model_output = Path(model_output)
        else:
            model_output = Path(HydraConfig.get().runtime.output_dir) / self.paths_cfg.model_output
        
        try:
            resume_ckpt_path = model_output / f"step_{resume_checkpoint:010d}" / "model.pt"
            
            if self.is_ddp:
                dist.barrier()
            
            # 文件系统可能存在不一致时（如NFS、分布式存储）
            if not resume_ckpt_path.exists():
                if self.is_master_process:
                    logger.error(f"Checkpoint not found: {resume_ckpt_path}")
                if self.is_ddp:
                    dist.barrier()  # 通知其他进程失败
                raise FileNotFoundError(f"Checkpoint {resume_checkpoint} not found at {resume_ckpt_path}")
            
            if self.is_master_process:
                if self.is_ddp:
                    checkpoint = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
                else:
                    checkpoint = resume_ckpt_path
            else:
                checkpoint = None
            
            if self.is_ddp:
                checkpoint_list = [checkpoint]
                dist.broadcast_object_list(checkpoint_list, src=0)
                checkpoint = checkpoint_list[0]
                dist.barrier()
                
            start_iter = run_load_checkpoint(
                checkpoint, 
                self.model, 
                self.optimizer
            )
            
            if self.is_master_process:
                logger.info(f"Resumed from checkpoint {resume_checkpoint} at iteration {start_iter} from path {resume_ckpt_path}")
            
            if self.is_ddp:
                # Synchronize all processes til all processes have done loading.
                dist.barrier()
                if self.is_master_process:
                    logger.info("All processes have loaded the checkpoint")
            return start_iter
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            if self.is_ddp:
                destroy_process_group()
            raise
    
    def get_num_params(self) -> int:
        """
        Model Parameters (P)
            Output Embedding: vocab_size * d_model
            Transformer Block per layer (L layers):
                Self-Attention(Q, K, V, O): 4 * d_model^2
                Feed-Forward(w1, w2, w3, dff = 4 * d_model): 3 * d_ff * d_model = 12 * d_model^2
                RMSNorm: d_model
            Final RMSNorm: d_model
            P = (vocab_size + L + 1) * d_model + 16 * L * d_model^2
        """
        L = self.model_cfg.num_layers
        D = self.model_cfg.d_model
        
        emb_param = self.model_cfg.vocab_size * D
        # Self-Attention(Q, K, V, O) + Feed-Forward + RMSNorm
        block_param = 4 * D * D + 3 * self.model_cfg.d_ff * D + D
        P = emb_param + L * block_param + D
        return P
    
    def estimate_FLOPS(self) -> int:
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
        model_cfg = self.model_cfg
        N = self.get_num_params()
        L = model_cfg.num_layers
        H = model_cfg.num_heads
        D = model_cfg.d_model
        Q = D // H
        T = model_cfg.context_length
        flops_per_token = 6 * N + 12 * L * H * Q * T
        return flops_per_token * T
    
    def estimate_MFU(self, fwdbwd_per_iter: int, dt: float) -> float:
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
        flops_per_fwdbwd = self.estimate_FLOPS()
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = GPU_PEAK_FLOPS["RTX4090"]  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    def estimate_activation_memory(
        self,
        batch_size: int,
        dtype: torch.dtype,
        use_flash_attention: bool = False,
        use_checkpointing: bool = True) -> float:
        unit = torch.finfo(dtype).bits // 8
        model_cfg = self.model_cfg
        L = model_cfg.num_layers
        H = model_cfg.num_heads
        D = model_cfg.d_model
        T = model_cfg.context_length
        activation_mem = 0
        
        if use_checkpointing:
            # 使用重计算时，只存储每层的输入，反向时重新计算中间结果
            # 激活值从 O(L) 降到 O(1)，但需要额外的前向时间
            # 这里简化: 只存储每层的输入，不存储中间激活
            activation_mem = T * D  # 只存输入
        else:
            # ============ 1. Attention 部分 ============
            if use_flash_attention:
                # Flash Attention: 不需要存储完整的 attention weights (SxS)
                # 只需要存储 Q, K, V 投影结果和少量统计信息
                # 参考: https://arxiv.org/abs/2205.14135
                qkv_proj = 3 * T * D  # Q, K, V 投影
                attn_stats = H * T * 2 # softmax 统计量
                attn_output = T * D
                activation_mem += qkv_proj + attn_stats + attn_output
            else:
                # 标准 Attention: 需要存储完整的 attention weights (主要瓶颈)
                # 1. Q, K, V 投影结果
                qkv_proj = 3 * T * D
                # 2. Q, K, O 矩阵 (用于计算 dQ, dK, dO)
                qkv_deriv = 3 * T * D
                # 3. Attention weights (用于计算dV和dA)
                #   shape: [batch, heads, seq_len, seq_len]
                attn_weights = H * T * T
                # 4. Attention output (用于计算dO)
                attn_output = T * D
                activation_mem += qkv_proj + qkv_deriv + attn_weights + attn_output
                
            # ============ 2. FFN 部分 (SwiGLU) ============
            d_ff = model_cfg.d_ff
            # SwiGLU: y = W2 * (SiLU(W1 * x) * (W3 * x))
            # 需要保留:
            #   - 输入 x (用于 dW2)
            #   - W1 输出 (用于 SiLU 和 dW1)
            #   - W3 输出 (用于 dW3)
            #   - 门控输出 (SiLU(W1*x) * (W3*x)) (用于 dW1/dW3)
            ffn_input = T * D
            w1_output = T * d_ff
            w3_output = T * d_ff
            gated_output = T * d_ff
            activation_mem += ffn_input + w1_output + w3_output + gated_output
            
            # ============ 3. LayerNorm/RMSNorm ============
            # 需要保留归一化前的输入用于反向传播
            norm_input = T * D
            norm_output = T * D
            activation_mem += norm_input + norm_output
            
        # 额外激活值 (非每层)
        # - Embeddings: 输入嵌入
        # - Final norm: 最后一层 LayerNorm 的输出
        # - Loss: 交叉熵计算需要的 labels (通常很小)
        extra_activations = (
            T * D +           # embeddings
            T * D +           # final norm
            T                 # labels
        )
        activation_mem = batch_size * (activation_mem * L + extra_activations)
        activation_mem = activation_mem * unit
        return activation_mem
    
    def estimate_train_memory(
        self,
        dtype: torch.dtype,
        batch_size: Optional[int] = None,
        include_activation: bool = True,
        use_flash_attention: bool = False,
        use_checkpointing: bool = True) -> float:
        N = self.get_num_params()
        unit = torch.finfo(dtype).bits // 8
        model_mem = unit * N
        gradient_mem = model_mem
        # Adam 优化器: 3P (参数副本 + 一阶矩 + 二阶矩，都是 fp32)
        optimizer_mem = 3 * N * 4
        peak_mem = model_mem + gradient_mem + optimizer_mem
        if include_activation:
            peak_mem += self.estimate_activation_memory(
                batch_size=batch_size or self.training_cfg.train_batch_size,
                dtype=dtype,
                use_flash_attention=use_flash_attention,
                use_checkpointing=use_checkpointing
            )
        return peak_mem
    
    def estimate_training_time(
        self,
        batch_size: int,
        mfu: float = 0.45) -> float:
        # 每个序列的 FLOPs
        flops_per_seq = self.estimate_FLOPS()
        # 每个 forward-backward 的 FLOPs
        # 注意：fwdbwd_per_iter = batch_size / grad_accum_steps
        flops_per_iter = flops_per_seq * batch_size
        
        # 计算总迭代次数
        tokens_per_iter = batch_size * self.model_cfg.context_length
        total_iters = self.args.total_tokens // tokens_per_iter
        
        # 考虑梯度累积
        effective_tokens_per_iter = tokens_per_iter * self.training_cfg.gradient_accumulation_steps
        total_effective_iters = self.args.total_tokens // effective_tokens_per_iter
        
        # 计算每秒可处理的 FLOPs
        peak_flops = GPU_PEAK_FLOPS["RTX4090"]
        achievable_flops = peak_flops * mfu
        
        # 计算时间
        seconds_per_iter = flops_per_iter / achievable_flops
        total_seconds = seconds_per_iter * total_effective_iters
        
        # 转换为可读格式
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.1f}s"
        
    def check_before_training(self):
        """
        1. check batch size
        2. esitmate memory usage
        3. estimate training time
        """
        train_mem = self.estimate_train_memory(self.dtype, include_activation=False)
        remain_mem = self.min_free_mem - train_mem 
        if remain_mem <= 0:
            raise ValueError(f"OOM: current available GPU memory: {self.min_free_mem / (1024 ** 3):.2f} GB")
        activation_mem_per_sample = self.estimate_activation_memory(1, self.dtype)
        recommend_bs = remain_mem // activation_mem_per_sample
        batch_size = min(recommend_bs, self.training_cfg.train_batch_size)
        occupied_mem = train_mem + batch_size * activation_mem_per_sample
        
        ddp_world_size = 1
        if self.is_ddp:
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            # 将 batch_size 广播到所有进程
            batch_tensor = torch.tensor([batch_size]).cuda()
            dist.broadcast(batch_tensor, src=0)
            synced_bs = batch_tensor.item()
            
            if synced_bs != batch_size and self.is_master_process:
                logger.warning(f"Batch size adjusted from {batch_size} to {synced_bs} for consistency")
            
            batch_size = synced_bs
        self.training_cfg.train_batch_size = batch_size
        
        if self.is_master_process:
            gb_unit = 1024 ** 3
            training_time = self.estimate_training_time(batch_size)
            logger.info(f"Recommended batch_size: {recommend_bs}, current batch_size: {batch_size}")
            logger.info(f"Estimated memory usage: {occupied_mem / gb_unit:.2f} GB / {self.min_free_mem / gb_unit:.2f} GB")
            logger.info(f"Estimated training time: {training_time}")
            logger.info(
                "Total number of tokens per training step: "
                + str(
                    self.training_cfg.gradient_accumulation_steps
                    * ddp_world_size
                    * self.training_cfg.train_batch_size
                    * self.model_cfg.context_length
                )
            )
    
    def train_step(
        self,
        step: int,
        train_dataset: np.ndarray,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor
    ) -> Tuple[float, float, float]:
        """执行单个训练步骤"""
        # 获取当前学习率
        lr = run_get_lr_cosine_schedule(
            step,
            self.training_cfg.lr,
            self.training_cfg.min_lr,
            self.training_cfg.train_steps * self.training_cfg.warmup_ratio,
            self.training_cfg.train_steps,
        )
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 梯度累积
        grad_accum_steps = self.training_cfg.gradient_accumulation_steps
        total_loss = 0.0
        
        for micro_step in range(grad_accum_steps):
            # DDP 梯度同步控制
            if self.is_ddp: # When using DDP, don't all-reduce gradients until the last step.
                self.model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            # 混合精度上下文
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                logits, loss = self.model(batch_x, batch_y)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                next_batch_x, next_batch_y = run_get_batch(
                    train_dataset,
                    self.training_cfg.train_batch_size,
                    self.model_cfg.context_length,
                    self.device
                )
                loss = loss / grad_accum_steps
            
            loss.backward()
            batch_x = next_batch_x
            batch_y = next_batch_y
        
        # 梯度裁剪
        gnorm = clip_grad_norm(self.model.parameters(), self.training_cfg.max_grad_norm)
        
        # 更新参数
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        # 计算 perplexity
        loss_float = loss.item() * grad_accum_steps
        ppl = math.exp(loss_float)
        
        return {"loss": total_loss,
                "ppl": ppl,
                "lr": lr,
                "gnorm": gnorm.item()}
    
    @torch.no_grad()
    def validate(self,
                 valid_dataset: np.ndarray,
                 test_infer: bool = False) -> Tuple[float, float]:
        """执行验证"""
        self.model.eval()
        eval_iters = self.training_cfg.eval_iters
        losses = torch.zeros(eval_iters, device=self.device)
        for k in tqdm(range(eval_iters), desc="Eval"):
            batch_x, batch_y =  val_batch_iter(
                valid_dataset,
                batch_size=self.training_cfg.eval_batch_size,
                context_length=self.model_cfg.context_length,
                device=self.device
            )
            val_logits, val_loss = self.model(batch_x, batch_y)
            losses[k] = val_loss.item()
        loss = losses.mean().item()
        ppl = math.exp(loss)
        ret = {"eval_loss": loss,
                "eval_ppl": ppl}
        if test_infer:
            response = self.inference(
                tokenizer=self.tokenizer,
                **self.infer_cfg)
            ret["response"] = response
        self.model.train()
        return ret
    
    @torch.no_grad()
    def inference(self,
                  tokenizer: BPETokenizer,
                  prompts: Optional[str | list[str]] = None,
                  max_new_tokens: Optional[int] = None,
                  temperature: Optional[float] = None,
                  top_k: Optional[int] = None,
                  top_p: float = 0.9,
                  repetition_penalty: float = 1.0) -> str:
        """生成样本"""
        if isinstance(prompts, str):
            prompts = [prompts]
        responses = []
        # 获取原始模型
        model = self.model.module if self.is_ddp else self.model
        
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], device=self.device).to(torch.int64)
            output_tokens = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
            output_ids = output_tokens[0].cpu().numpy().tolist()
            text = tokenizer.decode(output_ids)
            responses.append(f"Input: {prompt}\n\nOutput: {text}")
        sep = "=" * 100
        sep = f"\n{sep}\n"
        return sep.join(responses)
    
    def save_checkpoint(self, step: int):
        """保存检查点"""
        if not self.is_master_process:
            return
        save_path = self.paths_cfg.model_output / f"step_{step:010d}" / "model.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取原始模型
        model_to_save = self.model.module if self.is_ddp else self.model
        
        run_save_checkpoint(model_to_save, self.optimizer, step+1, save_path)
        return save_path
    
    def train(self):
        """
        主训练循环
        
        Args:
            auto_recommend: 是否自动推荐 batch_size（会覆盖配置中的值）
        """
        # DDP 相关
        self.setup_distributed()
        self.check_before_training()
        # 训练状态
        self.setup_optimizer()
        self.setup_comet_experiment()
        
        start_iter = 0
        # reload once, then broadcast
        if self.training_cfg.resume_checkpoint:
            start_iter = self.load_checkpoint()
        elif self.is_master_process:
            self.experiment.log_parameters(vars(self.args))
        
        train_dataset = np.memmap(self.paths_cfg.train_bin, dtype=np.uint16, mode="r")
        valid_dataset = np.memmap(self.paths_cfg.valid_bin, dtype=np.uint16, mode="r")
        
        # Training loop
        # Get the first batch
        batch_x, batch_y = run_get_batch(
            train_dataset,
            batch_size=self.training.train_batch_size,
            context_length=self.model_cfg.context_length,
            device=self.device,
        )
        for step in (pbar := trange(start_iter, self.training_cfg.train_steps,
                                    desc="Training", disable=not self.is_master_process)):
            # 训练步骤
            result = self.train_step(step, train_dataset, batch_x, batch_y)
            
            # 更新进度条
            if self.is_master_process:
                pbar.set_description(f"Training Step {step}, Loss: {result['loss']}, LR: {result['lr']}, PPL: {result['ppl']}")
            
                # 记录指标
                self.experiment.log_metrics(result, step=step)
            
            # 验证
            if (step + 1) % self.training_cfg.eval_interval == 0 and self.is_master_process:
                eval_result = self.validate(valid_dataset, test_infer=self.infer_cfg.test_infer)
                
                if self.is_master_process:
                    logger.info(f"Eval, Loss: {eval_result['eval_loss']:.4f}, PPL: {eval_result['eval_ppl']:.4f}")
                    response = eval_result.pop("response", None)
                    self.experiment.log_metrics(eval_result, step=step)
                    if response:
                        self.experiment.log_text(response, step=step)
            
                # 保存检查点
                if self.training_cfg.save_checkpoints:
                    save_path = self.save_checkpoint(step)
                    logger.info(f"Checkpoint saved to {save_path}")
                    
        # Calculate final estimated dev loss
        if self.is_master_process:
            eval_result = self.validate(step)
            logger.info(f"Final step Val Loss: {eval_result['eval_loss']:.4f}, Val PPL: {eval_result['eval_ppl']:.4f}")
            response = eval_result.pop("response", None)
            self.experiment.log_metrics(eval_result, step=step)
            if response:
                self.experiment.log_text(response, step=step)
            save_path = self.save_checkpoint(step)
            logger.info(f"Last checkpoint saved to {save_path}")
        
        # 9. 清理
        if self.is_ddp:
            destroy_process_group()
        
        self.experiment.end()
        
        if self.is_master_process:
            logger.info("Training completed!")
            

@hydra.main(config_path="configs/", config_name="pretrain_cs336_lm", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    tokenizer = BPETokenizer.from_files(**cfg.tokenizer)
    logger.info(f"vocab size: {tokenizer.vocab_size}")
    if cfg.model_type == "qwen2_5":
        model = Qwen2_5(**cfg.model)
    else:
        model = BasicsTransformerLM(**cfg.model)
    
    if not torch.cuda.is_available():
        cfg.training.device = "cpu"
    
    prompts = cfg.inference.prompts
    if isinstance(prompts, str) and os.path.isfile(prompts):
        import json
        prompts = json.load(open(prompts))
        cfg.inference.prompts = prompts
        
    trainer = Trainer(model, cfg, tokenizer)
    
    if cfg.training.enable:
        pprint(model)
        trainer.train()
    else:
        valid_dataset = np.memmap(cfg.paths.valid_bin, dtype=np.uint16, mode="r")
        trainer.validate(valid_dataset, test_infer=cfg.inference.test_infer)
        

async def main_chatbot(cfg: DictConfig):
    model_config, tokenizer_config = cfg.model, cfg.tokenizer
    
    if not torch.cuda.is_available():
        cfg.training.device = "cpu"
    
    if cfg.model_type == "qwen2_5":
        model = Qwen2_5(**model_config)
    else:
        model = BasicsTransformerLM(**model_config)
    tokenizer = BPETokenizer.from_files(**tokenizer_config)
    
    prompts = cfg.inference.prompts
    if isinstance(prompts, str) and os.path.isfile(prompts):
        import json
        prompts = json.load(open(prompts))
        cfg.inference.prompts = prompts
    chatbot = ChatBot(model=model, tokenizer=tokenizer, device=cfg.training.device)
    
    from scripts.printer import StreamPrinter
    printer = StreamPrinter()
    for prompt in prompts:
        print("=" * 50)
        printer.update(prompt)
        async for chunk in chatbot.stream(**cfg.inference):
            printer.update(chunk)
        printer.complete()
    

if __name__ == "__main__":
    setup_hydra_output_for_distributed()
    main()