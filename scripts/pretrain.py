import comet_ml as comet
import pathlib
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import BPETokenizer, get_adamw_cls, run_get_lr_cosine_schedule,\
    run_load_checkpoint, run_save_checkpoint, run_get_batch, clip_grad_norm, BasicsTransformerLM
# from cs336_basics.model import BasicsTransformerLM
import torch
import math
import logging
import hydra
import hashlib
import torch.distributed as dist
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rich.pretty import pprint as pprint
from rich.traceback import install
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

install(show_locals=True)

def get_memmap_dataset(path: str, dtype: np.dtype = np.uint16) -> np.ndarray:
    arr = np.memmap(path, dtype=dtype, mode="r")
    return arr

def val_batch_iter(memmap: np.ndarray, batch_size: int, context_length: int, device: str | torch.device):
    N = len(memmap)
    steps = (N-context_length-1) // batch_size
    for i in range(steps):
        start = i * batch_size
        end = start + batch_size
        x = np.stack([memmap[j: j + context_length] for j in range(start, end)])
        y = np.stack([memmap[j+1: j + context_length+1] for j in range(start, end)])
        yield torch.tensor(x).to(device).long(), torch.tensor(y).to(device).long()
        

def get_experiment(run_id: str,
                   project_name: str,
                   workspace: str,
                   api_key:str="SJASztLoOjQpW2Sakl2PDV4YZ"):
    experiment_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
    os.environ["COMET_EXPERIMENT_KEY"] = experiment_id

    api = comet.API(api_key=api_key)  # Assumes API key is set in config/env
    api_experiment = api.get_experiment_by_id(experiment_id)

    if api_experiment is None:
        return comet.Experiment(project_name=project_name,
                                workspace=workspace)

    else:
        return comet.ExistingExperiment(project_name=project_name)
    
def load_checkpoint_dist(resume_checkpoint: int,
                         save_path: str,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         is_master_process: bool,
                         is_ddp: bool) -> int:
    try:
        resume_ckpt_path = pathlib.Path(HydraConfig.get().runtime.output_dir) / f"{save_path}/ckpt_iter{resume_checkpoint}.pt"
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {resume_checkpoint} not found at {resume_ckpt_path}")
        start_iter = run_load_checkpoint(resume_ckpt_path, model, optimizer)
        if is_master_process:
            logger.info(f"Resuming from checkpoint {resume_checkpoint}")
            logger.info(f"Resumed at iteration {start_iter} from path {resume_ckpt_path}")
        
        if is_ddp:
            # Synchronize all processes til all processes have done loading.
            dist.barrier()
            if is_master_process:
                logger.info("All processes have loaded the checkpoint")
        return start_iter
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        if is_ddp:
            destroy_process_group()
        raise
        
        
def train(model: torch.nn.Module,
          device: str | torch.device,
          args,
          tokenizer: BPETokenizer):
    # Wrap model in DDP, if we're using it.
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        torch_device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(torch_device)
        seed = args.seed + ddp_rank  # each process gets a different seed
        # Rank 0 does logging, file creation, etc.
        is_master_process = ddp_rank == 0
        if is_master_process:
            logger.info("Using DDP")
    else:
        seed = args.seed
        ddp_word_size = 1
        is_master_process = True
        
    if is_master_process:
        logger.info(
            "Total number of tokens per training step: "
            + str(
                ddp_world_size
                * args.batch_size
                * model.context_length
            )
        )
    # Seed each process differently so we can be sure that they
    # see different data batches.
    # NOTE: This assumes that you're using torch RNG, you may have
    # to seed numpy too as well if your code uses numpy random functions.
    torch.manual_seed(seed)
    
    if is_master_process:
        # TODO control hydra writing only do in GPU 0
        pass
    
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    if is_master_process:
        logger.info(f"Using {torch_dtype} precision")
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
    
    model = model.to(device)
    
    # compile the model, requires torch 2.0
    if args.compile:
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    os.makedirs(args.save_path, exist_ok=True)
    # Load dataset
    train_dataset = get_memmap_dataset(args.train_data_path)
    val_dataset = get_memmap_dataset(args.valid_data_path)
    # Create optimizer
    # Set up the AdamW optimizer.
    # First, we need to group the parameters that should
    # be decayed and those that shouldn't.
    # In particular, we do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": args.weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = get_adamw_cls()(optim_groups,
                                lr=args.lr,
                                betas=(args.beta1, args.beta2),
                                eps=args.eps,
                                weight_decay=args.weight_decay)
    # Resume from checkpoint
    start_iter = 0
    if is_master_process:
        comet.login()
    experiment = get_experiment(run_id=args.run_id,
                                project_name="Pretrain",
                                workspace="leg-end",
                                api_key="SJASztLoOjQpW2Sakl2PDV4YZ")
                                 
    if args.resume_checkpoint:
        start_iter = load_checkpoint_dist(args.resume_checkpoint,
                                           args.save_path,
                                           model,
                                           optimizer,
                                           is_master_process,
                                           is_ddp)
    elif is_master_process:
        experiment.log_parameters(vars(args))
    
    # Training loop
    
    pbar = tqdm(range(start_iter, args.train_steps), desc="Training", leave=False, disable=not is_master_process)
    x, y = run_get_batch(train_dataset, args.batch_size, args.context_length, device)
    for step in pbar:
        lr = run_get_lr_cosine_schedule(
            step, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        for micro_step in range(args.gradient_accumulation_steps):
        
            if is_ddp:
                # When using DDP, don't all-reduce gradients until the last step.
                model.require_backward_grad_sync = micro_step == args.gradient_accumulation_steps - 1
            
            with amp_ctx:
                logits, loss = model(x, y)
                loss /= args.gradient_accumulation_steps
                
                x, y = run_get_batch(train_dataset, args.batch_size, args.context_length, device)
        
            loss.backward()
        gnorm = clip_grad_norm(model.parameters(), args.clip_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        loss_float = loss.item() * args.training.gradient_accumulation_steps
        ppl_float = math.exp(loss_float)
        experiment.log_metric("gradient_norm", gnorm.item(), step=step)
        experiment.log_metric("loss", loss_float, step=step)
        experiment.log_metric("ppl", ppl_float, step=step)
        experiment.log_metric("lr", lr, step=step)
        
        if is_master_process:
            pbar.set_postfix(loss=loss.item(), lr=lr)
        
        if is_master_process and (step + 1) % args.val_interval == 0:
            model.eval()
            avg_val_loss = evaluate(
                model=model,
                val_dataset=val_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_iters=args.val_batches,
                device=device
            )
            avg_val_ppl = math.exp(avg_val_loss)
            experiment.log_metric("val_loss", avg_val_loss, step=step)
            experiment.log_metric("val_ppl", avg_val_ppl, step=step)
            logger.info(f"Step [{step+1:05d} / {args.train_steps:05d}] Val Loss: {avg_val_loss:.4f}, Val PPL: {avg_val_ppl:.4f}")
            gen_response = inference(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt="Tom and Lily were friends.",  # args.prompt
                max_new_tokens=64,
                temperature=1.0,
                top_k=50,
                eos_token_id=tokenizer.eos_token_id
            )
            msg = f"Input: Tom and Lily were friends.\nOutput: {gen_response}"
            experiment.log_text(msg, step=step)
            logger.info(msg)
            model.train()
        if is_master_process and (step + 1) % args.save_interval == 0:
            ckpt_name = os.path.join(args.save_path, f"ckpt_iter{step+1}.pt")
            run_save_checkpoint(model, optimizer, step+1, ckpt_name)
            logger.info(f"Checkpoint saved to {ckpt_name}")
    experiment.end()
    if is_ddp:
        destroy_process_group()
    
    
@torch.no_grad()
def evaluate(model: BasicsTransformerLM,
             val_dataset: np.ndarray,
             batch_size: int,
             context_length: int,
             eval_iters: int,
             device: str | torch.device):
    avg_val_loss = 0.0
    count = 0
    for x_val, y_val in val_batch_iter(
        val_dataset, batch_size, context_length, device):
        val_logits, val_loss = model(x_val, y_val)
        avg_val_loss += val_loss.item()
        count += 1
        if count > eval_iters:
            break
    avg_val_loss /= count
    return avg_val_loss
    

@torch.no_grad()
def inference(
    model: BasicsTransformerLM,
    tokenizer: BPETokenizer,
    device: str | torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_token_id: int
) -> str:
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device).to(torch.int64)
    output_tokens = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id
    )
    output_ids = output_tokens[0].cpu().numpy().tolist()
    full_ids = input_ids + output_ids
    text = tokenizer.decode(full_ids)
    return text
        

@hydra.main(config_path="configs/", config_name="pretrain_cs336_lm", version_base=None)
def main(cfg: DictConfig):
    model_config, running_config, tokenizer_config = cfg.model, cfg.training, cfg.tokenizer
    tokenizer = BPETokenizer.from_files(**tokenizer_config)
    print(f"vocab size: {tokenizer.vocab_size}")
    model = BasicsTransformerLM(**model_config)
    pprint(model)
    
    if torch.cuda.is_available():
        gpu_id = running_config.get("gpu_id", 0)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    
    if hasattr(cfg, "training"):
        train(model, device, running_config, tokenizer)
    elif hasattr(cfg, "eval"):
        with open(os.path.join(running_config.save_path, f"ckpt_iter{running_config.iteration}.pt"), 'rb') as f:
            checkpoint = torch.load(f, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        
        gen_response = evaluate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=running_config.prompt,
            max_new_tokens=running_config.max_new_tokens,
            temperature=running_config.temperature,
            top_k=running_config.top_k,
            eos_token_id=tokenizer.eos_token_id
        )
        print("Input: ", running_config.prompt)
        print("Output: ", gen_response)
    

if __name__ == "__main__":
    main()