import comet_ml as comet
import pathlib
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
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rich.pretty import pprint as pprint
from rich.traceback import install
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.inference import ChatBot

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
        

def get_experiment(run_id: str,
                   project_name: str,
                   workspace: str,
                   resume: bool,
                   is_master_process: bool,
                   api_key:str="SJASztLoOjQpW2Sakl2PDV4YZ",
                   experiment_config: dict | None = None):
    if is_master_process and not resume:  # always create a new experiment if not resuming
        experiment_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
        api = comet.API(api_key=api_key)  # Assumes API key is set in config/env
        api_experiment = api.get_experiment_by_key(experiment_id)
        if api_experiment is not None:
            logger.warning(f"Experiment {run_id} already exists, for not resume mode, a random experiment_key will be used")
            experiment_id = comet.get_experiment_key(None)
        os.environ["COMET_EXPERIMENT_KEY"] = experiment_id
        exp_cfg = comet.ExperimentConfig(**experiment_config) if experiment_config is not None else None
        return comet.start(api_key=api_key,
                           workspace=workspace,
                           project_name=project_name,
                           experiment_key=experiment_id,
                           experiment_config=exp_cfg)
    experiment_id = os.environ["COMET_EXPERIMENT_KEY"]
    api = comet.API(api_key=api_key)  # Assumes API key is set in config/env
    api_experiment = api.get_experiment_by_key(experiment_id)

    if api_experiment is None:
        return comet.Experiment(project_name=project_name,
                                workspace=workspace,
                                api_key=api_key)
    else:
        return comet.ExistingExperiment(project_name=project_name,
                                        api_key=api_key)
    
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
        device = "cuda"  # model will be moved to CUDA device in current GPU
        torch.cuda.set_device(torch_device)
        seed = args.seed + ddp_rank  # each process gets a different seed
        # Rank 0 does logging, file creation, etc.
        is_master_process = ddp_rank == 0
        if is_master_process:
            logger.info("Using DDP")
    else:
        seed = args.seed
        ddp_world_size = 1
        is_master_process = True
        
    if is_master_process:
        logger.info(
            "Total number of tokens per training step: "
            + str(
                args.gradient_accumulation_steps
                * ddp_world_size
                * args.batch_size
                * args.context_length
            )
        )
    # Seed each process differently so we can be sure that they
    # see different data batches.
    # NOTE: This assumes that you're using torch RNG, you may have
    # to seed numpy too as well if your code uses numpy random functions.
    torch.manual_seed(seed)
    
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
    # if is_master_process:
    #     comet.login()
    experiment = get_experiment(run_id=args.run_id,
                                project_name="Pretrain",
                                workspace="leg-end",
                                resume=args.resume_checkpoint,
                                is_master_process=is_master_process,
                                api_key="SJASztLoOjQpW2Sakl2PDV4YZ",
                                experiment_config=args.get("experiment_config", None))
                                 
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
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.gradient_accumulation_steps):
        
            if is_ddp:
                # When using DDP, don't all-reduce gradients until the last step.
                model.require_backward_grad_sync = micro_step == args.gradient_accumulation_steps - 1
            
            with amp_ctx:
                x, y = run_get_batch(train_dataset, args.batch_size, args.context_length, device)
                logits, loss = model(x, y)
                loss = loss / args.gradient_accumulation_steps
        
            loss.backward()
        gnorm = clip_grad_norm(model.parameters(), args.clip_grad_norm)
        lr = run_get_lr_cosine_schedule(
            step, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        loss_float = loss.item() * args.gradient_accumulation_steps
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
                prompts=args.prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
            experiment.log_text(gen_response, step=step)
            model.train()
        if is_master_process and (step + 1) % args.save_interval == 0:
            ckpt_name = os.path.join(args.save_path, f"ckpt_iter{step+1}.pt")
            run_save_checkpoint(model, optimizer, step+1, ckpt_name)
            logger.info(f"Checkpoint saved to {ckpt_name}")
    if is_ddp:
        destroy_process_group()
    experiment.end()
    
    
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
    model: BasicsTransformerLM | DDP,
    tokenizer: BPETokenizer,
    device: str | torch.device,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
) -> str:
    if isinstance(model, DDP):
        model = model.module
    if isinstance(prompts, str):
        prompts = [prompts]
    responses = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device).to(torch.int64)
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
        responses.append(text)
        logger.info(f"[maximum new tokens: {max_new_tokens}] Generated {len(output_ids) - len(input_ids)} tokens")
        logger.info(f"Input: {prompt}\nOutput: {text}")
        logger.info("=" * 100)
    sep = "=" * 100
    sep = f"\n{sep}\n"
    return sep.join(responses)
        

@hydra.main(config_path="configs/", config_name="evaluate_cs336_lm", version_base=None)
def main(cfg: DictConfig):
    training = hasattr(cfg, "training")
    model_config, tokenizer_config = cfg.model, cfg.tokenizer
    running_config = cfg.training if training else cfg.eval
    tokenizer = BPETokenizer.from_files(**tokenizer_config)
    logger.info(f"vocab size: {tokenizer.vocab_size}")
    if cfg.model_type == "qwen2_5":
        model = Qwen2_5(**model_config)
    else:
        model = BasicsTransformerLM(**model_config)
    
    if torch.cuda.is_available():
        gpu_id = running_config.get("gpu_id", 0)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    
    if training:
        pprint(model)
        train(model, device, running_config, tokenizer)
    else:
        torch.manual_seed(running_config.seed)
        model = model.to(device)
        model.eval()
        ckpt_path = os.path.join(running_config.save_path, f"ckpt_iter{running_config.iteration}.pt")
        iteration = run_load_checkpoint(ckpt_path, model)
        logger.info(f"Loading from checkpoint {iteration} from path {ckpt_path}")
        prompts = running_config.prompts
        if isinstance(prompts, str) and os.path.isfile(prompts):
            import json
            prompts = json.load(open(prompts))
        # for test_name, prompt in prompts.items():
        #     print(test_name.center(50, "="))
        #     for (pname, p) in prompt:
        #         gen_response = inference(
        #             model=model,
        #             tokenizer=tokenizer,
        #             device=device,
        #             prompt=p,
        #             max_new_tokens=running_config.max_new_tokens,
        #             temperature=running_config.temperature,
        #             top_k=running_config.top_k,
        #             top_p=running_config.top_p,
        #             repetition_penalty=running_config.repetition_penalty
        #         )
        #         logger.info(f"[#{pname}] Input: {p}\nOutput: {gen_response}")
        chatbot = ChatBot(model=model, tokenizer=tokenizer, device=device)
        # import shutil
        # terminal_width = shutil.get_terminal_size().columns
        from scripts.printer import StreamPrinter
        printer = StreamPrinter()
        for prompt in prompts:
            print("=" * 50)
            # text = prompt
            printer.update(prompt)
            for chunk in chatbot.stream(prompt,
                                    max_new_tokens=running_config.max_new_tokens,
                                    temperature=running_config.temperature,
                                    top_k=running_config.top_k,
                                    top_p=running_config.top_p,
                                    repetition_penalty=running_config.repetition_penalty):
                printer.update(chunk)
            #     text += chunk
            #     if len(text) > terminal_width - 5:
            #         print(f"\r{text[:terminal_width-5]}\n", end="", flush=True)
            #         text = text[terminal_width-5:]
            #     print(f"\r{text}", end="", flush=True)
            # print()
            printer.complete()
    

if __name__ == "__main__":
    setup_hydra_output_for_distributed()
    main()