from tests.adapters import *
import pathlib
import numpy as np
import os
import torch
import comet_ml as comet
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig

def get_memmap_dataset(path: str, dtype: np.dtype = np.int32) -> np.ndarray:
    arr = np.memmap(path, dtype=dtype, mode="r")
    return arr

def val_batch_iter(memmap: np.ndarray, batch_size: int, context_length: int, device: str | torch.device):
    N = len(memmap)
    steps = (N-context_length-1) // batch_size
    for i in range(steps):
        start = i * batch_size
        end = start + batch_size
        x = np.stack(memmap[j: j + context_length] for j in range(start, end))
        y = np.stack(memmap[j+1: j + context_length+1] for j in range(start, end))
        yield torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.long).to(device)
        
def train(model: torch.nn.Module, device: str | torch.device, args):
    os.makedirs(args.save_path, exist_ok=True)
    # Load dataset
    train_dataset = get_memmap_dataset(args.train_data_path)
    val_dataset = get_memmap_dataset(args.val_data_path)
    # Create optimizer
    optimizer = get_adamw_cls()(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Resume from checkpoint
    start_iter = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint {args.resume_checkpoint}")
        resume_ckpt_path = pathlib.Path(HydraConfig.get().runtime.output_dir) / f"{args.save_path}/ckpt_iter{args.resume_checkpoint}.pt"
        start_iter = run_load_checkpoint(resume_ckpt_path, model, optimizer)
        print(f"Resumed at iteration {start_iter} from path {resume_ckpt_path}")
        experiment = comet.start(mode="get", experiment_key="Pretrain")
    else:
        experiment = comet.start(project_name="Pretrain")
        experiment.log_parameters(vars(args))
    
    # Training loop
    
    pbar = tqdm(range(start_iter, args.train_steps), desc="Training", leave=False)
    for step in pbar:
        model.train()
        x, y = run_get_batch(train_dataset, args.batch_size, args.context_length, device)
        
        logits, _ = model(x)
        loss = run_cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), args.grad_clip_norm)
        
        lr = run_get_lr_cosine_schedule(
            step, args.lr, args.min_lr, args.warmup_steps, args.cosine_cycle_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item(), lr=lr)
        experiment.log_metric("loss", loss.item(), step=step)
        experiment.log_metric("lr", lr, step=step)
        
        if step + 1 % args.val_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in val_batch_iter(
                    val_dataset, args.batch_size, args.context_length, device):
                    val_logits, _ = model(x_val)
                    val_loss = run_cross_entropy(val_logits.reshape(-1, val_logits.shape[-1]),
                                                 y_val.reshape(-1)).item()
                    val_losses.append(val_loss)
                    count += 1
                    if count > args.val_batches:
                        break
            val_loss_mean = np.mean(val_losses)
            experiment.log_metric("val_loss", val_loss_mean, step=step)
            print(f"Validation loss at step {step+1:05d}: {val_loss:.4f}")
        
        if step + 1 % args.save_interval == 0:
            ckpt_name = os.path.join(args.save_path, f"ckpt_iter{step+1}.pt")
            run_save_checkpoint(model, optimizer, step+1, ckpt_name)
            print(f"Checkpoint saved to {ckpt_name}")
    experiment.end()
    

def evaluate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    device: str | torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_token_id: int
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device).long()
    with torch.no_grad():
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
    if cfg.mode == "train":
        model_config, training_config = cfg.model, cfg.training
        model = TransformerLM(**model_config)
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            gpu_id = training_config.get("gpu_id", 0)
            device = torch.device(f"cuda:{gpu_id}")
        model = model.to(device)
        train(model, device, training_config)
    elif cfg.mode == "eval":
        model_config, eval_config, tokenizer_config = cfg.model, cfg.eval, cfg.tokenizer
        tokenizer = BPETokenizer.from_files(**tokenizer_config)
        print(f"vocab size: {tokenizer.vocab_size}")
        model = TransformerLM(**model_config)
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            gpu_id = training_config.get("gpu_id", 0)
            device = torch.device(f"cuda:{gpu_id}")
        model = model.to(device)
        with open(os.path.join(eval_config.save_path, f"ckpt_iter{eval_config.step}.pt"), 'rb') as f:
            checkpoint = torch.load(f, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        gen_response = evaluate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=eval_config.prompt,
            max_new_tokens=eval_config.max_new_tokens,
            temperature=eval_config.temperature,
            top_k=eval_config.top_k,
            eos_token_id=tokenizer.eos_token_id
        )
        print("Input: ", eval_config.prompt)
        print("Output: ", gen_response)
    

if __name__ == "__main__":
    main()