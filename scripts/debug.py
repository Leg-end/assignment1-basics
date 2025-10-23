import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.adapters import *
from scripts.pretrain import get_memmap_dataset, val_batch_iter
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

def val_np_batch_iter(memmap: np.ndarray, batch_size: int):
    N = len(memmap)
    steps = N // batch_size
    remain = N % batch_size
    for i in range(steps):
        start = i * batch_size
        end = start + batch_size
        yield np.array(memmap[start: end])
    if remain > 0:
        yield np.array(memmap[end:])


def analyze_data(tokenizer, dataset):
    """分析训练数据特征"""
    total_tokens = 0
    eos_count = 0
    batch_size = 32
    
    for tokens in tqdm(val_np_batch_iter(dataset, batch_size), total=-len(dataset) // -batch_size, desc="Analysing", leave=False):
        total_tokens += len(tokens)
        eos_count += np.sum(tokens == tokenizer.eos_token_id)
    
    eos_ratio = eos_count / total_tokens
    
    print(f"EOS比例: {eos_ratio:.4f}")
    
    return eos_ratio


@hydra.main(config_path="configs/", config_name="pretrain_cs336_lm", version_base=None)
def main(cfg: DictConfig):
    model_config, training_config, tokenizer_config = cfg.model, cfg.training, cfg.tokenizer
    tokenizer = BPETokenizer.from_files(**tokenizer_config)
    train_dataset = get_memmap_dataset(training_config.valid_data_path)
    print("=== 数据检查 ===")
    analyze_data(tokenizer, train_dataset)


if __name__ == "__main__":
    main()