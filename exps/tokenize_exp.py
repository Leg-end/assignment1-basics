import time
import pickle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cs336_basics.train_bpe import train_bpe as train_bpe_bl
from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH
from cs336_basics.Transformer import TransformerLM

import cProfile


def save_to_disk(vocab_path, merge_path, vocab, merges):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merge_path, "wb") as f:
        pickle.dump(merges, f)
        
        
def train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    # train_bpe_bl(
    #     input_path=input_path,
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )


def train_bpe_tinystories():
    input_path = "/data/lanyun/worksapce/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_process=os.cpu_count()
    )
    end_time = time.time()
    print(f"Finish training in {end_time - start_time}s")
    save_to_disk("/data/lanyun/worksapce/assignment1-basics/assets/tokenizer/tinystories_bpe_vocab.pkl",
                 "/data/lanyun/worksapce/assignment1-basics/assets/tokenizer/tinystories_bpe_merges.pkl",
                 vocab, merges)
    

if __name__ == "__main__":
    # train_bpe()
    # cProfile.run('train_bpe()', filename="/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana_heap.prof")
    # import pstats
    # p = pstats.Stats("/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana.prof")
    # p.sort_stats('cumtime').print_stats(10)
    # print("="*50)
    # p = pstats.Stats("/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana_heap.prof")
    # p.sort_stats('cumtime').print_stats(10)
    lm = TransformerLM(vocab_size=50257,
                       context_length=1024,
                       num_layers=48,
                       d_model=1600,
                       num_heads=25,
                       d_ff=6400)
    from torchsummary import summary
    summary(lm, input_size=(1024,), batch_size=1, device="cpu")
    # print(lm.get_num_params())
    # print(f"require {lm.get_mem() / 1024 * 10024:.1f}MB memory")
    # print(f"Total FLOPS for single input is {lm.get_FLOPS()}.")
    