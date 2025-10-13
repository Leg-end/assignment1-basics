import time
import pickle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cs336_basics.train_bpe_ori import train_tokenizer as train_tokenizer_ori
from cs336_basics.train_bpe_fast import train_tokenizer as train_tokenizer_fast
from cs336_basics.train_bpe_accelerate import train_tokenizer as train_tokenizer_accelerate
from tests.common import FIXTURES_PATH
from cs336_basics.Transformer import TransformerLM

import cProfile


def save_to_disk(vocab_path, merge_path, vocab, merges):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merge_path, "wb") as f:
        pickle.dump(merges, f)
        
        
def train_bpe_ori():
    input_path = FIXTURES_PATH / "corpus.en"
    _, _ = train_tokenizer_ori(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )


def train_bpe_fast():
    input_path = FIXTURES_PATH / "corpus.en"
    _, _ = train_tokenizer_fast(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
def train_bpe_accelerate():
    input_path = FIXTURES_PATH / "corpus.en"
    _, _ = train_tokenizer_accelerate(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )


def train_bpe_tinystories():
    input_path = "/data/lanyun/worksapce/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    vocab, merges = train_tokenizer_fast(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_chunks=8,
        num_process=8
    )
    end_time = time.time()
    print(f"Finish training in {end_time - start_time:.2f}s")
    save_to_disk("/data/lanyun/worksapce/assignment1-basics/assets/tokenizer/fast/tinystories_bpe_vocab.pkl",
                 "/data/lanyun/worksapce/assignment1-basics/assets/tokenizer/fast/tinystories_bpe_merges.pkl",
                 vocab, merges)
    

if __name__ == "__main__":
    # train_bpe_tinystories()
    import pstats
    cProfile.run('train_bpe_ori()', filename="/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana.prof")
    print("ori".center(50, "="))
    p = pstats.Stats("/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana.prof")
    p.sort_stats('cumtime').print_stats(10)
    # print("heap".center(50, "="))
    # p = pstats.Stats("/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana_heap.prof")
    # p.sort_stats('cumtime').print_stats(10)
    cProfile.run('train_bpe_fast()', filename="/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana_link.prof")
    print("link".center(50, "="))
    p = pstats.Stats("/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana_link.prof")
    p.sort_stats('cumtime').print_stats(10)
    cProfile.run('train_bpe_accelerate()', filename="/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana_link_heap.prof")
    print("link heap".center(50, "="))
    p = pstats.Stats("/data/lanyun/worksapce/assignment1-basics/exps/tokenize_ana_link_heap.prof")
    p.sort_stats('cumtime').print_stats(10)
    # lm = TransformerLM(vocab_size=50257,
    #                    context_length=1024,
    #                    num_layers=48,
    #                    d_model=1600,
    #                    num_heads=25,
    #                    d_ff=6400)
    # from torchsummary import summary
    # summary(lm, input_size=(1024,), batch_size=1, device="cpu")
    # print(lm.get_num_params())
    # print(f"require {lm.get_mem() / 1024 * 10024:.1f}MB memory")
    # print(f"Total FLOPS for single input is {lm.get_FLOPS()}.")
    