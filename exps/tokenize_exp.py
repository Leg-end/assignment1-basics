import time
import pickle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging

from scripts import get_bpe_trainer
from tests.common import FIXTURES_PATH
# from cs336_basics.Transformer import TransformerLM
from functools import wraps
from contextlib import contextmanager

import cProfile
import pathlib
import pstats

SAVE_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "assets"

def config_logging():
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)
    # FileHandler
    fh = logging.FileHandler('./debug_fast.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger 


def save_to_disk(vocab_path, merge_path, vocab, merges):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merge_path, "wb") as f:
        pickle.dump(merges, f)
        

def auto_path(param_names: list[str], func, *args, **kwargs):
    import inspect
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    path_parts = []
    for param_name in param_names:
        if param_name in bound_args.arguments:
            value = bound_args.arguments[param_name]
            # 处理不同类型的参数值
            if isinstance(value, (int, float, str, bool)):
                value_str = str(value)
                if '/' in value_str:
                    value_str = os.path.splitext(os.path.basename(value_str))[0]
                elif '.' in value_str:
                    value_str = os.path.splitext(value_str)[0]
            elif hasattr(value, '__name__'):  # 函数/类名
                value_str = value.__name__
            else:
                value_str = type(value).__name__
            path_parts.append(f"{param_name}_{value_str}")
    path = '-'.join(path_parts)
    return path
        
        
def profile_decorator(*param_names, save: bool = True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            try:
                result = prof.runcall(func, *args, **kwargs)
                return result
            finally:
                if save:
                    profile_dir = SAVE_PATH / "profile"
                    os.makedirs(profile_dir, exist_ok=True)
                    name = auto_path(param_names, func, *args, **kwargs)
                    profile_path = profile_dir / f"{name}.prof"
                    prof.dump_stats(profile_path)
                stas = pstats.Stats(prof)
                stas.sort_stats('cumtime')
                stas.print_stats(10)
        return wrapper
    return decorator

@contextmanager
def profile_context(name: str, save: bool = True):
    prof = cProfile.Profile()
    prof.enable()
    try:
        yield
    finally:
        prof.disable()
        if save:
            profile_dir = SAVE_PATH / "profile"
            os.makedirs(profile_dir, exist_ok=True)
            profile_path = profile_dir / f"{name}.prof"
            prof.dump_stats(str(profile_path))
        stas = pstats.Stats(prof)
        stas.sort_stats('cumtime')
        stas.print_stats(10)
        
@profile_decorator("alg", "corpus")
def train_bpe(alg: str,
              corpus: str,
              vocab_size: int,
              special_tokens: list[str] = ["<|endoftext|>"],
              save: bool = False,
              **kwargs):
    input_path = FIXTURES_PATH / corpus
    trainer = get_bpe_trainer(alg, **kwargs)
    vocab, merges = trainer.train(input_path, vocab_size, special_tokens)
    if save:
        corpus = os.path.splitext(os.path.basename(corpus))[0]
        save_dir = SAVE_PATH / "tokenizer"
        save_to_disk(save_dir/f"{alg}_{corpus}_bpe_vocab.pkl",
                     save_dir/f"{alg}_{corpus}_bpe_merges.pkl",
                     vocab, merges)
        

def main():
    corpus_params = {
        # "corpus.en": {
        #     "vocab_size": 500,
        #     "num_chunk": 4,
        #     "num_counter": 8
        # },
        "TinyStories/TinyStoriesV2-GPT4-train.txt": {
            "vocab_size": 10000,
            "num_chunk": 32,
            "num_counter": 8
        },
        # "owt/owt_train.txt": {
        #     "vocab_size": 32000,
        #     "num_chunk": 32,
        #     "num_counter": 8
        # }
    }
    for alg_name in ["ori", "fast", "accelerate"]:
        for corpus_path in corpus_params:
            train_bpe(alg=alg_name,
                      corpus=corpus_path,
                      **corpus_params[corpus_path])
    

def debug():
    # max_pairs = [('b', 'c'), ('bc', 'd'), ('bcd', 'e')]
    # new_indices = ['bc', 'bcd', 'bcde']
    # word = ['a', 'b', 'c', 'd', 'e', 'c', 'b', 'c' 'd', 'e', 'b', 'c', 'd', 'b', 'c', 'd', 'e']
    max_pairs = [('e', 'r')]
    new_indices = ['er']
    word = " ,T,e,r,re,r,o,s".split(',')
    update_pairs = set()
    for k in range(len(max_pairs)):
        print(f"Round {k}".center(100, '+'))
        max_pair = max_pairs[k]
        new_index = new_indices[k]
        merged_word = []
        pos_list = []   # Store positions of max_pair for each new pretoken after merge
        pos = 0
        j = 0

        # Replace max_pair with new_index in each pretoken
        while j < len(word):
            if (j < len(word)-1) and ((word[j], word[j+1]) == max_pair):
                merged_word.append(new_index)
                pos_list.append(pos)
                j += 2
            else:
                merged_word.append(word[j])
                j += 1
            pos += 1
        print(pos_list)
        print(merged_word)
        word = merged_word
        for pos in pos_list:
            print(f"pos = {pos}".center(50, "="))
            if pos > 0 and merged_word[pos-1] != new_index:
                old_left_pair = (merged_word[pos-1], max_pair[0])
                print(f"remove old_left_pair {old_left_pair}")
                update_pairs.add(old_left_pair)
                
                new_left_pair = (merged_word[pos-1], new_index)
                print(f"add new_left_pair {new_left_pair}")
                update_pairs.add(new_left_pair)

            if pos < len(merged_word) - 1:
                if merged_word[pos+1] != new_index:
                    old_right_pair = (max_pair[1], merged_word[pos+1])
                    new_right_pair = (new_index, merged_word[pos+1])
                else:
                    old_right_pair = (max_pair[1], max_pair[0])
                    new_right_pair = (new_index, new_index)
                print(f"remove old_right_pair {old_right_pair}")
                update_pairs.add(old_right_pair)
                print(f"add new_right_pair {new_right_pair}")
                update_pairs.add(new_right_pair)
    print(f"update_pairs = {update_pairs}")
    

if __name__ == "__main__":
    main()
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
    