import os
import regex as re
import traceback
import time

from collections import defaultdict, Counter
from typing import BinaryIO, Iterable, Generator
from multiprocessing import Pool, Queue, cpu_count
from .maxheapdict import HeapDictDescending


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def find_chunk_boundaries(file: BinaryIO,
                          desired_num_chunks: int,
                          split_special_token: bytes) -> list[int]:
    """
    split file into chunks, each chunk end at split token. last chunk end at EOF
    since number of split token may less than desired chunk number, the return 
    number of chunk may <= desired number of chunk
    The overlapping situation may happen
    e.g. first chunk and second chunk cllapse at some split token at index p
    [0, p, p, ...], that means second chunk totally surround by first one.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    chunk_size = file_size // desired_num_chunks
    
    # each number indicate start of chunk, last number indicate pos of EOF
    # [0, chunk_size, ..., file_size]
    chunk_boundaries = [i*chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    scope_chunk_size = 4096  # 4kb
    
    for i in range(1, len(chunk_boundaries) - 1):
        start = chunk_boundaries[i]
        file.seek(start)
        while True:
            mini_chunk = file.read(scope_chunk_size)
            if mini_chunk == b"":  # EOF
                chunk_boundaries[i] = file_size
                break
            
            # Find split token at pos > min_chunk_size in the mini chunk
            index = mini_chunk.find(split_special_token)
            if index != -1:  # update boundary pos
                chunk_boundaries[i] = start + index
                break
            start += scope_chunk_size
    return sorted(set(chunk_boundaries))


def split_by_special_tokens(text: str,
                            special_tokens: list[str]) -> list[str]:
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    if not special_tokens_sorted:
        return [text]
    else:
        pat = "|".join(map(re.escape, special_tokens_sorted))
        return re.split(f"({pat})", text)


def pretokenize(text: str,
                special_tokens: list[str],
                drop_special_tokens: bool = True) -> Generator[bytes, None, None]:
    """
    sentence -> words
    split sentence by special tokens, then split each part by regex pattern
    """
    parts = split_by_special_tokens(text, special_tokens)
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            if not drop_special_tokens:
                yield part.encode("utf-8")
        else:
            for match in re.finditer(PAT, part):
                word = match.group()
                if word:
                    yield word.encode("utf-8")
                

def worker(text: str, special_tokens: list[str]):
    return Counter(pretokenize(text, special_tokens))
                

def train_tokenizer(input_path: str | os.PathLike,
                    vocab_size: int,
                    special_tokens: list[str],
                    num_chunks: int = 4,
                    num_process: int = 8) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start = time.time()
    
    # Step 1: Initialize Vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        token = token.encode("utf-8")
        if token not in vocab.values():
            vocab[256 + i] = token
    
    # Step 2: Chunk the text file
    chunks = []
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, "".encode("utf-8"))
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
    print(f"Time taken before pretokenizatiuon: {time.time() - start:.2f} s")
            
    # Step 3: Parallelizing Pre-tokenization and Counting
    if num_process is None:
        num_process = min(cpu_count(), 8)
    num_process = min(num_process, len(chunks))
    word_freq = Counter()
    with Pool(processes=num_process) as pool:
        print(f"Starting pre-tokenization with {num_process} processes on {len(chunks)} chunks...")
        result_iter = pool.imap_unordered(worker, (chunks, special_tokens))
        
    