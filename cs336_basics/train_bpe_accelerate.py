import os
import time

from collections import defaultdict, Counter
from tqdm import tqdm
from multiprocessing import Pool
from .maxheapdict import heapdict
from .Tokenizer import find_chunk_boundaries
from .train_bpe_fast import worker, bpe_merge, LinkNode
                

def train_tokenizer(input_path: str | os.PathLike,
                    vocab_size: int,
                    special_tokens: list[str],
                    num_chunks: int = 4,
                    num_process: int = 8) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    begin = time.time()
    
    # Step 1: Initialize Vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        token = token.encode("utf-8")
        if token not in vocab.values():
            vocab[256 + i] = token
    
    # Step 2: Chunk the text file
    chunk_args = []
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunk_args.append((f.read(end - start).decode("utf-8", errors="ignore"), special_tokens))
    middle = time.time()
    print(f"Time taken before pretokenizatiuon: {middle - begin:.2f} s")
    begin = middle
            
    # Step 3: Parallelizing Pre-tokenization and Counting
    if num_process is None:
        num_process = min(cpu_count(), 8)
    num_process = min(num_process, len(chunk_args))
    word_freq = Counter()
    with Pool(processes=num_process) as pool:
        print(f"Starting pre-tokenization with {num_process} processes on {len(chunk_args)} chunks...")
        result_iter = pool.imap_unordered(worker, chunk_args)
        for counter in tqdm(result_iter, total=len(chunk_args), desc="Pre-tokenization", leave=True):
            word_freq.update(counter)
    middle = time.time()
    print(f"Pre-tokenization and word counting done in {middle - begin:.2f} s")
    begin = middle
    
    # Step 4: Generate merging rules
    pair_freq: dict[tuple[int, int], int] = defaultdict(int)
    pair2nodes: dict[tuple[int, int], set[LinkNode]] = defaultdict(set)
    for word, cnt in tqdm(word_freq.items(), desc="Generating merging rules", leave=True):
        if len(word) < 2:
            continue  # skip single-letter word
        freq = {'cnt': cnt}  # all link nodes share the same freq, saving memory
        head = LinkNode(word[0], freq)
        prev_node = head
        for i in range(1, len(word)):
            curr_node = LinkNode(word[i], freq)
            prev_node.next = curr_node
            curr_node.prev = prev_node
            pair = (prev_node.value, curr_node.value)
            pair2nodes[pair].add(prev_node)
            prev_node = curr_node
            pair_freq[pair] += cnt
    
    heap = heapdict()
    for pair, freq in pair_freq.items():
        heap[pair] = (freq, (vocab[pair[0]], vocab[pair[1]]))
    
    num_merge = max(vocab_size - len(special_tokens) - 256, 0)
    pbar = tqdm(total=num_merge, desc="Merging", leave=True)
    merges = []
    for i in range(num_merge):
        if not heap.heap:
            break
        
        max_pair = None
        while heap.heap:
            pair, value = heap.popitem()
            if pair not in pair_freq:
                continue
            if pair_freq[pair] == value[0]:
                max_pair = pair
                break
        if max_pair is None:
            break
        
        new_idx = 256 + len(special_tokens) + i
        idx1, idx2 = max_pair
        vocab[new_idx] = vocab[idx1] + vocab[idx2]  # merge into new token
        merges.append((vocab[idx1], vocab[idx2]))
        
        update_pairs = bpe_merge(pair_freq=pair_freq, pair2nodes=pair2nodes,
                                 max_pair=max_pair, new_index=new_idx)
        for pair in update_pairs:
            heap[pair] = (pair_freq[pair], (vocab[pair[0]], vocab[pair[1]]))
        pbar.update(1)
    pbar.close()
    end = time.time()
    print(f"Merging done in {end - begin:.2f} s")
    return vocab, merges