import os
import time
import logging
import heapq
from collections import defaultdict, Counter
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from cs336_basics.maxheapdict import heapdict
from cs336_basics.Tokenizer import find_chunk_boundaries
from .train_bpe_fast import worker, PQItem

# adapt from https://github.com/Spectual/stanford-cs336-a1/blob/main/cs336_basics/BPETokenizer.py

def bpe_merge_fast(pair_freq: dict[tuple[int, int], int],
          pair2wids: dict[tuple[int, int], set[int]],
          wid_freq: dict[int, int],
          words: list[list[int]],
          max_pair: tuple[int, int],
          new_index: int):
    """Merge the pairs with highest frequency and update pair_freq, index_dict
    BUG
    max_pair = ('r', 'e')
     ,T,e,r,r,e,r,o,s ->  ,T,e,r,re,r,o,s
    update_pairs = [('e', 'r'), ('r', 'r'), ('r', 'e'), ('r', 're'), ('re', 'r')]
    ('e', 'r') is old_right_pair, will be removed, i.e.
    pair_freq[('e', 'r')] -= cnt
    pair2wids[('e', 'r')].discard(i)
    but there's still 'e, r' (away from max_pair) in word, that means we can not remove i from pair2wids[('e', 'r')]
    So the BUG stays in operation of pair2wids[pair].discard(i)
    Dicard can only happen when pair is unique in word
    Solution: store {word index: cnt of pair in word} as pair2wids's value
    discard pair only when cnt == 1
    """
    indices = list(pair2wids[max_pair].keys())
    update_pairs = set()
    for i in indices:
        cnt = wid_freq[i]
        word = words[i]
        merged_word = []

        merge_pos_list = []   # Store positions of max_pair for each new pretoken after merge
        merge_pos = 0
        j = 0
        while j < len(word):
            # 检查当前和下一个token是否是要合并的pair
            if j < len(word) - 1 and (word[j], word[j+1]) == max_pair:
                # 合并这对token
                merged_word.append(new_index)
                merge_pos_list.append(merge_pos)
                j += 2
            else:
                # 保持原token
                merged_word.append(word[j])
                j += 1
            merge_pos += 1
        words[i] = merged_word
        # a,b,c,b,c,b,c,d
        # a,b c,b c,b c,d
        # _,p  ,p  ,p  ,_
        # Update pair_freq and index_dict
        for merge_pos in merge_pos_list:
            if merge_pos > 0 and merged_word[merge_pos-1] != new_index:
                old_left_pair = (merged_word[merge_pos-1], max_pair[0])
                pair_freq[old_left_pair] -= cnt
                if pair2wids[old_left_pair][i] == 1:
                    del pair2wids[old_left_pair][i]
                else:
                    pair2wids[old_left_pair][i] -= 1
                update_pairs.add(old_left_pair)
                
                new_left_pair = (merged_word[merge_pos-1], new_index)
                pair_freq[new_left_pair] += cnt
                if i not in pair2wids[new_left_pair]:
                    pair2wids[new_left_pair][i] = 1
                else:
                    pair2wids[new_left_pair][i] += 1
                update_pairs.add(new_left_pair)

            if merge_pos < len(merged_word) - 1:
                if merged_word[merge_pos+1] != new_index:
                    old_right_pair = (max_pair[1], merged_word[merge_pos+1])
                    new_right_pair = (new_index, merged_word[merge_pos+1])
                else:
                    old_right_pair = (max_pair[1], max_pair[0])
                    new_right_pair = (new_index, new_index)
                pair_freq[old_right_pair] -= cnt
                if pair2wids[old_right_pair][i] == 1:
                    del pair2wids[old_right_pair][i]
                else:
                    pair2wids[old_right_pair][i] -= 1
                update_pairs.add(old_right_pair)
                
                pair_freq[new_right_pair] += cnt
                if i not in pair2wids[new_right_pair]:
                    pair2wids[new_right_pair][i] = 1
                else:
                    pair2wids[new_right_pair][i] += 1
                update_pairs.add(new_right_pair)
    del pair_freq[max_pair]
    del pair2wids[max_pair]
    return update_pairs


def bpe_merge(pair_freq: dict[tuple[int, int], int],
          pair2wids: dict[tuple[int, int], dict[int, int]],
          wid_freq: dict[int, int],
          words: list[list[int]],
          max_pair: tuple[int, int],
          new_index: int):
    """Merge the pairs with highest frequency and update pair_freq, index_dict"""
    indices = list(pair2wids[max_pair].keys())
    update_pairs = set()
    for i in indices:
        cnt = wid_freq[i]
        word = words[i]
        merged_word = []
        
        j = 0
        while j < len(word):
            # 检查当前和下一个token是否是要合并的pair
            if j < len(word) - 1 and (word[j], word[j+1]) == max_pair:
                # 合并这对token
                merged_word.append(new_index)
                j += 2
            else:
                # 保持原token
                merged_word.append(word[j])
                j += 1
        
        # 更新words
        words[i] = merged_word
        
        # 删除旧的pair频率（所有包含被影响token的pair）
        # 重新计算这个word的所有pair频率
        delta_pair_freq = defaultdict(int)
        for k in range(len(word) - 1):
            old_pair = (word[k], word[k+1])
            pair_freq[old_pair] -= cnt
            delta_pair_freq[old_pair] -= cnt
            del pair2wids[old_pair][i]
        
        # 添加新的pair频率
        for k in range(len(merged_word) - 1):
            new_pair = (merged_word[k], merged_word[k+1])
            pair_freq[new_pair] += cnt
            delta_pair_freq[new_pair] += cnt
            pair2wids[new_pair][i] = 1
        for pair, freq in delta_pair_freq.items():
            if freq == 0 and pair == max_pair:
                continue
            update_pairs.add(pair)
    
    # 清理空的pair
    del pair_freq[max_pair]
    del pair2wids[max_pair]
    # TODO 定期清理频率为0的pair
    return update_pairs

def train_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
    num_chunks: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
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
    if num_processes is None:
        num_processes = min(cpu_count(), 8)
    num_processes = min(num_processes, len(chunk_args))
    word_freq = Counter()
    with Pool(processes=num_processes) as pool:
        print(f"Starting pre-tokenization with {num_processes} processes on {len(chunk_args)} chunks...")
        result_iter = pool.imap_unordered(worker, chunk_args)
        for counter in tqdm(result_iter, total=len(chunk_args), desc="Pre-tokenization", leave=True):
            word_freq.update(counter)
    middle = time.time()
    print(f"Pre-tokenization and word counting done in {middle - begin:.2f} s")
    begin = middle

    # Step 4: Generate merging rules
    pair_freq: dict[tuple[int, int], int] = defaultdict(int)
    pair2wids: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    wid_freq: dict[int, int] = defaultdict(int)
    words = []
    k = 0
    for word, cnt in tqdm(word_freq.items(), desc="Generating merging rules", leave=True):
        if len(word) < 2:
            continue  # skip single-letter word
        wid_freq[k] = cnt
        words.append(word)
        for idx1, idx2 in zip(word[:-1], word[1:]):
            pair_freq[(idx1, idx2)] += cnt
            if k not in pair2wids[(idx1, idx2)]:
                pair2wids[(idx1, idx2)][k] = 1
            else:
                pair2wids[(idx1, idx2)][k] += 1
        k += 1
            
    # heap = heapdict()
    # for pair, freq in pair_freq.items():
    #     heap[pair] = (freq, (vocab[pair[0]], vocab[pair[1]]))
    pq = [PQItem(freq, pair, (vocab[pair[0]], vocab[pair[1]])) 
          for pair, freq in pair_freq.items()]
    heapq.heapify(pq)

    num_merges = max(vocab_size - len(special_tokens) - 256, 0)
    pbar = tqdm(total=num_merges, desc="Merging", leave=True)
    merges = []
    for i in range(num_merges):
        # if not heap:
        #     break
        
        # max_pair, _ = heap.popitem()  # no zombie elements
        if not pq:
            break
        
        max_pair = None
        while pq:
            item = heapq.heappop(pq)
            if item.id_pair not in pair_freq:
                continue
            if pair_freq[item.id_pair] == item.freq:
                max_pair = item.id_pair
                break
        if max_pair is None:
            break
        
        new_idx = 256 + len(special_tokens) + i
        idx1, idx2 = max_pair
        vocab[new_idx] = vocab[idx1] + vocab[idx2]  # merge into new token
        merges.append((vocab[idx1], vocab[idx2]))
        update_pairs = bpe_merge_fast(pair_freq, pair2wids, wid_freq, words, max_pair, new_idx)
        # for pair in update_pairs:
        #     heap[pair] = (pair_freq[pair], (vocab[pair[0]], vocab[pair[1]]))
        for pair in update_pairs:
            heapq.heappush(pq, PQItem(pair_freq[pair], pair, (vocab[pair[0]], vocab[pair[1]])))
        pbar.update(1)
    pbar.close()
    end = time.time()
    print(f"Merging done in {end - begin:.2f} s")
    return vocab, merges
