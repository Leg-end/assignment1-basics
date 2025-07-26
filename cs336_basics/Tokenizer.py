import os
import regex as re
import traceback

from collections import defaultdict
from typing import BinaryIO, Iterable, Generator
from multiprocessing import Process, Queue
from .maxheapdict import HeapDictDescending


def get_pairs(word: tuple[str | bytes]) -> set[tuple[str, str] | tuple[bytes, bytes]]:
    """
    e.g. p,a,i,r => [(p, a), (a, i), (i, r)]
         of,t,en => [(of, t), (t, en)]
    """
    pairs = set()
    pre_char = word[0]
    for char in word[1:]:
        pairs.add((pre_char, char))
        pre_char = char
    return pairs
    

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
                drop_special_tokens: bool = True) -> Generator:
    """
    sentence -> words
    split sentence by special tokens, then split each part by regex pattern
    """
    parts = split_by_special_tokens(text, special_tokens)
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        if part in special_tokens:
            if not drop_special_tokens:
                yield part.encode("utf-8")
        else:
            for match in re.finditer(pat, part):
                word = match.group()
                yield word.encode("utf-8")


def worker(text: str, special_tokens: list[str], q: Queue):
    try:
        words = list(pretokenize(text, special_tokens))
        # counts: dict[str, int] = defaultdict(int)
        # for word in words:
        #     counts[word] += 1
        q.put(words)
    except Exception as e:
        q.put(Exception(f"Worker failed: {str(e)}\n{traceback.format_exc()}"))
        

def bpe_merge(pair_counts: dict[tuple[int, int], int],
          pair2word: dict[tuple[int, int], set[int]],
          words: list[list[int]],
          max_pair: tuple[int, int],
          new_index: int):
    """
    Args:
        pair_counts: store symbol pair and its frquency
        pair2word: store symbol pair and indices of word that contains it
        words: all words
        max_pair: symobal pair with maximum frquency
        new_index: new index for max_pair to store in vocab
    """
    pair_counts.pop(max_pair)
    delta_pair_counts = defaultdict(int)
    delat_pair2word = defaultdict(set)
    for i in pair2word[max_pair]:
        word = words[i]
        merged_word = []
        pos_list = []  # all pos of merged_pair in merged_word
        j = 0  # index w.r.t word
        k = 0  # index w.r.t merged word
        
        # Replace max_pair in word with merged_pair
        while j < len(word):
            if j < len(word) - 1 and (word[j], word[j+1]) == max_pair:
                merged_word.append(new_index)
                j += 2
                pos_list.append(k)
            else:
                merged_word.append(word[j])
                j += 1
            k += 1
        words[i] = merged_word
        # Update count and indices(pointing to its derived word) of pair around merged_pair 
        for pos in pos_list:
            # warning! exist pair_count[pair] < 0, max_pair = (48, 48) i.e. ('0', '0') or 000
            if pos > 0:
                if merged_word[pos-1] == new_index:  # e.g. a bc bc e 
                    delta_pair_counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    delta_pair_counts[(merged_word[pos-1], max_pair[0])] -= 1
                
                delta_pair_counts[(merged_word[pos-1], new_index)] += 1
                delat_pair2word[(merged_word[pos-1], new_index)].add(i)
            
            if pos < len(merged_word) - 1:
                if merged_word[pos+1] == new_index:  # e.g. a bc bc e 
                    delta_pair_counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    delta_pair_counts[(max_pair[1], merged_word[pos+1])] -= 1
                
                delta_pair_counts[(new_index, merged_word[pos+1])] += 1
                delat_pair2word[(new_index, merged_word[pos+1])].add(i)
    pair2word.pop(max_pair)
    for pair, delta in delta_pair_counts.items():
        if pair == max_pair:  # already pop max_pair, not need to update its freq
            continue
        pair_counts[pair] += delta
        # if pair_counts[pair] < 0:
        #     max_word = ''.join(bytes([p]).decode('utf-8', errors="ignore") if p < 256 else new_tok for p in max_pair)
        #     pair_word = ''.join(bytes([p]).decode('utf-8', errors="ignore") if p < 256 else new_tok for p in pair)
        #     print(max_pair, max_word)
        #     print(pair, pair_word, pair_counts[pair] - delta, pair_counts[pair])
        #     print("#"*30)
        if pair_counts[pair] == 0:
            pair_counts.pop(pair)
            pair2word.pop(pair)
    for pair, indices in delat_pair2word.items():
        pair2word[pair] = indices
    return list(delta_pair_counts.keys())
                

def train_tokenizer(input_path: str | os.PathLike,
                    vocab_size: int,
                    special_tokens: list[str],
                    num_process: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Why parallely counting frequency of words failed?
    Casue the parallel counting resulting in incorrect frequency of words.
    take word "abcbce" as example, the frequency of pair ("b", "c") will be 2 times of its word frequency,
    but its true frequency should be word count + 1.
    """
    # Step 1: Initialize Vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        token = token.encode("utf-8")
        if token not in vocab.values():
            vocab[256 + i] = token
            
    # Step 2: Chunk the text file
    chunks = []
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_process, "<|endoftext|>".encode("utf-8"))
        print(f"Parallel with {len(boundaries) - 1} chunks.")
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
            
    # Step 3: Parallelizing Pre-tokenization and Counting
    words = []
    processes = []
    q = Queue()
    for chunk in chunks:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    for _ in processes:  # main process retrieve result from queue
        result = q.get()
        if isinstance(result, Exception):
            print(f"[Error] {str(result)}")
            for p in processes:
                p.terminate()
            raise result
        words.extend(result)
        
    for p in processes:
        p.join()
    
    print(f"Finish parallely pretokeization.")
    
    # Step 4: Generate merging rules
    print("\nStart learning merge rules.")
    num_merge = max(vocab_size - len(special_tokens) - 256, 0)
    merges = []
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair2word: dict[tuple[int, int], set] = defaultdict(set)
    heap = HeapDictDescending()
    for i, word in enumerate(words):
        for idx1, idx2 in zip(word[:-1], word[1:]):  # Note: idx1 and idx2 are both int in range(0~255)
            pair_counts[(idx1, idx2)] += 1
            pair2word[(idx1, idx2)].add(i)
            
    for pair, cnt in pair_counts.items():
        heap[pair] = (cnt, (vocab[pair[0]].decode("utf-8", errors="ignore"),
                            vocab[pair[1]].decode("utf-8", errors="ignore")))
    
    for i in range(num_merge):
        # get most frequent pair, otherwise decided by length then by dict order
        # e.g. ("a", "b") prefer to ("a", "c"), thus ensure reproducibility of executation
        # max_pair = max(
        #     pair_counts.items(),
        #     key=lambda x: (
        #         x[1],  
        #         vocab[x[0][0]].decode("utf-8", errors="ignore"),
        #         vocab[x[0][1]].decode("utf-8", errors="ignore")
        #     )
        # )[0]
        max_pair, _ = heap.popitem()
        
        idx1, idx2 = max_pair
        new_idx = 256 + len(special_tokens) + i
        vocab[new_idx] = vocab[idx1] + vocab[idx2]  # merge into new token
        merges.append((vocab[idx1], vocab[idx2]))
        # recompute counts of pair of merged token
        update_pairs = bpe_merge(pair_counts, pair2word, words, max_pair, new_idx)
        for pair in update_pairs:
            if pair in heap:
                _, (p1, p2) = heap[pair]
                heap[pair] = (pair_counts[pair], (p1, p2))
            else:
                heap[pair] = (pair_counts[pair], (vocab[pair[0]].decode("utf-8", errors="ignore"),
                                                  vocab[pair[1]].decode("utf-8", errors="ignore")))
    return vocab, merges


class BPETokenizer:

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        special_tokens = special_tokens or []
        # Ensure special tokens are in the vocabulary
        self.cache: dict[str, tuple[bytes]] = {}
        self.decoder = vocab
        self.encoder: dict[bytes, int] = {v: k for k, v in vocab.items()}
        n = len(vocab)
        for i, token in enumerate(special_tokens):
            byte_token = token.encode('utf-8')
            self.cache[token] = [byte_token]  # must be list or tuple
            if byte_token not in self.encoder:
                token_id = n + i
                self.decoder[token_id] = byte_token
                self.encoder[byte_token] = token_id
        self.merges = merges
        self.bpe_rank = dict(zip(merges, range(len(merges))))
        self.special_tokens = special_tokens
        
    
    def bpe(self, word: bytes) -> list[bytes]:
        """
        Apply bpe merge
        turn word into subwords(split by space)
        unicode (int value using 16-base) -> utf-8: split into bytes, each byte has int within 0~255
        """
        if not word:  # empty word
            return [word]  # must be list or tuple
        if word in self.cache:
            return self.cache[word]
        # treat token as tuple of symbols
        tokens = tuple(bytes([w]) for w in word)
        
        pairs = get_pairs(tokens)
        
        if not pairs:
            return [word]  # must be list or tuple
        
        while True:
            # get most frequent merge combination
            bigram = min(pairs, key=lambda pair: self.bpe_rank.get(pair, float("inf")))
            if bigram not in self.bpe_rank:  # if not merge combination exist
                break
            # merge two symbols into one symbols
            # e.g. bigram = ab, abcabddaba -> ab c ab dd ab a
            first, second = bigram
            new_tokens = []  # store merged word
            i = 0
            while i < len(tokens):  # find all bigram in word
                try:
                    j = tokens.index(first, i)  # locate bigram start from i
                    new_tokens.extend(tokens[i: j]) # store symbols before merge pair
                    i = j  # next start position
                except: # already reached last bigram
                    new_tokens.extend(tokens[i:]) # store rest symbols
                    break
                # make sure located pair = bigram, then can we do merge
                if tokens[i] == first and i < len(tokens) - 1 and tokens[i+1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:  # mismatch to bigram
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens = tuple(new_tokens)
            tokens = new_tokens
            if len(tokens) == 1: # all symbols merge into single one
                break
            else:  # continue merging
                pairs = get_pairs(tokens)
        self.cache[word] = tokens
        return tokens
    
    def encode(self, text: str) -> list[int]:
        # text(str) -> words(bytes) -> subwords(bytes) -> ids(int)
        bpe_tokens = []
        for word in pretokenize(text, self.special_tokens, drop_special_tokens=False):
            for token in self.bpe(word):
                bpe_tokens.append(self.encoder[token])
        return bpe_tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield self.encode(chunk)
    
    def decode(self, token_ids: list[int]) -> str:
        # ids(int) -> subwords(bytes) -> words(bytes) -> text(str)
        text = b''.join([self.decoder[token_id] for token_id in token_ids])
        text = text.decode('utf-8', errors="replace")
        return text
    
    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None =None):
        import json
        vocab = json.load(open(vocab_filepath))
        merges = open(merges_filepath).read().splitlines()
        return cls(vocab, merges, special_tokens)