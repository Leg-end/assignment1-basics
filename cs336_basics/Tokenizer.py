import os
import regex as re
import ftfy
import html
import gzip
import traceback

from functools import lru_cache
from collections import defaultdict
from typing import BinaryIO
from multiprocessing import Process, Queue


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: tuple[str]) -> set[tuple[str, str]]:
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


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    # replace continue whitespaces into single whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()  # remove leading and tailing whitespace
    return text


class Tokenizer:

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        special_tokens = special_tokens or []
        # Ensure special tokens are in the vocabulary
        self.cache = {}
        self.decoder = vocab
        self.encoder = {v: k for k, v in vocab.items()}
        n = len(vocab)
        for i, token in enumerate(special_tokens):
            token = token.encode('utf-8')
            self.cache[token] = token
            if token not in self.encoder:
                token_id = n + i
                self.decoder[token_id] = token
                self.encoder[token] = token_id
        self.merges = merges
        self.bpe_rank = dict(zip(merges, range(len(merges))))
        # Sort special tokens by length (longest first) to avoid partial matches
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        pat = "|".join(map(re.escape, sorted_special_tokens))
        pat += r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
        self.pat = re.compile(pat, re.IGNORECASE)
        
    
    def bpe(self, word: bytes) -> bytes:
        """
        Apply bpe merge
        turn word into subwords(split by space)
        """
        if word in self.cache:
            return self.cache[word]
        # treat token as tuple of symbols
        word = tuple(bytes([w]) for w in word)
        
        pairs = get_pairs(word)
        
        if not pairs:
            return word
        
        while True:
            # get most frequent merge combination
            bigram = min(pairs, key=lambda pair: self.bpe_rank.get(pair, float("inf")))
            if bigram not in self.bpe_rank:  # if not merge combination exist
                break
            # merge two symbols into one symbols
            # e.g. bigram = ab, abcabddaba -> ab c ab dd ab a
            first, second = bigram
            new_word = []  # store merged word
            i = 0
            while i < len(word):  # find all bigram in word
                try:
                    j = word.index(first, i)  # locate bigram start from i
                    new_word.extend(word[i: j]) # store symbols before merge pair
                    i = j  # next start position
                except: # already reached last bigram
                    new_word.extend(word[i:]) # store rest symbols
                    break
                # make sure located pair = bigram, then can we do merge
                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:  # mismatch to bigram
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1: # all symbols merge into single one
                break
            else:  # continue merging
                pairs = get_pairs(word)
        word = b' '.join(word)
        self.cache[word] = word
        return word
    
    def encode(self, text: str) -> list[int]:
        # text(str) -> words(bytes) -> subwords(bytes) -> ids(int)
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for word in re.findall(self.pat, text):
            for token in self.bpe(word.encode('utf-8')).split(b' '):
                bpe_tokens.append(self.encoder[token])
        return bpe_tokens
    
    def decode(self, token_ids: list[int]) -> str:
        # ids(int) -> subwords(bytes) -> words(bytes) -> text(str)
        text = b''.join([self.decoder[token_id] for token_id in token_ids])
        text = text.decode('utf-8', errors="replace")
        return text
    

class SimpleTokenizer(object):
    """
    Byte-BPE
        
        byte_encoder: int -> str(unicode)
        byte_decoder: str(unicode) -> int
        
        encode: str -> int
            |!|convert e.g. emoji, Chinese, ... into unicode character sequence, thus it can represent any text without <unk>|!|
        e.g. '😊' -> b'\xf0\x9f\x98\x8a' -> [240, 159, 152, 138] -> ['ð', 'Ł', 'ĺ', 'Ĭ'] -> 'ðŁĺĬ
            |!|str -> bytes(utf-8) -> [int, ..., int] ->byte_encoder-> [str, ..., str] -> str|!| ->bpe-> subwords(str) ->encoder-> int
        decode: int -> str
            int ->decoder-> str -> [str, ..., str] ->byte_decoder-> [int, ..., int] -> bytearray ->utf-8 decode-> str
    """
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
    

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
    min_chunk_size = 1024  # each chunk's size should >= 1024
    
    for i in range(1, len(chunk_boundaries) - 1):
        start = chunk_boundaries[i]
        file.seek(start)
        while True:
            mini_chunk = file.read(scope_chunk_size)
            if mini_chunk == b"":  # EOF
                chunk_boundaries[i] = file_size
                break
            
            # Find split token at pos > min_chunk_size in the mini chunk
            index = 0
            while index != -1 and index < min_chunk_size:
                index = mini_chunk.find(split_special_token, index)
            if index != -1:  # update boundary pos
                if index > min_chunk_size:
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
                     drop_special_tokens: bool = True) -> list[bytes]:
    parts = split_by_special_tokens(text, special_tokens)
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_tokens:
                words.append(part.encode('utf-8'))
        else:
            for word in re.findall(pat, part):
                words.append(word.encode('utf-8'))
    return words


def worker(text: str, special_tokens: list[str], q: Queue):
    try:
        words = pretokenize(text, special_tokens)
        counts = defaultdict(int)
        for word in words:
            counts[word] += 1
        q.put(counts)
    except Exception as e:
        q.put(Exception(f"Worker failed: {str(e)}\n{traceback.format_exc()}"))
        

def merge(pair_counts: dict[tuple[int, int], int],
          pair2word: dict[tuple[int, int], set],
          words: list[bytes],
          max_pair: tuple[int, int],
          new_index: int):
    for i in pair2word[max_pair]:
        word = words[i]
        merged_word = []
        pos_list = []  # all pos of new_index in merged_word
        j = 0  # index w.r.t word
        k = 0  # index w.r.t merged word
        
        # Replace max_pair in word with new_index
        while j < len(word):
            if j < len(word) - 1 and (word[j], word[j+1]) == max_pair:
                merged_word.append[new_index]
                j += 2
                pos_list.append(k)
            else:
                merged_word.append(word[j])
                j += 1
            k += 1
        
        # Update count and indices(pointing to its derived word) of pair around new_index 
        for pos in pos_list:
            pair_counts[max_pair] -= 1
            
            if pos > 0:
                if merged_word[pos-1] == new_index:  # e.g. a bc bc e 
                    pair_counts[(max_pair[1], max_pair[0])] -= 1  # #(cb) - 1
                else:
                    pair_counts[(merged_word[pos-1], max_pair[0])] -= 1  # #(ab) - 1
                
                pair_counts[(merged_word[pos-1], new_index)] += 1  # #(abc) + 1
                pair2word[(merged_word[pos-1], new_index)].add(i)
            
            if pos < len(merged_word) - 1:
                if merged_word[pos-1] == new_index:  # e.g. a bc bc e 
                    pair_counts[(max_pair[1], max_pair[0])] -= 1  # #(cb) - 1
                else:
                    pair_counts[(max_pair[1], merged_word[pos+1])] -= 1  # #(ce) - 1
                
                pair_counts[(new_index, merged_word[pos+1])] += 1  # #(bce) + 1
                pair2word[(new_index, merged_word[pos+1])].add(i)
        words[i] = merged_word
                

def train_tokenizer(input_path: str | os.PathLike,
                    vocab_size: int,
                    special_tokens: list[str],
                    num_process: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
            
    # Step 3: Parallelizing Pre-tokenization and Counting
    counts = defaultdict(int)
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
        for word, cnt in result.items():
            counts[word] += cnt  # merge count
        
    for p in processes:
        p.join()
        
    # Step 4: Generate merging rules
    num_merge = max(vocab_size - len(special_tokens) - 256, 0)
    merges = []
    pair_counts = defaultdict(int)
    pair2word = defaultdict(set)
    words = list(counts.keys())
    for i, word in enumerate(words):
        for idx1, idx2 in zip(word[:-1], word[1:]):  # Note: idx1 and idx2 are both int in range(0~255)
            pair_counts[(idx1, idx2)] += counts[word]
            pair2word[(idx1, idx2)].add(i)
            
    def _rank_order(x):
        c1 = vocab[x[0][0]].decode('utf-8', errors="ignore")
        c2 = vocab[x[0][1]].decode('utf-8', errors="ignore")
        return x[1], len(c1) + len(c2), c1, c2
    
    for i in range(num_merge):
        # get most frequent pair, otherwise decided by length then by dict order
        # e.g. ("a", "b") prefer to ("a", "c"), thus ensure reproducibility of executation
        max_pair = max(pair_counts.items(), key=_rank_order)[0]
        
        idx1, idx2 = max_pair
        new_idx = 256 + len(special_tokens) + i
        
        merges.append(vocab[idx1], vocab[idx2])
        # recompute counts of pair of merged token
        merge(pair_counts, pair2word, words, max_pair, new_idx)
    return vocab, merges
    