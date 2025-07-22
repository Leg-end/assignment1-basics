import os
import regex as re
import ftfy
import html
import gzip
import traceback

from functools import lru_cache
from collections import defaultdict
from typing import BinaryIO, Iterable
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


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    # replace continue whitespaces into single whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()  # remove leading and tailing whitespace
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
                drop_special_tokens: bool = True) -> list[str]:
    """
    sentence -> words
    split sentence by special tokens, then split each part by regex pattern
    """
    parts = split_by_special_tokens(text, special_tokens)
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_tokens:
                words.append(part)
        else:
            for word in re.findall(pat, part):
                words.append(word)
    return words


def worker(text: str, special_tokens: list[str], q: Queue):
    try:
        words = pretokenize(text, special_tokens)
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
        
        # Update count and indices(pointing to its derived word) of pair around merged_pair 
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
                if merged_word[pos+1] == new_index:  # e.g. a bc bc e 
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
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
            
    # Step 3: Parallelizing Pre-tokenization and Counting
    words_list = []
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
        words_list.append(result)
        
    for p in processes:
        p.join()
        
    words = [word.encode("utf-8") for sublist in words_list for word in sublist]
        
    # Step 4: Generate merging rules
    num_merge = max(vocab_size - len(special_tokens) - 256, 0)
    merges = []
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair2word: dict[tuple[int, int], set] = defaultdict(set)
    for i, word in enumerate(words):
        for idx1, idx2 in zip(word[:-1], word[1:]):  # Note: idx1 and idx2 are both int in range(0~255)
            pair_counts[(idx1, idx2)] += 1
            pair2word[(idx1, idx2)].add(i)
    
    for i in range(num_merge):
        # get most frequent pair, otherwise decided by length then by dict order
        # e.g. ("a", "b") prefer to ("a", "c"), thus ensure reproducibility of executation
        max_pair = max(
            pair_counts.items(),
            key=lambda x: (
                x[1],  
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]
        
        idx1, idx2 = max_pair
        new_idx = 256 + len(special_tokens) + i
        vocab[new_idx] = vocab[idx1] + vocab[idx2]  # merge into new token
        merges.append((vocab[idx1], vocab[idx2]))
        # recompute counts of pair of merged token
        bpe_merge(pair_counts, pair2word, words, max_pair, new_idx)
    import json
    import base64
    json.dump({k: base64.b64encode(v).decode("utf-8") for k, v in vocab.items()},
              open("./data/vocab.json", "w"))
    with open("./data/merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('utf-8', errors='ignore')} {merge[1].decode('utf-8', errors='ignore')}\n")
    
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
        
    
    def bpe(self, word: str) -> list[bytes]:
        """
        Apply bpe merge
        turn word into subwords(split by space)
        """
        if not word:  # empty word
            return [word.encode('utf-8')]  # must be list or tuple
        if word in self.cache:
            return self.cache[word]
        # treat token as tuple of symbols
        tokens = tuple(bytes([w]) for w in word.encode('utf-8'))
        
        pairs = get_pairs(tokens)
        
        if not pairs:
            return [word.encode('utf-8')]  # must be list or tuple
        
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
    
    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        for chunk in iterable:
            yield self.encode(chunk)
    
    def decode(self, token_ids: list[int]) -> str:
        # ids(int) -> subwords(bytes) -> words(bytes) -> text(str)
        text = b''.join([self.decoder[token_id] for token_id in token_ids])
        text = text.decode('utf-8', errors="replace")
        return text
    