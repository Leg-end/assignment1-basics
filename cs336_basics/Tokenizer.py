from typing import Iterable
from typing import BinaryIO, Iterable, Generator
from tqdm import tqdm
import os
import regex as re
import numpy as np


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


class BPETokenizer:

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        special_tokens = special_tokens or []
        # Ensure special tokens are in the vocabulary
        self.cache: dict[bytes, tuple[bytes]] = {}  # word : tuple of subwords
        self.decoder = vocab
        self.encoder: dict[bytes, int] = {v: k for k, v in vocab.items()}
        n = len(vocab)
        for i, token in enumerate(special_tokens):
            byte_token = token.encode('utf-8')
            self.cache[byte_token] = (byte_token,)
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
        import pickle
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    

def encode_to_array_slow(tokenizer: BPETokenizer,
                         path: str,
                         save_path: str):
    total_tokens = 0
    with open(path, "r") as f:
        for line in tqdm(f, desc="Counting tokens"):
            total_tokens += len(tokenizer.encode(line))
            
    dtype = np.int32
    