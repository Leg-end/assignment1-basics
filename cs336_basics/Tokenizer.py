from typing import Iterable
from typing import BinaryIO, Iterable, Generator, Optional
from tqdm import tqdm
from multiprocessing.synchronize import Event
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from queue import Empty
import multiprocessing as mp
import time
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


class BPETrainer:
    
    def __init__(self,
                 vocab_size: Optional[int] = None,
                 special_tokens: Optional[list[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
    
    @staticmethod
    def _counter_worker(chunk: bytes, special_tokens: list[str]) -> Counter:
        chunk = chunk.decode("utf-8", errors="ignore")
        return Counter(pretokenize(chunk, special_tokens))
    
    @staticmethod
    def _chunk_counter_process(chunk_queue: mp.Queue,
                               counter_queue: mp.Queue,
                               special_tokens: list[str]):
        while True:
            chunk = chunk_queue.get()
            if chunk == None:
                break
            chunk = chunk.decode("utf-8", errors="ignore")
            counter = Counter(pretokenize(chunk, special_tokens))
            counter_queue.put(counter)
            
    @staticmethod       
    def _merge_counter_process(counter_queue: mp.Queue,
                               merged_queue: mp.Queue,
                               timeout: int = 5):
        merged_counter = Counter()
        active = True
        while active:
            try:
                counter = counter_queue.get(timeout=timeout)
                if counter is None:
                    break
                merged_counter.update(counter)
            except Empty:
                # 超时但可能还有数据，继续等待
                continue
        merged_queue.put(merged_counter)
        
    @staticmethod
    def _queue_monitor_process(counter_queue: mp.Queue,
                               merged_queue: mp.Queue,
                               event: Event,
                               interval: int = 10):
        while not event.is_set():
            try:
                c_size = counter_queue.qsize()
                m_size = merged_queue.qsize()
                print(f"[Monitor] counter_queue: {c_size}, merged_queue: {m_size}")
            except Exception as e:
                print(f"[Monitor] Error: {e}")
            time.sleep(interval)
            
    def pretokenize_and_count_pool(
        self,
        chunk_generator: Generator[bytes, None, None],
        num_chunk: int,
        num_counter_process: int,
        chunksize: int) -> Counter:
        """
        Counter进程: [=====工作=====] [======工作======] [======工作======]
        Merger进程:                                   [等待] [====合并====]
                            ↑ 严重延迟 ↑
        问题：Merger必须等所有Counter完成才能开始工作
        Pool则必须通过pickle序列化/IPC通信/反序列化这三个步骤实现两个进程的数据传递
        """
         # 优化 chunksize 以改善负载均衡
        optimal_chunksize = max(1, num_chunk // (num_counter_process * 4))
        chunksize = chunksize or optimal_chunksize
        print(f"Using chunksize={chunksize} for load balancing")
            
        # 准备 worker 函数（绑定 special_tokens）
        from functools import partial
        counter_worker = partial(self._counter_worker, special_tokens=self.special_tokens)
                
        with mp.Pool(processes=num_counter_process) as pool:
            count_iter = pool.imap_unordered(counter_worker, chunk_generator, chunksize=chunksize)
            word_freq = Counter()
            for counter in tqdm(count_iter, total=num_chunk, desc="Megering counter", leave=True):
                word_freq.update(counter)
                
        return word_freq
    
    
    def pretokenize_and_count_pipeline(
        self,
        chunk_generator: Generator[bytes, None, None],
        num_chunk: int,
        num_counter_process: int,
        num_merger_process: int,
        do_monitor: bool) -> Counter:
        """
        Process(在Linux默认)是使用fork来创建进程的，子进程直接继承父进程的地址空间，所以免去了进程间的数据拷贝
        更适合大文件,高并发场景
        Counter进程: [工作] [工作] [工作] [工作] [工作] [工作]
        Merger进程:  [合并] [合并] [合并] [合并] [合并] [合并]
                            ↑ 时间重叠 ↑
        注意：fork的方式启动进程虽然比spawn更快，但是在多线程环境会存在很多问题。根据文档，如果主进程使用了多线程
        就会存在死锁的问题
        """
        chunk_queue = mp.Queue(maxsize=max(1000, num_chunk * 2))
        counter_queue = mp.Queue(maxsize=max(1000, num_chunk * 2))
        merged_queue = mp.Queue(maxsize=num_merger_process)
        
        counter_processes = []
        for i in range(num_counter_process):
            p = mp.Process(target=BPETrainer._chunk_counter_process,
                           args=(chunk_queue, counter_queue, self.special_tokens),
                           name=f"Counter-{i+1}")
            p.start()
            counter_processes.append(p)
            
        merge_processes = []
        for i in range(num_merger_process):
            p = mp.Process(target=BPETrainer._merge_counter_process,
                           args=(counter_queue, merged_queue),
                           name=f"Merger-{i+1}")
            p.start()
            merge_processes.append(p)
            
        if do_monitor:
            stop_event = mp.Event()
            monitor_process = mp.Process(target=BPETrainer._queue_monitor_process,
                                         args=(counter_queue, merged_queue, stop_event))
            monitor_process.start()
        
        for chunk in chunk_generator:
            chunk_queue.put(chunk)
        # 发送终止信号（每个消费者一个None）
        for _ in range(num_counter_process):
            chunk_queue.put(None)
            
        for p in counter_processes:
            p.join()
            
        for _ in range(num_merger_process):
            counter_queue.put(None)
            
        # use main process to merge into final counter
        word_counts = merged_queue.get()
        if num_merger_process > 1:
            for _ in tqdm(range(num_merger_process - 1), desc="Megering counter", leave=True):
                word_counts.update(merged_queue.get())
        
        for p in merge_processes:
            p.join(timeout=10)
            
        if do_monitor:
            stop_event.set()
            monitor_process.join(timeout=5)
            
        return word_counts
        
    
    def get_merging_rules(self,
                          vocab: dict[int, bytes],
                          word_freq: dict[bytes, int],
                          num_merge: int) -> list[tuple[bytes, bytes]]:
        raise NotImplementedError("Subclass must implement this method")
    
    def train(self,
              input_path: str | os.PathLike,
              vocab_size: Optional[int] = None,
              special_tokens: Optional[list[str]] = None,
              num_chunk: int = 4,
              num_counter_process: int = 8,
              num_merger_process: int = 1,
              do_monitor: bool = False,
              chunksize: Optional[int] = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        训练 BPE 模型
        
        Args:
            input_path: 输入文件路径
            vocab_size: 词表大小
            special_tokens: 特殊 token 列表
            num_chunk: 文件分块数量
            num_counter_process: 计数进程数
            num_merger_process: 合并进程数（1表示主进程合并）
            do_monitor: 是否启用队列监控
            chunksize: 每个进程处理的 chunk 数量（负载均衡）
        """
        if not os.path.exists(input_path):
            raise FileExistsError(f"{input_path} not exist!")
        vocab_size = vocab_size or self.vocab_size
        if vocab_size is None:
            raise ValueError("vocab_size must be specified! Either through arguments or constructor.")
        special_tokens = special_tokens or self.special_tokens
        if special_tokens is None:
            raise ValueError("special_tokens must be specified! Either through arguments or constructor.")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
        begin = time.time()
        
        # Step 1: Initialize Vocabulary
        vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(special_tokens):
            token = token.encode("utf-8")
            if token not in vocab.values():
                vocab[256 + i] = token
        
        # Step 2: Chunk the text file
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_chunk, "<|endoftext|>".encode("utf-8"))
        num_chunk = len(boundaries) - 1
        print(f"Split file into {num_chunk} chunks")
        
        def _chunk_generator() -> Generator[bytes, None, None]:
            """
            Read file as generator of chunks, avoiding loading all file into memory.
            """
            with open(input_path, 'rb') as f:
                for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                    f.seek(start)
                    # Faster when transfering between processes with byte data
                    yield f.read(end - start)
        middle = time.time()
        print(f"Time taken before pre-tokenization: {middle - begin:.2f} s")
        begin = middle
                
        # Step 3: Parallelizing Pre-tokenization and Counting
        if num_counter_process is None:
            num_counter_process = min(mp.cpu_count(), 8)
        num_counter_process = min(num_counter_process, num_chunk)
        num_merger_process = max(min(num_merger_process, num_chunk // 2), 1)
        
        print(f"Starting pre-tokenization with {num_counter_process} processes on {num_chunk} chunks...")
        print(f"Merging with {num_merger_process} processes on {num_chunk} counters...")
        if num_counter_process == 1:  # 单chunk时直接顺序处理，避免多进程开销
            chunk = next(_chunk_generator())
            word_freq = Counter(pretokenize(chunk.decode("utf-8", errors="ignore"), self.special_tokens))
        elif num_merger_process == 1:  # 方法一：Pool模式 + 主进程合并（高效简洁）
            word_freq = self.pretokenize_and_count_pool(
                chunk_generator=_chunk_generator(),
                num_chunk=num_chunk,
                num_counter_process=num_counter_process,
                chunksize=chunksize)
        else:  # 方法二：Pipeline模式 + 多进程合并（流水线并行）
            word_freq = self.pretokenize_and_count_pipeline(
                chunk_generator=_chunk_generator(),
                num_chunk=num_chunk,
                num_counter_process=num_counter_process,
                num_merger_process=num_merger_process,
                do_monitor=do_monitor)
            
        middle = time.time()
        print(f"Pre-tokenization and word counting done in {middle - begin:.2f} s")
        print(f"Total unique tokens: {len(word_freq)}")
        begin = middle
        
        # Step 4: Generate merging rules
        num_merge = max(vocab_size - len(special_tokens) - 256, 0)
        merges = self.get_merging_rules(vocab, word_freq, num_merge)
        end = time.time()
        print(f"Merging done in {end - begin:.2f} s")
        
        return vocab, merges
    

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
        self.vocab_size = len(self.encoder)
        self.eos_token_id = 256
        
    
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
                   vocab_path: str,
                   merges_path: str,
                   special_tokens: list[str] | None =None):
        import pickle
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    

def encode_to_nparray_slow(tokenizer: BPETokenizer,
                           path: str,
                           save_path: str):
    with open(path, "r") as f:
        num_lines = sum(1 for _ in f)
    
    total_tokens = 0
    with open(path, "r") as f:
        for line in tqdm(f, total=num_lines, desc="Counting tokens"):
            total_tokens += len(tokenizer.encode(line))
            
    dtype = np.int32
    tokens_mm = np.memmap(save_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    
    pos = 0
    with open(path, "r") as f:
        for line in tqdm(f, total=num_lines, desc="Tokenizing"):
            ids = tokenizer.encode(line)
            n = len(ids)
            tokens_mm[pos:pos+n] = ids
            pos += n
    tokens_mm.flush()
    

def batch_tokenize(batch: list[str],
                   tokenizer: BPETokenizer):
    out = []
    for line in batch:
        out.extend(tokenizer.encode(line))
    return np.array(out, dtype=np.int32)


def encode_to_nparray(tokenizer: BPETokenizer,
                      path: str,
                      save_path: str,
                      batch_size: int = 4096,
                      n_workers: int = 8):
    
    # split into batches
    
    batches = []
    with open(path, "r") as f:
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
            
    total_tokens = 0
    results = []
    # multi-processing tokenization
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = []
        for batch in batches:
            futures.append(exe.submit(batch_tokenize, batch, tokenizer))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
            result = future.result()
            results.append(result)
            total_tokens += result.shape[0]
    
    # write into memmap
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokens_mm = np.memmap(save_path, dtype=np.int32, mode="w+", shape=(total_tokens,))
    pos = 0
    for result in results:
        tokens_mm[pos:pos+result.shape[0]] = result
        pos += result.shape[0]
    tokens_mm.flush()
    