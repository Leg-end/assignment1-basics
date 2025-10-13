import os
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from .Tokenizer import find_chunk_boundaries, pretokenize

# adapt from https://github.com/Spectual/stanford-cs336-a1/blob/main/cs336_basics/BPETokenizer.py

def worker(text: str, special_tokens: list[str], q: Queue):
    """Worker pretokenizing process for multiprocessing"""
    pretokens = list(pretokenize(text, special_tokens))
    q.put(pretokens)
    # print("done")

def train_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
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
    special_tokens = special_tokens or []
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    # Initialize vocab
    vocab = {}
    vocab = {x:bytes([x]) for x in range(0,256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    merges = []

    # Chunk the text file
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
    middle = time.time()
    print(f"Time taken before pretokenizatiuon: {middle - begin:.2f} s")
    begin = middle

    # Parallelizing pretokenization
    pretokens_list = []
    processes = []
    q = Queue()
    for chunk in chunk_list:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    pretokens_list = [q.get() for _ in processes]

    for p in processes:
        p.join()

    pretokens = [token for tokens in pretokens_list for token in tokens]
    middle = time.time()
    print(f"Pre-tokenization and word counting done in {middle - begin:.2f} s")
    begin = middle

    # Merging
    counts = defaultdict(int)
    index_dict = defaultdict(set)  # Store pretoken location for each pair

    for j, pretoken in enumerate(pretokens):
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[index1, index2] += 1
            index_dict[index1, index2].add(j)

    for i in range(num_merges):
        # Prefer lexicographically greater pair
        # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],  
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]

        index1, index2 = max_pair

        new_index = 256 + len(special_tokens) + i

        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))

        merge(counts, index_dict, pretokens, max_pair, new_index)
    end = time.time()
    print(f"Merging done in {end - begin:.2f} s")
    return (vocab, merges)

def merge(counts: dict[tuple[int, int], int], index_dict: dict[tuple[int, int],set[int]], pretokens: list[list[int]], max_pair: tuple[int, int], new_index: int):
    """Merge the pairs with highest frequency and update counts, index_dict"""
    index_set = index_dict[max_pair]

    for i in index_set:
        pretoken = pretokens[i]
        new_pretoken = []

        pos_list = []   # Store positions of max_pair for each new pretoken after merge
        pos = 0
        j = 0

        # Replace max_pair with new_index in each pretoken
        while j < len(pretoken):
            if (j < len(pretoken)-1) and ((pretoken[j], pretoken[j+1]) == max_pair):
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        # Update counts and index_dict
        for pos in pos_list:
            counts[max_pair] -= 1

            if pos > 0:
                if new_pretoken[pos-1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1    
                else:
                    counts[(new_pretoken[pos-1], max_pair[0])] -= 1

                counts[(new_pretoken[pos-1], new_pretoken[pos])] += 1
                index_dict[(new_pretoken[pos-1], new_pretoken[pos])].add(i)

            if pos < len(new_pretoken)-1:
                if new_pretoken[pos+1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1     
                else:
                    counts[(max_pair[1], new_pretoken[pos+1])] -= 1

                counts[(new_pretoken[pos], new_pretoken[pos+1])] += 1
                index_dict[(new_pretoken[pos], new_pretoken[pos+1])].add(i)

        pretokens[i] = new_pretoken
