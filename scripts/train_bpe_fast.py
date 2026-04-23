import os
import heapq
import time

from collections import defaultdict
from cs336_basics.Tokenizer import encode_to_nparray, BPETokenizer, BPETrainer
from tqdm import tqdm

import hydra
import pickle
import tiktoken
from omegaconf import DictConfig


class LinkNode:
    """表示词内一个 token 节点，便于链表原地更新。"""
    def __init__(self, value, word_freq):
        self.value = value
        self.word_freq = word_freq  # 共享引用，节省内存
        self.prev = None
        self.next = None


class PQItem:
    """定义优先队列元素，实现自定义比较：频率优先，其次按字典序逆序。"""
    def __init__(self, freq: int, id_pair: tuple[int, int], byte_pair: tuple[bytes, bytes]):
        self.freq = freq
        self.id_pair = id_pair
        self.byte_pair = byte_pair

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq > other.freq  # 频率高的先出
        return self.byte_pair > other.byte_pair  # 字典序大的先出
    

def bpe_merge(
    pair_freq: dict[tuple[int, int], int],
    pair2nodes: dict[tuple[int, int], set[LinkNode]],
    max_pair: tuple[int, int],
    new_index: int
    ):
    """
    For case of continuious max_pair, e.g. abcbcbcd, we need do carefully by following steps:
    1. remove related pairs around max_pair and avoid redundance removal, e.g. remove (c, b) between bcbc twice
    2. after merge, we get a,bc,bc,bc,d, then we add new pairs around max_pair and avoid redundance addition, e.g.
       add (bc, bc) twice. so for new left pair, we only add when left is not merging pair, and always add new right pair.
    3. update pair_freq
    Args:
        pair_freq: store symbol pair and its frquency
        pair2nodes: store symbol pair and indices of word that contains it
        max_pair: symobal pair with maximum frquency
        new_index: new index for max_pair to store in vocab
    """
    nodes = list(pair2nodes[max_pair])
    update_pairs = set()
    # remove related pairs
    for node1 in nodes:  # max_pair : node, node.next
        node2 = node1.next
        if node2 is None:
            continue
        cnt = node1.word_freq['cnt']
        left = node1.prev
        right = node2.next
        if left:
            # remove left pair if left is not merging pair, e.g. not case "a,b_1 c_1,[b_2,c_2],b_3,c_3,d"
            # if b_1 c_1 already merged, then b_1.next = b_2, left = b_2.prev = b_1, left.value = b_1 c_1 = new_index
            # no need to remove (b_1 c_1, b_2)
            # else b_1 c_1 not merged, then b_1.next = c_1, left = b_2.prev = c_1, left.value = c_1 != new_index
            # need to remove (c_1, b_2)
            if left.value != new_index:
                old_left_pair = (left.value, node1.value)
                pair2nodes[old_left_pair].discard(left)
                pair_freq[old_left_pair] -= cnt
                update_pairs.add(old_left_pair)
        if right:
            # remove right pair if right is not merging pair, e.g. not case "a,b_1 c_1,[b_2,c_2],b_3,c_3,d"
            # if b_3 c_3 already merged, then right = c_2.next = b_3, right.value = b_3 c_3 = new_index
            # no need to remove (c_2, b_3 c_3)
            # else b_3 c_3 not merged, then right.value = b_3 != new_index
            # need to remove (c_2, b_3)
            if right.value != new_index:
                old_right_pair = (node2.value, right.value)
                pair2nodes[old_right_pair].discard(node2)
                pair_freq[old_right_pair] -= cnt
                update_pairs.add(old_right_pair)
        # merge node1 and node2 into node1
        node1.value = new_index
        node1.next = right
        if right:
            right.prev = node1
    # add new pairs after merged, e.g. now we have a, b_1 c_1, b_2 c_2, b_3 c_3, d
    # link list now is a -> b_1 -> b_2 -> b_3 -> d
    for node1 in nodes:
        cnt = node1.word_freq['cnt']
        left = node1.prev
        right = node1.next
        # add new left pair if left is not merging pair
        if left and left.value != new_index:
            new_left_pair = (left.value, node1.value)
            pair2nodes[new_left_pair].add(left)
            pair_freq[new_left_pair] += cnt
            update_pairs.add(new_left_pair)
        if right:
            new_right_pair = (new_index, right.value)
            pair2nodes[new_right_pair].add(node1)
            pair_freq[new_right_pair] += cnt
            update_pairs.add(new_right_pair)
    del pair_freq[max_pair]
    del pair2nodes[max_pair]
    return update_pairs


class BPETrainerFast(BPETrainer):
    
    def get_merging_rules(self,
                          vocab: dict[int, bytes],
                          word_freq: dict[bytes, int],
                          num_merge: int) -> list[tuple[bytes, bytes]]:
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
            
        pq = [PQItem(freq, pair, (vocab[pair[0]], vocab[pair[1]])) 
            for pair, freq in pair_freq.items()]
        heapq.heapify(pq)
        
        pbar = tqdm(total=num_merge, desc="Merging", leave=True)
        merges = []
        for i in range(num_merge):
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
            
            new_idx = 256 + len(self.special_tokens) + i
            idx1, idx2 = max_pair
            vocab[new_idx] = vocab[idx1] + vocab[idx2]  # merge into new token
            merges.append((vocab[idx1], vocab[idx2]))
            
            update_pairs = bpe_merge(pair_freq=pair_freq, pair2nodes=pair2nodes,
                                    max_pair=max_pair, new_index=new_idx)
            for pair in update_pairs:
                heapq.heappush(pq, PQItem(pair_freq[pair], pair, (vocab[pair[0]], vocab[pair[1]])))
            pbar.update(1)
        pbar.close()
        return merges

@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    if not os.path.exists(cfg.merges_path) or not os.path.exists(cfg.vocab_path):
        print(f"No vocab and mergers found in {cfg.merges_path} and {cfg.vocab_path}. Training a BPETokenizer.")
        start_time = time.time()
        trainer = BPETrainerFast(special_tokens=cfg.special_tokens,
                                 num_chunks=cfg.num_chunks,
                                 num_counter=cfg.num_processes,
                                 num_merger=cfg.get('num_merger', 1))
        vocab, merges = trainer.train(input_path=cfg.input_path,
                                      vocab_size=cfg.vocab_size)
        end_time = time.time()
        print(f"Finish training in {end_time - start_time:.2f}s")
        with open(cfg.vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(cfg.merges_path, "wb") as f:
            pickle.dump(merges, f)
    tokenizer = BPETokenizer.from_files(vocab_path=cfg.vocab_path,
                                        merges_path=cfg.merges_path,
                                        special_tokens=cfg.special_tokens)
    # tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.vocab_size)
    encode_to_nparray(tokenizer, cfg.train_txt_path, cfg.train_dat_path, cfg.batch_size, cfg.n_workers)
    encode_to_nparray(tokenizer, cfg.valid_txt_path, cfg.valid_dat_path, cfg.batch_size, cfg.n_workers)
    

if __name__ == "__main__":
    main()