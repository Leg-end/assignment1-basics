from collections import defaultdict
from tqdm import tqdm
from cs336_basics.maxheapdict import heapdict
from cs336_basics.Tokenizer import BPETrainer
from .train_bpe_fast import bpe_merge, LinkNode
                

class BPETrainerAcc(BPETrainer):
    
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
        
        heap = heapdict()
        for pair, freq in pair_freq.items():
            heap[pair] = (freq, (vocab[pair[0]], vocab[pair[1]]))
        
        pbar = tqdm(total=num_merge, desc="Merging", leave=True)
        merges = []
        for i in range(num_merge):
            if not heap:
                break
            
            max_pair, _ = heap.popitem()  # no zombie elements
            
            new_idx = 256 + len(self.special_tokens) + i
            idx1, idx2 = max_pair
            vocab[new_idx] = vocab[idx1] + vocab[idx2]  # merge into new token
            merges.append((vocab[idx1], vocab[idx2]))
            
            update_pairs = bpe_merge(pair_freq=pair_freq, pair2nodes=pair2nodes,
                                    max_pair=max_pair, new_index=new_idx)
            for pair in update_pairs:
                heap[pair] = (pair_freq[pair], (vocab[pair[0]], vocab[pair[1]]))
            pbar.update(1)
        pbar.close()
        return merges