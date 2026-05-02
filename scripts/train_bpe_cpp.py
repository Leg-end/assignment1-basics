from cpp_scripts.train_bpe_wrapper import py_train_bpe
from cs336_basics.Tokenizer import BPETrainer


class BPETrainerCPP(BPETrainer):
    
    def get_merging_rules(self,
                          vocab: dict[int, bytes],
                          word_freq: dict[bytes, int],
                          num_merge: int):
        # initial word encodings are utf-8
        word_ids = {}
        wordid_counts = {}
        wordid_encodings = {}
        for wid, (word_byte, count) in enumerate(word_freq.items()):
            word = word_byte.decode("utf-8", errors="ignore")
            word_ids[word] = wid
            wordid_counts[wid] = count
            wordid_encodings[wid] = list(word_byte)
        
        merges = []
        vocab_size = num_merge + len(vocab)
        merges_cpp, vocab_cpp = py_train_bpe(vocab_size,
                                             vocab,
                                             wordid_counts,
                                             wordid_encodings,
                                             merges)
        for k, v in vocab_cpp.items():
            vocab[k] = bytes(v)
        merges = [(bytes(arr[0]), bytes(arr[1])) for arr in merges_cpp]
        return merges