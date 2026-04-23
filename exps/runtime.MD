# corpus.en
| Algorithm | Pre-tokenize time(s) | Merging time(s) | Total time(s) |
|--------|:--------:|:--------:|:--------:|
| train_bpe_ori (headpdict) | 0.08 | 0.22 | 0.306 |
| train_bpe_ori (heapq) | 0.07 | <u>0.17</u> | 0.239 |
| train_bpe_fast (linklist+heapq) | 0.12 | **0.13** | 0.248 |
| train_bpe_accelerate (linklist+heapdict) | 0.17 | 0.19 | 0.365 |  
Note that Pre-tokenize use the same implementation, their diverse in runtime is due to multiprocessing.  

# TinyStoriesV2
| Algorithm | Pre-tokenize time(s) | Merging time(s) | Total time(s) |
|--------|:--------:|:--------:|:--------:|
| train_bpe_ori (headpdict) | 98.99 | 5.11 | 108.49 |
| train_bpe_ori (heapq) | 97.07 | 4.16 | 105.57 |
| train_bpe_fast (linklist+heapq) | 99.36 | **2.47** | 106.36 |
| train_bpe_accelerate (linklist+heapdict) | 99.80 | <u>3.38</u> | 107.70 |  