# corpus.en
| Algorithm | Pre-tokenize time(s) | Merging time(s) | Total time(s) | Device |
|--------|:--------:|:--------:|:--------:|:--------:|
| train_bpe_ori (headpdict) | 0.08 | 0.22 | 0.306 | A100 |
| train_bpe_ori (headpdict) | 0.18 | 0.48 | 0.67 | RTX4090 |
| train_bpe_ori (heapq) | 0.07 | <u>0.17</u> | 0.239 | A100 |
| train_bpe_ori (heapq) | 0.10 | 0.35 | 0.46 | RTX4090 |
| train_bpe_fast (linklist+heapq) | 0.12 | **0.13** | 0.248 | A100 |
| train_bpe_fast (linklist+heapq) | 0.18 | 0.25 | 0.43 | RTX4090 |
| train_bpe_accelerate (linklist+heapdict) | 0.17 | 0.19 | 0.365 | A100 |
| train_bpe_accelerate (linklist+heapdict) | 0.14 | 0.38 | 0.52 | RTX4090 |
Note that Pre-tokenize use theame implementation, their diverse in runtime is due to multiprocessing.  

# TinyStoriesV2
| Algorithm | Pre-tokenize time(s) | Merging time(s) | Total time(s) | Device |
|--------|:--------:|:--------:|:--------:|:--------:|
| train_bpe_ori (headpdict) | 98.99 | 5.11 | 108.49 | A100 |
| train_bpe_ori (headpdict) | 223.5 | 7.8 | 232.53 | RTX4090 |
| train_bpe_ori (heapq) | 97.07 | 4.16 | 105.57 | A100 |
| train_bpe_ori (heapq) | 215.28 | 5.62 | 222.016 | RTX4090 |
| train_bpe_fast (linklist+heapq) | 99.36 | **2.47** | 106.36 | A100 |
| train_bpe_fast (linklist+heapq) | 221.89 | 4.02 | 226.707 | RTX4090 |
| train_bpe_accelerate (linklist+heapdict) | 99.80 | <u>3.38</u> | 107.70 | A100 |
| train_bpe_accelerate (linklist+heapdict) | 221.54 | 7.3 | 230.031 | RTX4090 |