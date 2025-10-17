| Algorithm | Pre-tokenize time(s) | Merging time(s) | Total time(s) |
|--------|:--------:|:--------:|:--------:|
| train_bpe_ori (headpdict) | 0.08 | 0.22 | 0.306 |
| train_bpe_ori (heapq) | 0.07 | <u>0.17</u> | 0.239 |
| train_bpe_fast (linklist+heapq) | 0.12 | **0.13** | 0.248 |
| train_bpe_fast (linklist+heapdict) | 0.17 | 0.19 | 0.365 |  
Note that Pre-tokenize use theame implementation, their diverse in runtime is due to multiprocessing.