# corpus.en
| Algorithm | Pre-tokenize time(s) | Merging time(s) | Total time(s) | Device |
|--------|:--------:|:--------:|:--------:|:--------:|
| ori (headpdict) | 0.08 | 0.22 | 0.306 | A100 |
| ⬆ori (headpdict) | 0.13 | 0.48 | 0.61 | RTX4090 |
| ori (heapq) | 0.07 | <u>0.17</u> | 0.239 | A100 |
| ⬆ori (heapq) | 0.14 | 0.25 | 0.39 | RTX4090 |
| fast (linklist+heapq) | 0.12 | **0.13** | 0.248 | A100 |
| ⬆fast (linklist+heapq) | 0.12 | 0.25 | 0.39 | RTX4090 |
| accelerate (linklist+heapdict) | 0.17 | 0.19 | 0.365 | A100 |
| ⬆accelerate (linklist+heapdict) | 0.12 | 0.38 | 0.50 | RTX4090 |

Note that Pre-tokenize use the same implementation, their diverse in runtime is due to multiprocessing.  

# TinyStoriesV2
| Algorithm | Pre-tokenize time(s) | Merging time(s) | Total time(s) | Device |
|--------|:--------:|:--------:|:--------:|:--------:|
| ori (headpdict) | 98.99 | 5.11 | 108.49 | A100 |
| ⬆ori (headpdict) | 404.03 | 7.85 | 411.89 | RTX4090 |
| ori (heapq) | 97.07 | 4.16 | 105.57 | A100 |
| ⬆ori (heapq) | 0.14 | 0.25 | 0.39 | RTX4090 |
| fast (linklist+heapq) | 99.36 | **2.47** | 106.36 | A100 |
| ⬆fast (linklist+heapq) | 407.84 | 4.04 | 411.878 | RTX4090 |
| accelerate (linklist+heapdict) | 99.80 | <u>3.38</u> | 107.70 | A100 |
| ⬆accelerate (linklist+heapdict) | 405.89 | 6.57 | 412.490 | RTX4090 |