import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingLR(LRScheduler):
    
    def __init__(self, optimizer, last_epoch=-1, verbose="deprecated"):
        super().__init__(optimizer, last_epoch, verbose)