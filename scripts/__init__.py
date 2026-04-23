from .train_bpe_ori import BPETrainerOri, BPETrainer
from .train_bpe_fast import BPETrainerFast
from .train_bpe_accelerate import BPETrainerAcc

BPE_TRAINERS = {
    "ori": BPETrainerOri,
    "fast": BPETrainerFast,
    "accelerate": BPETrainerAcc
}

def get_bpe_trainer(name: str,
                    **kwargs) -> BPETrainer:
    try:
        trainer = BPE_TRAINERS[name](**kwargs)
        return trainer
    except KeyError:
        raise KeyError(f"Invalid BPE trainer name: {name}, allowed: {list(BPE_TRAINERS.keys())}")