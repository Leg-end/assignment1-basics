from .train_bpe_ori import BPETrainerOri, BPETrainer
from .train_bpe_fast import BPETrainerFast
from .train_bpe_accelerate import BPETrainerAcc
from .train_bpe_cpp import BPETrainerCPP

BPE_TRAINERS = {
    "ori": BPETrainerOri,
    "fast": BPETrainerFast,
    "accelerate": BPETrainerAcc,
    "cpp": BPETrainerCPP
}

def get_bpe_trainer(name: str,
                    **kwargs) -> BPETrainer:
    try:
        trainer = BPE_TRAINERS[name](**kwargs)
        return trainer
    except KeyError:
        raise KeyError(f"Invalid BPE trainer name: {name}, allowed: {list(BPE_TRAINERS.keys())}")