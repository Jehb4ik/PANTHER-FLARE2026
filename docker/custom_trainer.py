"""Inference-only stand-in for the training trainer.

The checkpoints were trained with `nnUNetTrainer_Epoch5000_Lr1e3` (see
../training/). nnU-Net only needs a class with this name to resolve the
checkpoint at inference; the architecture itself comes from plans.json.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_Epoch5000_Lr1e3(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 5000
        self.initial_lr = 1e-3
