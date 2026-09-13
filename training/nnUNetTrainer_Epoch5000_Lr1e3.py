"""nnU-Net trainer with a 5000-epoch schedule and initial learning rate 1e-3.

Everything else (SGD + Nesterov 0.99, PolyLR 0.9, Dice + CE with deep
supervision, default augmentation, 250 iterations per epoch, oversampling
0.33) is inherited from stock nnUNetTrainer (nnunetv2 2.8.1). The schedule
follows the public FLARE 2025 Task 1 code (github.com/zpy2223/MICCAI-FLARE-2025-Task-1).

Install: copy this file into
nnunetv2/training/nnUNetTrainer/variants/training_length/ and train with
`-tr nnUNetTrainer_Epoch5000_Lr1e3`.
"""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_Epoch5000_Lr1e3(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5000
        self.initial_lr = 1e-3
