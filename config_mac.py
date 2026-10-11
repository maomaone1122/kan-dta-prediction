
import os
import torch
from dataclasses import dataclass
from typing import List

@dataclass
class Config:

    data_dir: str = "./data"
    davis_path: str = "./data/davis.csv"
    kiba_path: str = "./data/kiba.csv"
    

    dataset: str = "kiba"
    drug_max_len: int = 100
    target_max_len: int = 1000
    embed_dim: int = 64
    

    hidden_dim: int = 48
    drug_kernels: tuple = (3, 5, 7)
    target_kernels: tuple = (5, 7, 9)
    fusion_dim: int = 48
    

    grid_size: int = 4
    spline_order: int = 3
    

    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    

    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 2
    seed: int = 42
    

    new_drug_setting: bool = True
    n_folds: int = 2
    n_repeats: int = 3
    

    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.dataset == "kiba":
            self.drug_max_len = 100
            self.target_max_len = 1000


config = Config()
