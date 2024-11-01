from dataclasses import dataclass

import torch


@dataclass
class LLMConfig:
    vocab_size: int = 1024
    seq_len: int = 128
    dim_emb: int = 512
    num_layers: int = 8
    num_heads: int = 8
    emb_dropout: float = 0.0
    ffn_dim_hidden: int = 4 * 512
    ffn_bias: bool = True


@dataclass
class TrainingConfig:
    retrain_tokenizer: bool = False
    device: torch.device = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 2
    log_frequency: int = 10
