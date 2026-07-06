from dataclasses import dataclass


@dataclass
class ModelConfig:

    # ----------------------------
    # Tokenizer
    # ----------------------------

    vocab_size: int = 600

    # ----------------------------
    # Input
    # ----------------------------

    context_length: int = 16 # for now we will leave it with 16 later we will increase it to 128 or 256

    # ----------------------------
    # Embeddings
    # ----------------------------

    embedding_dim: int = 256

    # ----------------------------
    # Transformer
    # ----------------------------

    num_heads: int = 4
    num_layers: int = 4
    feed_forward_dim: int = 512

    # ----------------------------
    # Regularization
    # ----------------------------

    dropout: float = 0.1
    attention_dropout: float = 0.1

    # ----------------------------
    # Training
    # ----------------------------

    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 15
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # ----------------------------
    # Logging
    # ----------------------------

    log_every: int = 10

    # ----------------------------
    # Checkpointing
    # ----------------------------

    checkpoint_dir: str = "checkpoints"
    save_every: int = 5

    # ----------------------------
    # Dataset
    # ----------------------------

    dataset_path: str = "data/processed/train.txt"

    # ----------------------------
    # Miscellaneous
    # ----------------------------

    device: str = "cpu"
    seed: int = 42

    model_name: str = "HarshaLM"
    model_version: str = "0.1.0"
    optimizer: str = "adamw"

    max_training_steps: int = 10000

    num_workers: int = 0
    pin_memory: bool = False
    # Logging
    log_every: int = 10