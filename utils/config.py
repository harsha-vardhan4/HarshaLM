from dataclasses import dataclass


@dataclass
class ModelConfig:

    # Tokenizer
    vocab_size: int = 600

    # Input
    context_length: int = 256

    # Embeddings
    embedding_dim: int = 256

    # Transformer
    num_heads: int = 4
    num_layers: int = 4
    feed_forward_dim: int = 512

    # Regularization
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 30

    # Saving
    checkpoint_dir: str = "checkpoints"

    # Dataset
    dataset_path: str = "data/processed/train.txt"

    # Device
    device: str = "cpu"