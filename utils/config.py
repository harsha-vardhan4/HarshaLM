from dataclasses import dataclass
from dataclasses import asdict

from data.parsers.parser_factory import DatasetType


@dataclass
class ModelConfig:

    # ----------------------------
    # Tokenizer
    # ----------------------------

    vocab_size: int = 10000

    # ----------------------------
    # Input
    # ----------------------------

    context_length: int = 32 # for now we will leave it with 16 later we will increase it to 128 or 256

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

    batch_size: int = 16 # from 16 to 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 30
    # warmup_steps: int = 100
    max_grad_norm: float = 1.0

    stride: int = 8

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

    dataset_path: str = "datasets"

    # ----------------------------
    # Miscellaneous
    # ----------------------------

    device: str = "cpu"
    seed: int = 42

    model_name: str = "HarshaLM"
    model_version: str = "0.1.0"
    optimizer: str = "adamw"

    # max_training_steps: int = 10000

    num_workers: int = 0
    pin_memory: bool = False
    # Logging
    log_every: int = 10
    validation_split: float = 0.2
    best_checkpoint_name: str = (
        "best_model.pt"
    )

    #
    # Early Stopping
    #

    early_stopping_patience: int = 4

    early_stopping_min_delta: float = 0.001

    #
    # Logging
    #

    log_dir: str = "logs"

    training_log_name: str = "training_metrics.json"


    resume_training: bool = False

    resume_checkpoint: str = "checkpoints/best_model.pt"

    dataset_type: DatasetType = (
        DatasetType.PLAIN_TEXT
    )

    def to_dict(self) -> dict:
        config = asdict(self)

        #
        # Convert enums to strings
        #

        config["dataset_type"] = (
            self.dataset_type.value
        )

        return config
    
    @classmethod
    def from_dict(
        cls,
        data: dict,
    ):

        data = data.copy()

        data["dataset_type"] = DatasetType(
            data["dataset_type"]
        )

        return cls(**data)