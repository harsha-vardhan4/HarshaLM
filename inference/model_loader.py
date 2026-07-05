import torch

from model.harsha_lm import HarshaLM
from tokenizer.tokenizer import create_tokenizer
from training.checkpoint import CheckpointManager
from utils.config import ModelConfig


class ModelLoader:
    """
    Loads a trained HarshaLM model.
    """

    def __init__(
        self,
        config: ModelConfig,
    ):
        self.config = config

    def load(
        self,
        checkpoint_path: str,
    ) -> HarshaLM:

        tokenizer = create_tokenizer()

        # Ensure the config matches training.
        self.config.vocab_size = tokenizer.vocab_size

        model = HarshaLM(
            self.config
        )

        checkpoint_manager = CheckpointManager(
            self.config
        )

        checkpoint_manager.load(
            checkpoint_path,
            model,
        )

        model.to(
            torch.device(self.config.device)
        )

        model.eval()

        return model