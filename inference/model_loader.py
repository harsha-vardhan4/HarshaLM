from pathlib import Path

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
        checkpoint_path: str | None = None,
    ) -> HarshaLM:
        """
        Loads a trained model.

        If checkpoint_path is None,
        the best checkpoint is loaded.
        """

        #
        # Default to best checkpoint
        #

        if checkpoint_path is None:

            checkpoint_path = (
                Path(
                    self.config.checkpoint_dir
                )
                /
                self.config.best_checkpoint_name
            )

        #
        # Read checkpoint metadata
        #

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.config.device,
        )

        #
        # Restore training configuration
        #

        saved_config = ModelConfig(
            **checkpoint["config"]
        )

        #
        # Keep current runtime device
        #

        saved_config.device = (
            self.config.device
        )

        #
        # Tokenizer
        #

        tokenizer = create_tokenizer()

        saved_config.vocab_size = (
            tokenizer.vocab_size
        )

        #
        # Build model
        #

        model = HarshaLM(
            saved_config
        )

        #
        # Load weights
        #

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(
            torch.device(
                saved_config.device
            )
        )

        model.eval()

        print(
            f"✓ Loaded checkpoint: "
            f"{checkpoint_path}"
        )

        return model