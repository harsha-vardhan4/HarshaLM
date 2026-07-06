from data.loader import TextDatasetLoader
from data.data_preparation import DataPreparation

from model.harsha_lm import HarshaLM

from training.components import (
    TrainingComponents,
)

from training.dataloader import (
    create_dataloader,
)

from training.trainer import Trainer


class TrainingPipeline:
    """
    Prepares everything required for training HarshaLM.
    """

    def __init__(
        self,
        config,
    ):

        self.config = config

    def prepare(
        self,
        dataset_path: str = "datasets",
    ) -> TrainingComponents:
        """
        Builds the complete training pipeline.
        """

        #
        # Load dataset
        #

        loader = TextDatasetLoader()

        text = loader.load(
            dataset_path
        )

        #
        # Prepare data
        #

        preparation = DataPreparation()

        token_ids = preparation.prepare(
            text
        )

        tokenizer = preparation.tokenizer

        #
        # Update configuration
        #

        self.config.vocab_size = (
            tokenizer.vocab_size
        )

        #
        # DataLoader
        #

        dataloader = create_dataloader(
            token_ids,
            self.config,
        )

        #
        # Model
        #

        model = HarshaLM(
            self.config
        )

        #
        # Trainer
        #

        trainer = Trainer(
            model=model,
            config=self.config,
        )

        #
        # Return everything
        #

        return TrainingComponents(
            model=model,
            trainer=trainer,
            dataloader=dataloader,
            tokenizer=tokenizer,
        )