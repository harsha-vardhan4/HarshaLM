from data.data_preparation import DataPreparation

from model.harsha_lm import HarshaLM

from training.components import (
    TrainingComponents,
)

from data.dataset_builder import DatasetBuilder
from data.parsers.parser_factory import (
    DatasetType,
)

from training.dataloader import (
    create_train_validation_dataloaders,
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

        builder = DatasetBuilder()

        training_corpus = builder.build(
            path=self.config.dataset_path,
            dataset_type=self.config.dataset_type,
        )

        #
        # Prepare data
        #

        preparation = DataPreparation()

        training_samples = preparation.prepare(
            training_corpus
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

        #
        # Train & Validation DataLoaders
        #

        train_dataloader, validation_dataloader = (
            create_train_validation_dataloaders(
                training_samples,
                self.config,
            )
        )

        #
        # Compute training schedule
        #

        steps_per_epoch = len(train_dataloader)

        self.config.max_training_steps = (
            steps_per_epoch * self.config.num_epochs
        )

        #
        # Warm up for 5% of training
        #

        self.config.warmup_steps = max(
            5,
            int(self.config.max_training_steps * 0.05),
        )

        print()

        print("=" * 60)
        print("Training Schedule")
        print("=" * 60)
        print(f"Steps / Epoch : {steps_per_epoch}")
        print(f"Total Steps   : {self.config.max_training_steps}")
        print(f"Warmup Steps  : {self.config.warmup_steps}")
        print("=" * 60)
        print()

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
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            tokenizer=tokenizer,
        )