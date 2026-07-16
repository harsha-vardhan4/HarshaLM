from data.data_preparation import DataPreparation

from model.harsha_lm import HarshaLM

from training.components import (
    TrainingComponents,
)

from data.dataset_builder import (
    DatasetBuilder,
)

from training.dataloader import (
    create_train_validation_dataloaders,
)

from training.trainer import Trainer

from data.validation.corpus_validator import (
    CorpusValidator,
)



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
    ) -> TrainingComponents:
        """
        Builds the complete training pipeline.
        """


        #
        # Build canonical training corpus
        #

        builder = DatasetBuilder()


        training_corpus = builder.build(
            path=self.config.dataset_path,
        )

        validator = CorpusValidator()

        training_corpus, report = (
            validator.validate(
                training_corpus
            )
        )


        #
        # Prepare training samples
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
        # Create dataloaders
        #

        (
            train_dataloader,
            validation_dataloader,
        ) = create_train_validation_dataloaders(
            training_samples,
            self.config,
        )



        #
        # Compute training schedule
        #

        steps_per_epoch = len(
            train_dataloader
        )


        self.config.max_training_steps = (
            steps_per_epoch
            * self.config.num_epochs
        )


        self.config.warmup_steps = max(
            5,
            int(
                self.config.max_training_steps
                * 0.05
            ),
        )


        print()

        print("=" * 60)
        print("Training Schedule")
        print("-" * 60)

        print(
            f"Steps / Epoch : {steps_per_epoch}"
        )

        print(
            f"Total Steps   : {self.config.max_training_steps}"
        )

        print(
            f"Warmup Steps  : {self.config.warmup_steps}"
        )

        print("=" * 60)

        print()



        #
        # Build model
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



        return TrainingComponents(
            model=model,
            trainer=trainer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            tokenizer=tokenizer,
        )