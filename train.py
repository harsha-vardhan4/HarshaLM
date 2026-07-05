import argparse

from model.harsha_lm import HarshaLM

from tokenizer.tokenizer import create_tokenizer

from training.data_preparation import prepare_training_data
from training.dataloader import create_dataloader
from training.trainer import Trainer

from utils.profiles import (
    development_config,
    production_config,
)


def print_training_summary(
    profile,
    config,
    tokenizer,
    token_ids,
    dataloader,
):
    """
    Prints a summary of the training configuration.
    """

    print("\n" + "=" * 50)
    print("HarshaLM Training")
    print("=" * 50)

    print(f"Profile        : {profile}")
    print(f"Device         : {config.device}")
    print(f"Vocabulary     : {tokenizer.vocab_size:,}")
    print(f"Context Length : {config.context_length}")
    print(f"Batch Size     : {config.batch_size}")
    print(f"Epochs         : {config.num_epochs}")
    print(f"Learning Rate  : {config.learning_rate}")

    print("-" * 50)

    print(f"Dataset Path   : {config.dataset_path}")
    print(f"Dataset Tokens : {len(token_ids):,}")
    print(f"Batches/Epoch  : {len(dataloader)}")

    print("=" * 50)
    print()


def main():

    parser = argparse.ArgumentParser(
        description="Train HarshaLM"
    )

    parser.add_argument(
        "--profile",
        choices=[
            "development",
            "production",
        ],
        default="development",
        help="Training profile",
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    if args.profile == "development":
        config = development_config()
    else:
        config = production_config()

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    tokenizer = create_tokenizer()

    config.vocab_size = tokenizer.vocab_size

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    token_ids = prepare_training_data(
        config.dataset_path
    )

    dataloader = create_dataloader(
        token_ids,
        config,
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print_training_summary(
        args.profile,
        config,
        tokenizer,
        token_ids,
        dataloader,
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = HarshaLM(config)

    # -------------------------------------------------
    # Trainer
    # -------------------------------------------------

    trainer = Trainer(
        model=model,
        config=config,
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    trainer.train(
        dataloader
    )


if __name__ == "__main__":
    main()