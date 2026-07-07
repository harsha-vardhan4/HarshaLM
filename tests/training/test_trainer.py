from training.pipeline import TrainingPipeline

from utils.config import ModelConfig


def test_trainer():

    config = ModelConfig()

    config.num_epochs = 1

    pipeline = TrainingPipeline(
        config
    )

    components = pipeline.prepare()

    history = components.trainer.train(
        components.train_dataloader,
        components.validation_dataloader,
    )

    assert history is not None

    print()

    print(history)

    print()

    print("✓ Trainer test passed")


if __name__ == "__main__":
    test_trainer()