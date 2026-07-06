from training.pipeline import TrainingPipeline

from utils.config import ModelConfig


def main():

    config = ModelConfig()

    pipeline = TrainingPipeline(
        config
    )

    components = pipeline.prepare()

    history = components.trainer.train(
        components.dataloader
    )

    print()

    print("=" * 60)

    print("Training Finished")

    print("=" * 60)

    print()

    print(history)


if __name__ == "__main__":
    main()