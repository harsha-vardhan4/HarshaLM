from training.pipeline import TrainingPipeline

from utils.config import ModelConfig


def test_pipeline():

    config = ModelConfig()

    pipeline = TrainingPipeline(
        config
    )

    components = pipeline.prepare()

    assert components.model is not None
    assert components.trainer is not None
    assert components.dataloader is not None
    assert components.tokenizer is not None

    print()

    print("Vocabulary Size:")

    print(
        components.tokenizer.vocab_size
    )

    print()

    print(
        f"Batches: {len(components.dataloader)}"
    )

    print()

    print(
        "✓ TrainingPipeline test passed"
    )


if __name__ == "__main__":
    test_pipeline()