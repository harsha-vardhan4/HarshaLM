import json
from pathlib import Path

from training.logger import (
    TrainingLogger,
)


def test_logger():

    log_path = Path(
        "logs/test_metrics.json"
    )

    logger = TrainingLogger(
        log_path
    )

    logger.log(
        epoch=1,
        train_loss=2.1,
        validation_loss=2.0,
        train_perplexity=8.16,
        validation_perplexity=7.39,
        learning_rate=0.0003,
    )

    assert log_path.exists()

    with open(
        log_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    assert len(data) == 1

    assert (
        data[0]["epoch"]
        == 1
    )

    print()

    print(
        "✓ Logger test passed"
    )


if __name__ == "__main__":

    test_logger()