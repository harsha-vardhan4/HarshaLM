import math

from training.metrics import TrainingMetrics


def test_perplexity():

    loss = 2.0

    perplexity = (
        TrainingMetrics.perplexity(
            loss
        )
    )

    assert abs(
        perplexity - math.exp(loss)
    ) < 1e-6

    print()

    print(
        "Perplexity:",
        perplexity,
    )

    print()

    print(
        "✓ Metrics test passed"
    )


if __name__ == "__main__":

    test_perplexity()