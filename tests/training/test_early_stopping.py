from training.early_stopping import (
    EarlyStopping,
)


def test_early_stopping():

    stopper = EarlyStopping(
        patience=3,
    )

    losses = [
        5.0,
        4.0,
        3.0,
        3.1,
        3.2,
        3.3,
    ]

    stopped = False

    for epoch, loss in enumerate(
        losses,
        start=1,
    ):

        print(
            f"Epoch {epoch}: {loss}"
        )

        if stopper.step(loss):

            stopped = True

            print(
                "Early stopping triggered."
            )

            break

    assert stopped

    print()

    print(
        "✓ Early stopping test passed"
    )


if __name__ == "__main__":

    test_early_stopping()