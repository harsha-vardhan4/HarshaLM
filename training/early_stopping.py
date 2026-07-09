class EarlyStopping:
    """
    Stops training when the validation loss
    has not improved for a specified number
    of epochs.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False

    def step(
        self,
        validation_loss: float,
    ) -> bool:
        """
        Updates the early stopping state.

        Returns True if training should stop.
        """

        if (
            validation_loss
            <
            self.best_loss
            -
            self.min_delta
        ):

            self.best_loss = validation_loss

            self.counter = 0

            return False

        self.counter += 1

        if self.counter >= self.patience:

            self.should_stop = True

        return self.should_stop
    
    @property
    def remaining_patience(
        self,
    ) -> int:
        """
        Returns the remaining epochs before
        early stopping is triggered.
        """

        return (
            self.patience -
            self.counter
        )


    @property
    def improved(
        self,
    ) -> bool:
        """
        Returns whether the most recent
        validation loss improved.
        """

        return self.counter == 0