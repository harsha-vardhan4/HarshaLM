import math


class TrainingMetrics:
    """
    Computes training metrics for HarshaLM.
    """

    @staticmethod
    def perplexity(
        loss: float,
    ) -> float:
        """
        Computes perplexity from cross-entropy loss.
        """

        try:

            return math.exp(loss)

        except OverflowError:

            return float("inf")