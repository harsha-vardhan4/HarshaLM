import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):
    """
    Cross-entropy loss for autoregressive language modeling.

    Expected input:

        logits:
            (batch, sequence, vocab_size)

        targets:
            (batch, sequence)
    """

    def __init__(
        self,
        ignore_index: int = -100
    ):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes language modeling loss.
        """

        batch_size, sequence_length, vocab_size = logits.shape

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size
        )

        targets = targets.reshape(
            batch_size * sequence_length
        )

        loss = self.loss_fn(
            logits,
            targets
        )

        return loss