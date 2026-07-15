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

        loss_mask:
            (batch, sequence)

    Only tokens whose loss_mask == 1 contribute
    to the final loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
    ):
        super().__init__()

        #
        # Compute one loss per token.
        #

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="none",
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes masked language modeling loss.
        """

        batch_size, sequence_length, vocab_size = (
            logits.shape
        )

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        )

        targets = targets.reshape(
            batch_size * sequence_length,
        )

        #
        # Compute per-token loss.
        #

        loss = self.loss_fn(
            logits,
            targets,
        )

        #
        # Standard language modeling.
        #

        if loss_mask is None:

            return loss.mean()

        #
        # Flatten mask.
        #

        loss_mask = loss_mask.reshape(
            batch_size * sequence_length,
        ).float()

        #
        # Apply loss mask.
        #

        masked_loss = (
            loss * loss_mask
        )

        #
        # Average only over valid tokens.
        #

        loss = (
            masked_loss.sum()
            /
            loss_mask.sum().clamp(min=1.0)
        )

        return loss