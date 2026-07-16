from data.conversations.formatted_conversation import (
    FormattedConversation,
)

from tokenizer.tokenizer import create_tokenizer


class ConversationLossMaskBuilder:
    """
    Builds GPT training tensors.

    Produces

    input_ids
    labels
    loss_mask

    Labels are shifted by one token so the model
    learns next-token prediction.
    """

    def __init__(self):

        self.tokenizer = create_tokenizer()

    def build(
        self,
        conversation: FormattedConversation,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
    ]:

        #
        # Build one complete token sequence.
        #

        sequence = [
            self.tokenizer.bos_token_id
        ]

        #
        # BOS is never predicted.
        #

        mask = [0]

        for message in conversation.messages:

            tokens = self.tokenizer.encode(
                message.formatted_text,
                add_special_tokens=False,
            )

            sequence.extend(tokens)

            learn = 1 if message.learn else 0

            mask.extend(
                [learn] * len(tokens)
            )

        #
        # End of conversation.
        #

        sequence.append(
            self.tokenizer.eos_token_id
        )

        #
        # We want the model to learn EOS.
        #

        mask.append(1)

        #
        # ----------------------------------------------------
        # GPT next-token prediction
        # ----------------------------------------------------
        #

        input_ids = sequence[:-1]

        labels = sequence[1:]

        #
        # Labels correspond to the shifted sequence,
        # so shift the mask too.
        #

        loss_mask = mask[1:]

        assert (
            len(input_ids)
            ==
            len(labels)
            ==
            len(loss_mask)
        )

        return (
            input_ids,
            labels,
            loss_mask,
        )