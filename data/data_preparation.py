from data.formatted_conversation import FormattedConversation
from data.training_sample import TrainingSample
from tokenizer.tokenizer import create_tokenizer
from data.loss_mask_builder import LossMaskBuilder


class DataPreparation:
    """
    Converts conversations into token IDs.
    """

    def __init__(self):

        self.tokenizer = create_tokenizer()
        self.loss_mask_builder = LossMaskBuilder()

    def prepare(
        self,
        conversations: list[FormattedConversation],
    ) -> list[TrainingSample]:
        """
        Converts conversations into one stream of token IDs.
        """

        training_samples = []

        total_characters = 0
        total_tokens = 0

        for conversation in conversations:

            input_ids, labels, loss_mask = (
                self.loss_mask_builder.build(
                    conversation
                )
            )

            labels = input_ids[1:] + [-100]

            loss_mask = [1] * len(input_ids)

            training_samples.append(
                TrainingSample(
                    input_ids=input_ids,
                    labels=labels,
                    loss_mask=loss_mask,
                    conversation_length=len(input_ids),
                )
            )

            total_characters += len(
                conversation.text
            )

            total_tokens += len(
                input_ids
            )

        print(
            f"Conversations : {len(conversations):,}"
        )

        print(
            f"Characters    : {total_characters:,}"
        )

        print(
            f"Tokens        : {total_tokens:,}"
        )

        if conversations:

            print(
                f"Avg Tokens    : "
                f"{total_tokens / len(conversations):.2f}"
            )

        return training_samples