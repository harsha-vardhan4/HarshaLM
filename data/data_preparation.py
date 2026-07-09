from tokenizer.tokenizer import create_tokenizer


class DataPreparation:
    """
    Converts conversations into token IDs.
    """

    def __init__(self):

        self.tokenizer = create_tokenizer()

    def prepare(
        self,
        conversations: list[str],
    ) -> list[int]:
        """
        Converts conversations into one stream of token IDs.
        """

        all_token_ids = []

        total_characters = 0
        total_tokens = 0

        for conversation in conversations:

            token_ids = (
                [self.tokenizer.bos_token_id]
                +
                self.tokenizer.encode(
                    conversation,
                    add_special_tokens=False,
                )
                +
                [self.tokenizer.eos_token_id]
            )

            all_token_ids.extend(
                token_ids
            )

            total_characters += len(
                conversation
            )

            total_tokens += len(
                token_ids
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

        return all_token_ids