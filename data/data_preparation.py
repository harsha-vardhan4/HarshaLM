from tokenizer.tokenizer import create_tokenizer


class DataPreparation:
    """
    Converts raw text into token IDs.
    """

    def __init__(self):

        self.tokenizer = create_tokenizer()

    def prepare(
        self,
        text: str,
    ) -> list[int]:
        """
        Converts raw text into token IDs.
        """

        token_ids = self.tokenizer.encode(
            text
        )

        print(
            f"Characters : {len(text):,}"
        )

        print(
            f"Tokens      : {len(token_ids):,}"
        )

        return token_ids