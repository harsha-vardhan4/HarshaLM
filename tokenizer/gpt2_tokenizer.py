from transformers import GPT2TokenizerFast


class GPTTokenizer:
    """
    Wrapper around Hugging Face GPT-2 tokenizer.
    """

    def __init__(self):

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2"
        )

        # GPT-2 has no pad token by default
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )

    @property
    def vocab_size(self) -> int:

        return len(self.tokenizer)

    def encode(
        self,
        text: str,
    ) -> list[int]:

        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def save(
        self,
        directory: str,
    ):

        self.tokenizer.save_pretrained(
            directory
        )

    @classmethod
    def load(
        cls,
        directory: str,
    ):

        obj = cls()

        obj.tokenizer = GPT2TokenizerFast.from_pretrained(
            directory
        )

        return obj