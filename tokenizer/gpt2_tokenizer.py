from transformers import GPT2TokenizerFast


class GPTTokenizer:
    """
    Wrapper around Hugging Face GPT-2 tokenizer.
    """

    def __init__(self):

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2"
        )

        #
        # Use GPT-2 end-of-text token for padding.
        #

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )

        #
        # Avoid warnings for long datasets.
        #

        self.tokenizer.model_max_length = (
            1_000_000
        )

    @property
    def vocab_size(self) -> int:

        return len(self.tokenizer)

    @property
    def pad_token(self):

        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):

        return self.tokenizer.pad_token_id

    @property
    def bos_token(self):

        return self.tokenizer.bos_token

    @property
    def bos_token_id(self):

        return self.tokenizer.bos_token_id

    @property
    def eos_token(self):

        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):

        return self.tokenizer.eos_token_id

    @property
    def unk_token(self):

        return self.tokenizer.unk_token

    @property
    def unk_token_id(self):

        return self.tokenizer.unk_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:

        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
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

        obj.tokenizer = (
            GPT2TokenizerFast.from_pretrained(
                directory
            )
        )

        obj.tokenizer.pad_token = (
            obj.tokenizer.eos_token
        )

        obj.tokenizer.model_max_length = (
            1_000_000
        )

        return obj