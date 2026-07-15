from transformers import GPT2TokenizerFast


class GPTTokenizer:
    """
    Wrapper around Hugging Face GPT-2 tokenizer.
    """

    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    SYSTEM_TOKEN = "<|system|>"

    def __init__(self):

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2"
        )

        #
        # Add HarshaLM chat special tokens.
        #

        special_tokens = {
            "additional_special_tokens": [

                self.USER_TOKEN,

                self.ASSISTANT_TOKEN,

                self.SYSTEM_TOKEN,

            ]
        }

        self.tokenizer.add_special_tokens(
            special_tokens
        )

        #
        # GPT-2 uses the end-of-text token for padding.
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

    #
    # Padding
    #

    @property
    def pad_token(self):

        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):

        return self.tokenizer.pad_token_id

    #
    # Beginning of sequence
    #

    @property
    def bos_token(self):

        return self.tokenizer.bos_token

    @property
    def bos_token_id(self):

        return self.tokenizer.bos_token_id

    #
    # End of sequence
    #

    @property
    def eos_token(self):

        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):

        return self.tokenizer.eos_token_id

    #
    # Unknown
    #

    @property
    def unk_token(self):

        return self.tokenizer.unk_token

    @property
    def unk_token_id(self):

        return self.tokenizer.unk_token_id

    #
    # User token
    #

    @property
    def user_token(self):

        return self.USER_TOKEN

    @property
    def user_token_id(self):

        return self.tokenizer.convert_tokens_to_ids(
            self.USER_TOKEN
        )

    #
    # Assistant token
    #

    @property
    def assistant_token(self):

        return self.ASSISTANT_TOKEN

    @property
    def assistant_token_id(self):

        return self.tokenizer.convert_tokens_to_ids(
            self.ASSISTANT_TOKEN
        )

    #
    # System token
    #

    @property
    def system_token(self):

        return self.SYSTEM_TOKEN

    @property
    def system_token_id(self):

        return self.tokenizer.convert_tokens_to_ids(
            self.SYSTEM_TOKEN
        )

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

        #
        # Restore HarshaLM special tokens
        #

        special_tokens = {
            "additional_special_tokens": [

                cls.USER_TOKEN,

                cls.ASSISTANT_TOKEN,

                cls.SYSTEM_TOKEN,

            ]
        }

        obj.tokenizer.add_special_tokens(
            special_tokens
        )

        obj.tokenizer.pad_token = (
            obj.tokenizer.eos_token
        )

        obj.tokenizer.model_max_length = (
            1_000_000
        )

        return obj