import os
import sentencepiece as spm


class Tokenizer:
    """
    Wrapper around SentencePiece tokenizer.

    This class is responsible for:
    - Loading the tokenizer model
    - Encoding text into token IDs
    - Decoding token IDs back into text
    """

    def __init__(self, model_path: str):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found:\n{model_path}"
            )

        self.processor = spm.SentencePieceProcessor()
        self.processor.load(model_path)

    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs.
        """

        return self.processor.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back into text.
        """

        return self.processor.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        """
        Return vocabulary size.
        """

        return self.processor.vocab_size()

    @property
    def pad_id(self):
        return self.processor.pad_id()

    @property
    def bos_id(self):
        return self.processor.bos_id()

    @property
    def eos_id(self):
        return self.processor.eos_id()

    @property
    def unk_id(self):
        return self.processor.unk_id()