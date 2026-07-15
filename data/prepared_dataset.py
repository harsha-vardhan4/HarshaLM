from dataclasses import dataclass

@property
def bos_token_id(self):
    return self.tokenizer.bos_token_id


@property
def eos_token_id(self):
    return self.tokenizer.eos_token_id


@property
def pad_token_id(self):
    return self.tokenizer.pad_token_id


@property
def user_token_id(self):
    return self.tokenizer.convert_tokens_to_ids("<|user|>")


@property
def assistant_token_id(self):
    return self.tokenizer.convert_tokens_to_ids("<|assistant|>")


@property
def system_token_id(self):
    return self.tokenizer.convert_tokens_to_ids("<|system|>")

@dataclass
class PreparedDataset:
    """
    Tokenized training dataset.

    Attributes
    ----------
    token_ids:
        Complete token stream.

    loss_mask:
        One value per token.

            1 -> compute loss
            0 -> ignore loss
    """

    token_ids: list[int]

    loss_mask: list[int]