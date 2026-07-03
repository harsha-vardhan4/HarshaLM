import os
import sentencepiece as spm

from utils.config import ModelConfig


def train_tokenizer():
    """
    Train a SentencePiece tokenizer on our conversation dataset.
    """

    config = ModelConfig()

    input_file = "data/raw/conversations.txt"
    output_dir = "data/tokenizer"

    # Create tokenizer directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    model_prefix = os.path.join(output_dir, "mytokenizer")

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=config.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

    print("\n===================================")
    print("Tokenizer trained successfully!")
    print("===================================")
    print(f"Model : {model_prefix}.model")
    print(f"Vocab : {model_prefix}.vocab")


if __name__ == "__main__":
    train_tokenizer()