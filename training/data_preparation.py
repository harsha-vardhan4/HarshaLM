from pathlib import Path

from tokenizer.tokenizer import create_tokenizer


def prepare_training_data(
    dataset_path: str,
) -> list[int]:
    """
    Loads a text file and converts it into token IDs.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    tokenizer = create_tokenizer()

    token_ids = tokenizer.encode(text)

    print(f"Characters : {len(text):,}")
    print(f"Tokens      : {len(token_ids):,}")

    return token_ids