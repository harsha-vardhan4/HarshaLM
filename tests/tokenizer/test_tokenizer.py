from tokenizer.tokenizer import create_tokenizer


def test_tokenizer():

    tokenizer = create_tokenizer()

    text = "Hello Harsha!"

    ids = tokenizer.encode(text)

    print("Token IDs:")
    print(ids)

    decoded = tokenizer.decode(ids)

    print("\nDecoded:")
    print(decoded)

    assert decoded == text

    print("\n✓ Tokenizer test passed")


if __name__ == "__main__":
    test_tokenizer()