from tokenizer.tokenizer import create_tokenizer


def test_special_tokens():

    tokenizer = create_tokenizer()

    text = "Hello world"

    print("\n" + "=" * 60)
    print("SPECIAL TOKENS")
    print("=" * 60)

    print(f"PAD Token : {tokenizer.pad_token}")
    print(f"PAD ID    : {tokenizer.pad_token_id}")

    print(f"BOS Token : {tokenizer.bos_token}")
    print(f"BOS ID    : {tokenizer.bos_token_id}")

    print(f"EOS Token : {tokenizer.eos_token}")
    print(f"EOS ID    : {tokenizer.eos_token_id}")

    print(f"UNK Token : {tokenizer.unk_token}")
    print(f"UNK ID    : {tokenizer.unk_token_id}")

    print()

    ids_without = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    print("Without Special Tokens")
    print(ids_without)
    print(tokenizer.decode(ids_without))

    print()

    ids_with = (
        [tokenizer.bos_token_id]
        +
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        +
        [tokenizer.eos_token_id]
    )

    print("With Manual BOS/EOS")
    print(ids_with)

    print(
        tokenizer.decode(
            ids_with
        )
    )

    print()

    print(
        "Extra Tokens Added:",
        len(ids_with) - len(ids_without)
    )

    print()

    print("First Token :", ids_with[0])
    print("Last Token  :", ids_with[-1])

    assert (
        ids_with[0]
        == tokenizer.bos_token_id
    )

    assert (
        ids_with[-1]
        == tokenizer.eos_token_id
    )

    assert (
        len(ids_with)
        ==
        len(ids_without) + 2
    )

    print()

    print("✓ Manual BOS/EOS insertion test passed")


if __name__ == "__main__":
    test_special_tokens()