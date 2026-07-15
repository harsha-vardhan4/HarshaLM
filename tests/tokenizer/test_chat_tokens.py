from tokenizer.tokenizer import create_tokenizer


def test_chat_tokens():

    tokenizer = create_tokenizer()

    print()

    print("=" * 60)
    print("CHAT SPECIAL TOKENS")
    print("=" * 60)

    print()

    print(f"Vocabulary Size : {tokenizer.vocab_size}")

    print()

    print(f"User Token      : {tokenizer.user_token}")
    print(f"User ID         : {tokenizer.user_token_id}")

    print()

    print(f"Assistant Token : {tokenizer.assistant_token}")
    print(f"Assistant ID    : {tokenizer.assistant_token_id}")

    print()

    print(f"System Token    : {tokenizer.system_token}")
    print(f"System ID       : {tokenizer.system_token_id}")

    print()

    user_ids = tokenizer.encode(
        tokenizer.user_token
    )

    assistant_ids = tokenizer.encode(
        tokenizer.assistant_token
    )

    system_ids = tokenizer.encode(
        tokenizer.system_token
    )

    print(
        "Encoded User Token      :",
        user_ids,
    )

    print(
        "Encoded Assistant Token :",
        assistant_ids,
    )

    print(
        "Encoded System Token    :",
        system_ids,
    )

    #
    # Verify each chat token is encoded
    # as a single token.
    #

    assert len(user_ids) == 1
    assert len(assistant_ids) == 1
    assert len(system_ids) == 1

    assert user_ids[0] == tokenizer.user_token_id
    assert assistant_ids[0] == tokenizer.assistant_token_id
    assert system_ids[0] == tokenizer.system_token_id

    print()

    print("✓ Chat token test passed")


if __name__ == "__main__":

    test_chat_tokens()