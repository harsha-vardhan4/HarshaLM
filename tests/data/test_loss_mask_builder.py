from data.conversations.formatted_conversation import (
    FormattedConversation,
    FormattedMessage,
)

from data.conversations.conversation_loss_mask_builder import ConversationLossMaskBuilder


def test_loss_mask_builder():

    conversation = FormattedConversation(
        text="",
        messages=[
            FormattedMessage(
                role="user",
                text="Hello",
                formatted_text="<|user|>\nHello\n\n",
                learn=False,
            ),
            FormattedMessage(
                role="assistant",
                text="Hi!",
                formatted_text="<|assistant|>\nHi!\n\n",
                learn=True,
            ),
        ],
    )

    print("=" * 60)
    print("Formatted Conversation")
    print("=" * 60)

    print(f"Messages: {len(conversation.messages)}")
    print()

    for message in conversation.messages:
        print(message)
        print()

    builder = ConversationLossMaskBuilder()

    input_ids, labels, loss_mask = builder.build(
        conversation
    )

    print("=" * 60)
    print("Generated Training Data")
    print("=" * 60)

    print("\nInput IDs")
    print(input_ids)

    print("\nLabels")
    print(labels)

    print("\nLoss Mask")
    print(loss_mask)

    print("\nLengths")
    print(f"Input IDs : {len(input_ids)}")
    print(f"Labels    : {len(labels)}")
    print(f"Loss Mask : {len(loss_mask)}")

    #
    # Decode every token with its mask.
    #

    hf = builder.tokenizer.tokenizer

    print()
    print("=" * 60)
    print("Decoded Tokens")
    print("=" * 60)

    for index, (token_id, mask) in enumerate(
        zip(input_ids, loss_mask)
    ):

        token = hf.decode(
            [token_id],
            skip_special_tokens=False,
        )

        print(
            f"{index:03d} | "
            f"{repr(token):25} | "
            f"mask={mask}"
        )


if __name__ == "__main__":
    test_loss_mask_builder()