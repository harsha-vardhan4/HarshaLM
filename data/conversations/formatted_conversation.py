from dataclasses import dataclass


@dataclass
class FormattedMessage:
    """
    One formatted message inside a conversation.
    """

    role: str

    text: str

    formatted_text: str

    learn: bool = False


@dataclass
class FormattedConversation:
    """
    A formatted conversation ready for tokenization.
    """

    text: str

    messages: list[FormattedMessage]