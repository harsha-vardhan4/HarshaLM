from data.formatted_conversation import (
    FormattedConversation,
    FormattedMessage,
)

from utils.special_tokens import (
    USER_TOKEN,
    ASSISTANT_TOKEN,
    SYSTEM_TOKEN,
    END_OF_TEXT_TOKEN,
)


class ConversationFormatter:
    """
    Formats conversations into HarshaLM's
    canonical training format.
    """

    ROLE_TOKENS = {
    "user": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
    "system": SYSTEM_TOKEN,
}

    LEARN_ROLES = {
        "assistant": True,
        "user": False,
        "system": False,
    }

    def format(
        self,
        conversation: list[tuple[str, str]],
    ) -> FormattedConversation:

        text = ""

        messages = []

        for role, message in conversation:

            if role not in self.ROLE_TOKENS:

                raise ValueError(
                    f"Unknown role: {role}"
                )

            message = message.strip()

            #
            # Preserve message structure
            #
            formatted_text = (
                f"{self.ROLE_TOKENS[role]}\n"
                f"{message}\n\n"
            )

            messages.append(

                FormattedMessage(
                    role=role,
                    text=message,
                    formatted_text=formatted_text,
                    learn=self.LEARN_ROLES[role],
                )
            )

            #
            # Build canonical training text
            #

            text += formatted_text

        text += END_OF_TEXT_TOKEN

        return FormattedConversation(
            text=text,
            messages=messages,
        )