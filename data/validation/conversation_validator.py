from data.validation.base_validator import (
    BaseValidator,
)

from data.conversations.formatted_conversation import (
    FormattedConversation,
)


class ConversationValidator(
    BaseValidator
):
    """
    Validates formatted conversations.
    """

    VALID_ROLES = {

        "user",

        "assistant",

        "system",

    }

    def validate(
        self,
        conversation: FormattedConversation,
    ) -> bool:

        if not conversation.messages:

            return False

        has_user = False

        has_assistant = False

        for message in conversation.messages:

            if not message.text.strip():

                return False

            if message.role not in self.VALID_ROLES:

                return False

            if message.role == "user":

                has_user = True

            elif message.role == "assistant":

                has_assistant = True

        return (
            has_user
            and has_assistant
        )