from data.conversations.formatted_conversation import (
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


            text += formatted_text



        #
        # GPT-2 EOS token
        #

        text += END_OF_TEXT_TOKEN


        return FormattedConversation(
            text=text,
            messages=messages,
        )
    

    def build_chat_prompt(
        self,
        conversation: list[tuple[str, str]],
    ) -> str:
        """
        Builds a prompt for interactive chat.

        Unlike training, the prompt ends with an
        assistant token instead of END_OF_TEXT_TOKEN,
        allowing the model to continue the assistant's
        response.
        """

        text = ""

        for role, message in conversation:

            if role not in self.ROLE_TOKENS:

                raise ValueError(
                    f"Unknown role: {role}"
                )

            message = message.strip()

            text += (
                f"{self.ROLE_TOKENS[role]}\n"
                f"{message}\n\n"
            )

        #
        # Tell the model it's now the assistant's turn.
        #

        text += (
            f"{ASSISTANT_TOKEN}\n"
        )

        return text