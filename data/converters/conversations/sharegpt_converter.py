from data.converters.conversations.json_conversation_converter import (
    JSONConversationConverter,
)


class ShareGPTConverter(
    JSONConversationConverter,
):
    """
    Converts ShareGPT datasets into HarshaLM's
    canonical conversation format.
    """

    ROLE_MAP = {

        "human": "user",

        "gpt": "assistant",

        "system": "system",

    }

    def parse_record(
        self,
        record: dict,
    ) -> list[tuple[str, str]] | None:
        """
        Converts one ShareGPT record into

        [
            ("user", "..."),
            ("assistant", "..."),
        ]
        """

        conversations = record.get(
            "conversations"
        )

        if not conversations:

            return None

        formatted = []

        for message in conversations:

            role = self.ROLE_MAP.get(
                message.get("from", "").lower()
            )

            if role is None:

                #
                # Ignore unsupported roles
                #

                continue

            text = (
                message.get("value", "")
                .strip()
            )

            if not text:

                continue

            formatted.append(
                (
                    role,
                    text,
                )
            )

        #
        # Must contain at least one user and
        # one assistant message.
        #

        roles = {
            role
            for role, _ in formatted
        }

        if (
            "user" not in roles
            or "assistant" not in roles
        ):

            return None

        return formatted