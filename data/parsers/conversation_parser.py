class ConversationParser:
    """
    Parses plain-text conversations into
    a structured conversation.

    Expected format:

    User: Hello
    Assistant: Hi!

    User: How are you?
    Assistant: I'm good.
    """

    USER_PREFIX = "User:"
    ASSISTANT_PREFIX = "Assistant:"
    SYSTEM_PREFIX = "System:"

    def parse(
        self,
        text: str,
    ) -> list[tuple[str, str]]:
        """
        Parses a conversation into
        (role, message) tuples.
        """

        conversation = []

        current_role = None

        current_message = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            if line.startswith(self.USER_PREFIX):

                if current_role is not None:

                    conversation.append(
                        (
                            current_role,
                            "\n".join(current_message).strip(),
                        )
                    )

                current_role = "user"

                current_message = [
                    line[len(self.USER_PREFIX):].strip()
                ]

            elif line.startswith(
                self.ASSISTANT_PREFIX
            ):

                if current_role is not None:

                    conversation.append(
                        (
                            current_role,
                            "\n".join(current_message).strip(),
                        )
                    )

                current_role = "assistant"

                current_message = [
                    line[
                        len(
                            self.ASSISTANT_PREFIX
                        ):
                    ].strip()
                ]

            elif line.startswith(
                self.SYSTEM_PREFIX
            ):

                if current_role is not None:

                    conversation.append(
                        (
                            current_role,
                            "\n".join(current_message).strip(),
                        )
                    )

                current_role = "system"

                current_message = [
                    line[
                        len(
                            self.SYSTEM_PREFIX
                        ):
                    ].strip()
                ]

            else:

                current_message.append(
                    line
                )

        if current_role is not None:

            conversation.append(
                (
                    current_role,
                    "\n".join(current_message).strip(),
                )
            )

        return conversation