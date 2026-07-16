from data.converters.conversations.json_conversation_converter import (
    JSONConversationConverter,
)


class AlpacaConverter(
    JSONConversationConverter,
):
    """
    Converts Alpaca instruction datasets into
    HarshaLM's canonical conversation format.
    """

    def parse_record(
        self,
        record: dict,
    ) -> list[tuple[str, str]] | None:
        """
        Converts one Alpaca record into

        [
            ("user", "..."),
            ("assistant", "..."),
        ]
        """

        instruction = (
            record.get(
                "instruction",
                "",
            )
            .strip()
        )

        input_text = (
            record.get(
                "input",
                "",
            )
            .strip()
        )

        output = (
            record.get(
                "output",
                "",
            )
            .strip()
        )

        #
        # Output is required.
        #

        if not instruction or not output:

            return None

        #
        # Build user prompt.
        #

        if input_text:

            user_message = (
                f"{instruction}\n\n"
                f"{input_text}"
            )

        else:

            user_message = instruction

        return [

            (
                "user",
                user_message,
            ),

            (
                "assistant",
                output,
            ),

        ]