from abc import ABC, abstractmethod

import json

from pathlib import Path

from data.converters.base_converter import (
    BaseConverter,
)

from data.training_corpus import (
    TrainingCorpus,
)

from data.conversations.conversation_formatter import (
    ConversationFormatter,
)


class JSONConversationConverter(
    BaseConverter,
    ABC,
):
    """
    Base class for JSON conversation datasets.

    Handles:

    - discovering JSON files
    - loading JSON
    - statistics
    - formatting conversations

    Subclasses only implement how one JSON
    record becomes a conversation.
    """

    def __init__(self):

        self.formatter = (
            ConversationFormatter()
        )

    @abstractmethod
    def parse_record(
        self,
        record: dict,
    ) -> list[tuple[str, str]] | None:
        """
        Converts one JSON record into

        [
            ("user", "..."),
            ("assistant", "..."),
        ]

        Return None to skip the record.
        """
        ...

    def convert(
        self,
        dataset_directory: str,
    ) -> TrainingCorpus:

        dataset_directory = Path(
            dataset_directory
        )

        if not dataset_directory.exists():

            raise FileNotFoundError(
                dataset_directory
            )

        json_files = sorted(
            dataset_directory.glob("*.json")
        )

        if not json_files:

            raise ValueError(
                "No .json files found."
            )

        conversations = []

        skipped = 0

        for json_file in json_files:

            with open(
                json_file,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            #
            # Allow either:
            #
            # [ ... ]
            #
            # or
            #
            # { ... }
            #

            if isinstance(
                data,
                dict,
            ):
                data = [data]

            for record in data:

                try:

                    conversation = (
                        self.parse_record(
                            record
                        )
                    )

                except Exception:

                    skipped += 1

                    continue

                if not conversation:

                    skipped += 1

                    continue

                formatted = (
                    self.formatter.format(
                        conversation
                    )
                )

                conversations.append(
                    formatted
                )

        print()

        print("=" * 60)
        print(
            self.__class__.__name__
        )
        print("-" * 60)

        print(
            f"JSON files        : {len(json_files)}"
        )

        print(
            f"Conversations     : {len(conversations):,}"
        )

        print(
            f"Skipped           : {skipped:,}"
        )

        print("=" * 60)

        print()

        return TrainingCorpus(

            dataset_name=(
                dataset_directory.name
            ),

            conversations=conversations,

        )