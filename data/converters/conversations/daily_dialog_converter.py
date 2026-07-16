from pathlib import Path

import pandas as pd

from data.converters.base_converter import (
    BaseConverter,
)

from data.training_corpus import (
    TrainingCorpus,
)

from data.conversations.conversation_formatter import (
    ConversationFormatter,
)


class DailyDialogConverter(BaseConverter):
    """
    Converts a DailyDialog CSV file into
    HarshaLM's canonical conversation format.
    """


    def __init__(self):

        self.formatter = ConversationFormatter()


    def _parse_dialog(
        self,
        dialog: str,
    ) -> list[str]:
        """
        Parses DailyDialog dialog column.
        """

        dialog = str(dialog).strip()

        if not dialog:

            return []


        if (
            dialog.startswith("[")
            and dialog.endswith("]")
        ):

            dialog = dialog[1:-1]


        utterances = dialog.split("' '")


        cleaned = []


        for utterance in utterances:

            utterance = (
                utterance
                .replace("'", "")
                .strip()
            )


            if utterance:

                cleaned.append(
                    utterance
                )


        return cleaned

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

        conversations = []

        skipped = 0

        total_user_messages = 0

        total_assistant_messages = 0

        csv_files = sorted(
            dataset_directory.glob("*.csv")
        )

        if not csv_files:

            raise ValueError(
                "No .csv files found."
            )

        for csv_file in csv_files:

            dataframe = pd.read_csv(
                csv_file
            )

            if "dialog" not in dataframe.columns:

                raise ValueError(
                    f"'dialog' column not found in {csv_file.name}"
                )

            for _, row in dataframe.iterrows():

                try:

                    utterances = self._parse_dialog(
                        row["dialog"]
                    )

                except Exception:

                    skipped += 1

                    continue

                if len(utterances) < 2:

                    skipped += 1

                    continue

                conversation = []

                for index, utterance in enumerate(
                    utterances
                ):

                    role = (
                        "user"
                        if index % 2 == 0
                        else "assistant"
                    )

                    conversation.append(
                        (
                            role,
                            utterance.strip(),
                        )
                    )

                    if role == "user":

                        total_user_messages += 1

                    else:

                        total_assistant_messages += 1

                formatted = self.formatter.format(
                    conversation
                )

                conversations.append(
                    formatted
                )

        print()

        print("=" * 60)
        print("DailyDialog Converter")
        print("-" * 60)

        print(
            f"CSV files            : {len(csv_files)}"
        )

        print(
            f"Conversations        : {len(conversations):,}"
        )

        print(
            f"Skipped              : {skipped:,}"
        )

        print(
            f"User Messages        : {total_user_messages:,}"
        )

        print(
            f"Assistant Messages   : {total_assistant_messages:,}"
        )

        print("=" * 60)

        print()

        return TrainingCorpus(

            dataset_name=dataset_directory.name,

            conversations=conversations,

        )