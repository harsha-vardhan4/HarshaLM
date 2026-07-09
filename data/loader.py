from pathlib import Path


class TextDatasetLoader:
    """
    Loads one or more text files as individual conversations.
    """

    def _split_conversations(
        self,
        text: str,
    ) -> list[str]:
        """
        Splits text into conversations.

        Conversations are separated by
        one or more blank lines.
        """

        conversations = []

        for conversation in text.split("\n\n"):

            conversation = conversation.strip()

            if conversation:

                conversations.append(
                    conversation
                )

        return conversations


    def load_file(
        self,
        filepath: str,
    ) -> list[str]:

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(filepath)

        text = path.read_text(
            encoding="utf-8",
        )

        return self._split_conversations(
            text
        )


    def load_directory(
        self,
        directory: str,
    ) -> list[str]:

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(directory)

        conversations = []

        files = sorted(
            directory.glob("*.txt")
        )

        if not files:

            raise ValueError(
                "No .txt files found."
            )

        for file in files:

            print(
                f"Loading {file.name}"
            )

            text = file.read_text(
                encoding="utf-8",
            )

            conversations.extend(
                self._split_conversations(
                    text
                )
            )

        return conversations


    def load(
        self,
        path: str,
    ) -> list[str]:

        path = Path(path)

        if path.is_file():

            return self.load_file(
                str(path)
            )

        if path.is_dir():

            return self.load_directory(
                str(path)
            )

        raise FileNotFoundError(path)