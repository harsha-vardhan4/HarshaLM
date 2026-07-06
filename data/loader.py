from pathlib import Path


class TextDatasetLoader:
    """
    Loads one or more text files for training.
    """

    def load_file(
        self,
        filepath: str,
    ) -> str:

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(filepath)

        return path.read_text(
            encoding="utf-8"
        )

    def load_directory(
        self,
        directory: str,
    ) -> str:

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(directory)

        texts = []

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

            texts.append(
                file.read_text(
                    encoding="utf-8"
                )
            )

        return "\n".join(texts)

    def load(
        self,
        path: str,
    ) -> str:

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