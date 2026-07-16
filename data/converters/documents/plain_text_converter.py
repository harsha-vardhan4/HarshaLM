from pathlib import Path

from data.converters.base_converter import (
    BaseConverter,
)

from data.training_corpus import (
    TrainingCorpus,
)

from data.document.document_formatter import (
    DocumentFormatter,
)


class PlainTextConverter(BaseConverter):
    """
    Converts one or more plain-text files into
    HarshaLM document format.
    """

    def __init__(self):

        self.formatter = DocumentFormatter()

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

        txt_files = sorted(
            dataset_directory.glob("*.txt")
        )

        if not txt_files:

            raise ValueError(
                "No .txt files found."
            )

        documents = []

        total_characters = 0

        for txt_file in txt_files:

            text = txt_file.read_text(
                encoding="utf-8"
            )

            total_characters += len(
                text
            )

            documents.append(

                self.formatter.format(
                    text
                )

            )

        print()

        print("=" * 60)
        print("Plain Text Converter")
        print("-" * 60)

        print(
            f"TXT files            : {len(txt_files)}"
        )

        print(
            f"Documents            : {len(documents)}"
        )

        print(
            f"Characters           : {total_characters:,}"
        )

        print("=" * 60)

        print()

        return TrainingCorpus(

            dataset_name=dataset_directory.name,

            documents=documents,

        )