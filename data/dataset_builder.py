from pathlib import Path

from data.converters.converter_factory import (
    ConverterFactory,
)

from data.parsers.parser_factory import (
    DatasetType,
)

from data.training_corpus import (
    TrainingCorpus,
)


class DatasetBuilder:
    """
    Builds the final training corpus by discovering
    dataset folders.

    Example:

    datasets/
        plain_text/
        daily_dialog/
        sharegpt/
        alpaca/
    """

    def build(
        self,
        path: str,
    ) -> TrainingCorpus:

        root = Path(path)

        if not root.exists():

            raise FileNotFoundError(root)

        if not root.is_dir():

            raise ValueError(
                "Dataset path must be a directory."
            )

        corpus = TrainingCorpus(
            dataset_name=root.name
        )

        #
        # Every subdirectory represents one dataset.
        #

        dataset_directories = sorted(

            directory

            for directory in root.iterdir()

            if directory.is_dir()

        )

        for directory in dataset_directories:

            try:

                dataset_type = DatasetType(
                    directory.name.lower()
                )

            except ValueError:

                print(
                    f"Skipping unknown dataset: "
                    f"{directory.name}"
                )

                continue

            print(
                f"Loading dataset: "
                f"{directory.name}"
            )

            try:
                converter = ConverterFactory.create(
                    dataset_type
                )
            except ValueError:
                print(
                    f"Skipping unsupported dataset: {directory.name}"
                )
                continue

            partial_corpus = converter.convert(
                str(directory)
            )

            corpus.documents.extend(
                partial_corpus.documents
            )

            corpus.conversations.extend(
                partial_corpus.conversations
            )

        print()

        print("=" * 60)
        print("Dataset Builder")
        print("-" * 60)

        print(
            f"Dataset        : {corpus.dataset_name}"
        )

        print(
            f"Documents      : "
            f"{len(corpus.documents):,}"
        )

        print(
            f"Conversations  : "
            f"{len(corpus.conversations):,}"
        )

        print("=" * 60)

        print()

        return corpus