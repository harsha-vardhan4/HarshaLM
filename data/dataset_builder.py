from data.loader import TextDatasetLoader
from data.conversation_formatter import (
    ConversationFormatter,
)
from data.parsers.parser_factory import (
    ParserFactory,
    DatasetType,
)


class DatasetBuilder:
    """
    Builds the final training corpus.
    """

    def __init__(self):

        self.loader = TextDatasetLoader()

        self.formatter = (
            ConversationFormatter()
        )

    def build(
        self,
        path: str,
        dataset_type: DatasetType,
    ) -> str:
        """
        Loads a dataset and converts it into
        HarshaLM's canonical training format.
        """

        raw_conversations = (
            self.loader.load(path)
        )

        parser = ParserFactory.create(
            dataset_type
        )

        formatted_conversations = []

        for raw_conversation in raw_conversations:

            parsed = parser.parse(raw_conversation)

            if isinstance(parsed, str):

                formatted = parsed

            else:

                formatted = self.formatter.format(parsed)

            formatted_conversations.append(formatted)

        print()

        print("=" * 60)
        print("Dataset Builder")
        print("-" * 60)
        print(
            f"Conversations : "
            f"{len(raw_conversations):,}"
        )
        
        print("=" * 60)

        print()

        return formatted_conversations