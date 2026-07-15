from enum import Enum

from data.parsers.plain_text_parser import PlainTextParser
from data.parsers.conversation_parser import ConversationParser


class DatasetType(Enum):

    PLAIN_TEXT = "plain_text"

    CONVERSATION = "conversation"

    SHAREGPT = "sharegpt"

    ALPACA = "alpaca"

    JSON = "json"


class ParserFactory:
    """
    Creates parsers based on dataset type.
    """

    @staticmethod
    def create(
        dataset_type: DatasetType,
    ):

        if dataset_type == DatasetType.PLAIN_TEXT:

            return PlainTextParser()
        
        if dataset_type == DatasetType.CONVERSATION:

            return ConversationParser()

        raise ValueError(
            f"Unsupported dataset type: {dataset_type}"
        )