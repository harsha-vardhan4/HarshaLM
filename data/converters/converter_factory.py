from data.converters.documents.plain_text_converter import (
    PlainTextConverter,
)

from data.converters.conversations.daily_dialog_converter import (
    DailyDialogConverter,
)

from data.converters.conversations.sharegpt_converter import (
    ShareGPTConverter,
)

from data.converters.conversations.alpaca_converter import (
    AlpacaConverter,
)

from data.parsers.parser_factory import (
    DatasetType,
)


class ConverterFactory:
    """
    Factory responsible for creating dataset converters.
    """

    @staticmethod
    def create(
        dataset_type: DatasetType,
    ):

        if dataset_type == DatasetType.PLAIN_TEXT:

            return PlainTextConverter()

        if dataset_type == DatasetType.DAILY_DIALOG:

            return DailyDialogConverter()

        if dataset_type == DatasetType.SHAREGPT:

            return ShareGPTConverter()

        if dataset_type == DatasetType.ALPACA:

            return AlpacaConverter()

        raise ValueError(
            f"Unsupported dataset type: {dataset_type}"
        )