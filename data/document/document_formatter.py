from data.document.formatted_document import (
    FormattedDocument,
)

from utils.special_tokens import (
    END_OF_TEXT_TOKEN,
)


class DocumentFormatter:
    """
    Formats plain text into HarshaLM's
    canonical document format.

    Documents use GPT-2 style EOS:
    
    <BOS>
    text
    <|endoftext|>
    """

    def format(
        self,
        text: str,
    ) -> FormattedDocument:

        text = text.strip()

        if not text.endswith(
            END_OF_TEXT_TOKEN
        ):
            text += END_OF_TEXT_TOKEN

        return FormattedDocument(
            text=text
        )