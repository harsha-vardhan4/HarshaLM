from dataclasses import dataclass, field

from data.conversations.formatted_conversation import (
    FormattedConversation,
)

from data.document.formatted_document import (
    FormattedDocument,
)


@dataclass
class TrainingCorpus:
    """
    Canonical dataset representation used
    throughout HarshaLM.

    Supports:

    - Conversation datasets
    - Plain text datasets
    """


    dataset_name: str


    conversations: list[
        FormattedConversation
    ] = field(
        default_factory=list
    )


    documents: list[
        FormattedDocument
    ] = field(
        default_factory=list
    )


    def __len__(self):

        return (
            len(self.conversations)
            +
            len(self.documents)
        )