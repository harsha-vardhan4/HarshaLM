from data.conversations.formatted_conversation import (
    FormattedConversation,
)

from data.document.formatted_document import (
    FormattedDocument,
)

from data.training_sample import (
    TrainingSample,
)

from tokenizer.tokenizer import (
    create_tokenizer,
)

from data.conversations.conversation_loss_mask_builder import (
    ConversationLossMaskBuilder,
)

from data.document.document_loss_mask_builder import (
    DocumentLossMaskBuilder,
)

from data.training_corpus import (
    TrainingCorpus,
)


class DataPreparation:
    """
    Converts formatted documents into TrainingSamples.

    Supports both:

    - Conversation datasets
    - Plain-text documents
    """

    def __init__(self):

        self.tokenizer = create_tokenizer()

        self.conversation_builder = (
            ConversationLossMaskBuilder()
        )

        self.document_builder = (
            DocumentLossMaskBuilder()
        )

    def prepare(
        self,
        corpus: TrainingCorpus,
    ) -> list[TrainingSample]:
        """
        Converts formatted documents into
        TrainingSamples.
        """

        training_samples = []

        total_characters = 0
        total_tokens = 0

        items = (
            corpus.conversations
            +
            corpus.documents
        )

        for document in items:

            #
            # Conversation
            #

            if isinstance(
                document,
                FormattedConversation,
            ):

                (
                    input_ids,
                    labels,
                    loss_mask,
                ) = self.conversation_builder.build(
                    document
                )

            #
            # Plain Text
            #

            elif isinstance(
                document,
                FormattedDocument,
            ):

                (
                    input_ids,
                    labels,
                    loss_mask,
                ) = self.document_builder.build(
                    document
                )

            #
            # Unknown type
            #

            else:

                raise TypeError(
                    "Unsupported document type: "
                    f"{type(document)}"
                )

            training_samples.append(

                TrainingSample(
                    input_ids=input_ids,
                    labels=labels,
                    loss_mask=loss_mask,
                    conversation_length=len(
                        input_ids
                    ),
                )

            )

            total_characters += self._count_characters(document)

            total_tokens += len(
                input_ids
            )

        print(
            f"Conversations : {len(corpus.conversations):,}"
        )

        print(
            f"Documents     : {len(corpus.documents):,}"
        )

        print(
            f"Characters    : {total_characters:,}"
        )

        print(
            f"Tokens        : {total_tokens:,}"
        )

        if items:

            print(
                "Avg Tokens    : "
                f"{total_tokens / len(items):.2f}"
            )

        return training_samples
    
    def _count_characters(self, item):

        if isinstance(item, FormattedDocument):
            return len(item.text)

        if isinstance(item, FormattedConversation):
            return sum(
                len(message.text)
                for message in item.messages
            )

        return 0