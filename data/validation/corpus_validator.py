from data.training_corpus import (
    TrainingCorpus,
)

from data.validation.validation_report import (
    ValidationReport,
)

from data.validation.document_validator import (
    DocumentValidator,
)

from data.validation.conversation_validator import (
    ConversationValidator,
)


class CorpusValidator:
    """
    Validates an entire TrainingCorpus.

    Removes invalid and duplicate items while
    collecting validation statistics.
    """

    def __init__(self):

        self.document_validator = (
            DocumentValidator()
        )

        self.conversation_validator = (
            ConversationValidator()
        )

    def validate(
        self,
        corpus: TrainingCorpus,
    ) -> tuple[
        TrainingCorpus,
        ValidationReport,
    ]:

        report = ValidationReport()

        validated = TrainingCorpus(
            dataset_name=corpus.dataset_name,
        )

        #
        # Duplicate detection
        #

        seen_documents = set()

        seen_conversations = set()

        #
        # ----------------------------
        # Documents
        # ----------------------------
        #

        for document in corpus.documents:

            report.total_documents += 1

            #
            # Empty document
            #

            if not document.text.strip():

                report.empty_documents += 1

                continue

            #
            # Validation
            #

            if not self.document_validator.validate(
                document
            ):

                report.invalid_documents += 1

                continue

            #
            # Duplicate detection
            #

            key = hash(
                document.text
            )

            if key in seen_documents:

                report.duplicate_documents += 1

                continue

            seen_documents.add(key)

            validated.documents.append(
                document
            )

            report.valid_documents += 1

        #
        # ----------------------------
        # Conversations
        # ----------------------------
        #

        for conversation in corpus.conversations:

            report.total_conversations += 1

            if not conversation.messages:

                report.empty_conversations += 1

                continue

            if not self.conversation_validator.validate(
                conversation
            ):

                report.invalid_conversations += 1

                continue

            key = hash(
                conversation.text
            )

            if key in seen_conversations:

                report.duplicate_conversations += 1

                continue

            seen_conversations.add(
                key
            )

            validated.conversations.append(
                conversation
            )

            report.valid_conversations += 1

        self._print_report(
            report
        )

        return (
            validated,
            report,
        )

    def _print_report(
        self,
        report: ValidationReport,
    ):

        print()

        print("=" * 60)
        print("Corpus Validator")
        print("-" * 60)

        print(
            f"Documents                : {report.total_documents:,}"
        )

        print(
            f"Conversations            : {report.total_conversations:,}"
        )

        print()

        print(
            f"Valid Documents          : {report.valid_documents:,}"
        )

        print(
            f"Valid Conversations      : {report.valid_conversations:,}"
        )

        print()

        print(
            f"Duplicate Documents      : {report.duplicate_documents:,}"
        )

        print(
            f"Duplicate Conversations  : {report.duplicate_conversations:,}"
        )

        print()

        print(
            f"Empty Documents          : {report.empty_documents:,}"
        )

        print(
            f"Empty Conversations      : {report.empty_conversations:,}"
        )

        print()

        print(
            f"Invalid Documents        : {report.invalid_documents:,}"
        )

        print(
            f"Invalid Conversations    : {report.invalid_conversations:,}"
        )

        print("=" * 60)

        print()