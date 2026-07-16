from dataclasses import dataclass


@dataclass
class ValidationReport:
    """
    Statistics produced by CorpusValidator.
    """

    total_documents: int = 0

    total_conversations: int = 0

    valid_documents: int = 0

    valid_conversations: int = 0

    duplicate_documents: int = 0

    duplicate_conversations: int = 0

    empty_documents: int = 0

    empty_conversations: int = 0

    invalid_documents: int = 0

    invalid_conversations: int = 0