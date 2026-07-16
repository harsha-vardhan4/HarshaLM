from dataclasses import dataclass


@dataclass
class FormattedDocument:
    """
    Represents a plain-text training document.

    Used for books, Wikipedia, TinyStories,
    articles, blogs, research papers, etc.

    Unlike conversations, every token is
    considered learnable.
    """

    text: str