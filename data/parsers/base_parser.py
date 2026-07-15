from abc import ABC, abstractmethod


class BaseParser(ABC):
    """
    Base class for all dataset parsers.
    """

    @abstractmethod
    def parse(
        self,
        text: str,
    ) -> list[tuple[str, str]]:
        """
        Converts raw dataset text into
        a structured conversation.
        """
        pass