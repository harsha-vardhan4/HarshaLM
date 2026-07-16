from abc import ABC, abstractmethod


class BaseConverter(ABC):
    """
    Base class for all dataset converters.

    Every converter must return a list of
    formatted conversations.
    """

    @abstractmethod
    def convert(
        self,
        path: str,
    ) -> list[str]:
        """
        Converts a dataset into HarshaLM's
        canonical conversation format.
        """
        pass