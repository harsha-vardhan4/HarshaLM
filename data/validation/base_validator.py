from abc import ABC, abstractmethod


class BaseValidator(ABC):
    """
    Base class for validators.
    """

    @abstractmethod
    def validate(
        self,
        item,
    ) -> bool:
        """
        Returns True if the item is valid.
        """
        ...