from abc import ABC, abstractmethod
from typing import Any


class DataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.

    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """

    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> "DataLoader":
        """Return self as iterator."""
        return self

    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the iterator to the beginning."""
        pass
