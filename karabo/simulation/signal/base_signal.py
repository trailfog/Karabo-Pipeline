"""Base signal class."""
from abc import ABC, abstractmethod
from typing import Union

from karabo.simulation.signal.typing import Image2D, Image2DOriented


# pylint: disable=too-few-public-methods
class BaseSignal2D(ABC):
    """Base signal class."""

    @abstractmethod
    def simulate(self) -> list[Union[Image2D, Image2DOriented]]:
        """Simulate a signal to get a 2D image output."""
