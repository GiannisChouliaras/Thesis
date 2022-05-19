import numpy as np
from enum import IntEnum, Enum


class Cases:
    def __init__(self, case: int) -> None:
        """Class that contains the paths of the given case number."""
        self.case = case

    @property
    def before_path(self) -> str:
        return f"../data/raw/CASE{self.case}/paired/before.jpg"

    @property
    def after_path(self) -> str:
        return f"../data/raw/CASE{self.case}/paired/after.jpg"


class Image:
    def __init__(self, array: np.ndarray):
        """Class representing an image that has the array and the desired cnn features."""
        self.array = array
        self.early = None
        self.late = None
        self.fully_connected = None


class Layer(IntEnum):
    """Enum Class that contains the numbers of the layers that will be used for vgg 16."""
    FULLY_CONNECTED = 0
    EARLY = 5
    LATE = 28


class HyperParameters(Enum):
    """Enum class for the hyperparameters of the network."""
    LR = 1e-2
    ITERATIONS = 500
    INPUT = 3
    OUTPUT = 1
    FIRST_HIDDEN = 100
    SECOND_HIDDEN = 50
    DROPOUT = 0.20
