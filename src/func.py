import torch
import numpy as np
from typing import Tuple, Callable


def check_ranges(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if len(kwargs.get('range1')) != 2 or len(kwargs.get('range2')) != 2:
            raise Exception('Ranges should have length of 2')
        number = func(*args, **kwargs)
        return number
    return wrapper


@check_ranges
def convert_number(number: float, range1: Tuple, range2: Tuple) -> int:
    (a, b), (c, d) = range1, range2
    return d if number <= 0.1 else (c + d) - round(c + (d - c / b - a) * (number - a))


def array_to_tensor(array: np.array) -> torch.tensor:
    return torch.tensor(array, dtype=torch.float32)
