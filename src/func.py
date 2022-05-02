import torch
import numpy as np
from typing import Tuple, Callable, List
from ImageFunctions import Image


def check_ranges(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if len(kwargs.get("range1")) != 2 or len(kwargs.get("range2")) != 2:
            raise Exception("Ranges should have length of 2")
        number = func(*args, **kwargs)
        return number

    return wrapper


@check_ranges
def convert_number(number: float, range1: Tuple, range2: Tuple) -> int:
    (a, b), (c, d) = range1, range2
    return d if number <= 0.1 else (c + d) - round(c + (d - c / b - a) * (number - a))


def array_to_tensor(array: np.array) -> torch.tensor:
    return torch.tensor(array, dtype=torch.float32)


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def average(lst: List) -> float:
    return sum(lst) / len(lst)


def calculate_similarities(
    wound: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    c1 = cosine_similarity(wound, first)
    c2 = cosine_similarity(wound, second)
    return average([c1, c2])


def results(lst: List[Image]) -> List:
    layer30 = calculate_similarities(
        lst[0].layer30.flatten(), lst[1].layer30.flatten(), lst[2].layer30.flatten()
    )
    layer11 = calculate_similarities(
        lst[0].layer11.flatten(), lst[1].layer11.flatten(), lst[2].layer11.flatten()
    )
    fully_c = calculate_similarities(
        lst[0].fully_connected.flatten(),
        lst[1].fully_connected.flatten(),
        lst[2].fully_connected.flatten(),
    )
    return [layer30, layer11, fully_c]
