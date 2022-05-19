import torch
import numpy as np
import cv2 as cv

from typing import Tuple, List
from classes import Image
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.transform import resize


# def check_ranges(func: Callable) -> Callable:
#     """Decorator to check the ranges values."""
#
#     def wrapper(*args, **kwargs):
#         if len(kwargs.get("range1")) != 2 or len(kwargs.get("range2")) != 2:
#             raise Exception("Ranges should have length of 2")
#         number = func(*args, **kwargs)
#         return number
#
#     return wrapper
#
#
# @check_ranges
# def convert_number(
#     number: float, range1: Tuple = (0, 1), range2: Tuple = (2, 8)
# ) -> int:
#     (a, b), (c, d) = range1, range2
#     return d if number <= 0.1 else (c + d) - round(c + (d - c / b - a) * (number - a))


# @check_ranges
# def convert_number(number: float, range1: Tuple, range2: Tuple) -> int:
#     """@return the value of the @param number from the @param range1 to the @param range2."""
#     (a, b), (c, d) = range1, range2
#
#     val = (d - c) * (number - a) / (b - a) + c
#     return (d + 1) - round(val, 0)


def array_to_tensor(array: np.array) -> torch.tensor:
    """@return the conversion of an array to tensor."""
    return torch.tensor(array, dtype=torch.float32)


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """@return the cosine similarity between two numpy arrays"""
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def log_euclidean_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """@return the log euclidean distance"""
    pl = np.log2(P.astype(float) + 1)
    ql = np.log2(Q.astype(float) + 1)
    dist = np.sqrt(np.sum(np.square(pl - ql)))
    return dist


def segmentation(img: np.ndarray, n_segments=4, compactness=3) -> np.ndarray:
    """@return the segmentation of the @param img with slic."""
    image_slic = slic(image=img, n_segments=n_segments, compactness=compactness)
    return label2rgb(image_slic, img, kind="avg")


def convert_rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """@return the normalized and converted image from RGB to LAB colour space."""
    return cv.cvtColor(image.astype(np.float32) / 255, cv.COLOR_RGB2Lab)


def resize_image(image: np.ndarray, shape: Tuple[int, int], order: int) -> np.ndarray:
    """@return the resized image given the output shape and the order (=0 is nearest neighbour, 1=bi-linear,
    2=bi-quadratic, 3=bi-cubic etc"""
    return resize(image=image, output_shape=shape, order=order)


def average(lst: List) -> float:
    """@return the average value of the given List"""
    return sum(lst) / len(lst)


def calculate_similarities(
    wound: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    """@return the average cosine similarity between the two images and the wound."""
    c1 = cosine_similarity(wound, first)
    c2 = cosine_similarity(wound, second)
    return average([c1, c2])


def calculate_log_euclidean_distances(lst: List) -> float:
    """@return the average value of the log euclidean distance"""
    led1 = log_euclidean_distance(lst[0], lst[1])
    led2 = log_euclidean_distance(lst[0], lst[2])
    return average([led1, led2])


def cosine_similarity_scores_of_layers(list_of_images_instances: List[Image]) -> List[float]:
    """Execute the function calculate_similarities in a List that contains Image objects."""
    early = calculate_similarities(
        list_of_images_instances[0].early.flatten(),
        list_of_images_instances[1].early.flatten(),
        list_of_images_instances[2].early.flatten(),
    )
    fully_connected = calculate_similarities(
        list_of_images_instances[0].fully_connected.flatten(),
        list_of_images_instances[1].fully_connected.flatten(),
        list_of_images_instances[2].fully_connected.flatten(),
    )
    return [early, fully_connected]


def global_histogram(img, bins=32, norm="value", factor=512) -> np.ndarray:
    """@return the histogram of the @param img using number of @param bins"""
    if len(img.shape) == 2:  # for single channel
        hist, _ = np.histogram(img, bins=bins)

    if len(img.shape) == 3:  # for RGB - LAB - YIQ
        hL, _ = np.histogram(img[:, :, 0], bins=bins)
        hA, _ = np.histogram(img[:, :, 1], bins=bins)
        hB, _ = np.histogram(img[:, :, 2], bins=bins)
        # hist = np.concatenate([hL, hA, hB])
        hist = hL  # only the L from LAB space

    hist = hist.astype(float)

    if norm == "sum":
        hist /= hist.sum() + 1e-4
        return hist

    hist /= hist.sum() + 1e-4
    hist *= factor
    return hist.astype(int)
