import torch
import numpy as np
import cv2 as cv

from typing import List

from numpy import ndarray

from classes import Image
from skimage.color import yiq2rgb
from statistics import mean


def array_to_tensor(array: List[float]) -> torch.tensor:
    """@return the conversion of an array to tensor."""
    return torch.tensor(array, dtype=torch.float32)


def cosine_similarity(A: ndarray, B: ndarray) -> float:
    """@return the cosine similarity between two numpy arrays"""
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def euclidean_distance(p, q):
    dist = np.sqrt(np.sum(np.square(p - q)))
    return dist


def log_euclidean_distance(P: ndarray, Q: ndarray) -> float:
    """@return the log euclidean distance"""
    pl = np.log2(P.astype(float) + 1)
    ql = np.log2(Q.astype(float) + 1)
    dist = np.sqrt(np.sum(np.square(pl - ql)))
    return dist


def get_means_of_channels(image: ndarray) -> ndarray:
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return np.array([np.mean(R), np.mean(G), np.mean(B)])


def get_report_of_channels(images: List[ndarray]) -> float:
    """@return the average euclidean distance of the images in @param images in YIQ colour space"""
    images = [yiq2rgb(img) for img in images]
    wound = get_means_of_channels(images[0])
    n_sk1 = get_means_of_channels(images[1])
    n_sk2 = get_means_of_channels(images[2])
    average = mean([euclidean_distance(wound, n_sk2), euclidean_distance(wound, n_sk1)])
    return average


def convert_rgb_to_lab(image: ndarray) -> ndarray:
    """@return the normalized and converted image from RGB to LAB colour space."""
    return cv.cvtColor(image.astype(np.float32) / 255, cv.COLOR_RGB2Lab)


def calculate_similarities(wound: ndarray, first: ndarray, second: ndarray) -> float:
    """@return the average cosine similarity between the two images and the wound."""
    c1 = cosine_similarity(wound, first)
    c2 = cosine_similarity(wound, second)
    return mean([c1, c2])


def cosine_similarity_scores_of_layers(
    list_of_images_instances: List[Image],
) -> List[float]:
    """Execute the function calculate_similarities in a List that contains Image objects."""
    early = calculate_similarities(
        list_of_images_instances[0].early.flatten(),
        list_of_images_instances[1].early.flatten(),
        list_of_images_instances[2].early.flatten(),
    )

    # late = calculate_similarities(
    #     list_of_images_instances[0].late.flatten(),
    #     list_of_images_instances[1].late.flatten(),
    #     list_of_images_instances[2].late.flatten(),
    # )

    fully_connected = calculate_similarities(
        list_of_images_instances[0].fully_connected.flatten(),
        list_of_images_instances[1].fully_connected.flatten(),
        list_of_images_instances[2].fully_connected.flatten(),
    )
    return [early, fully_connected]