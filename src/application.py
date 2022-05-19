import argparse
from typing import Tuple, List

import torch
import torchvision.models.vgg
import numpy as np

from functions import (
    cosine_similarity_scores_of_layers,
    convert_rgb_to_lab,
    global_histogram,
    calculate_log_euclidean_distances,
    array_to_tensor,
)
from opencv_functions import draw_squares, open_and_crop_images
from neural_network import NeuralNetwork
from vgg16 import populate_images_with_features, load_vgg16
from classes import Cases, HyperParameters as HP
from support_vector_machine import initialize_svr

parser = argparse.ArgumentParser(description="Evaluating AK treatment")
parser.add_argument(
    "--case",
    metavar="case_num",
    type=int,
    help="Enter the number of the case",
    required=True,
)
args = parser.parse_args()


def network(before: torch.Tensor, after: torch.Tensor) -> Tuple[float, float]:
    """Initialize and load the pretrained neural network and
    @return the scores for @param before and @param after lists.
    """
    net = NeuralNetwork(
        inFeats=HP.INPUT.value,
        outFeats=HP.OUTPUT.value,
        fHidden=HP.FIRST_HIDDEN.value,
        sHidden=HP.SECOND_HIDDEN.value,
    )
    net.load_state_dict(torch.load("../models/net_reg.pt"))
    net.eval()

    prediction_before = net(before).item()
    prediction_after = net(after).item()
    return round(prediction_before, 1), round(prediction_after, 1)


def svm_regressor(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    """Initialize svm as regressor and @return the scores for the @param before and @param after lists."""
    svm_reg = initialize_svr("linear")
    svm_before = svm_reg.predict([list(before)])[0]
    svm_after = svm_reg.predict([list(after)])[0]
    return round(svm_before, 2), round(svm_after, 2)


def main(case_num: int) -> None:
    case = Cases(case_num)

    before, extract_before, before_images_for_color = open_and_crop_images(
        case.before_path
    )
    after, extract_after, after_images_for_color = open_and_crop_images(case.after_path)

    model: torchvision.models.vgg.VGG = load_vgg16()

    populate_images_with_features(model=model, lst=before)
    populate_images_with_features(model=model, lst=after)

    # find the cosine similarities and convert them to tensor
    before_results: List[float] = cosine_similarity_scores_of_layers(before)
    after_results: List[float] = cosine_similarity_scores_of_layers(after)

    # Convert the images into lab space and calculate the histogram
    hist_lab_before: List[np.ndarray] = [
        global_histogram(img=convert_rgb_to_lab(img)) for img in before_images_for_color
    ]
    hist_lab_after: List[np.ndarray] = [
        global_histogram(img=convert_rgb_to_lab(img)) for img in after_images_for_color
    ]

    led_before: float = calculate_log_euclidean_distances(hist_lab_before)
    led_after: float = calculate_log_euclidean_distances(hist_lab_after)

    before_results.append(led_before)
    after_results.append(led_after)

    before_results: torch.Tensor = array_to_tensor(before_results)
    after_results: torch.Tensor = array_to_tensor(after_results)

    result_before, result_after = network(before=before_results, after=after_results)

    draw_squares(case.before_path, extract_before, result_before)
    draw_squares(case.after_path, extract_after, result_after)


if __name__ == "__main__":
    main(case_num=args.case)
