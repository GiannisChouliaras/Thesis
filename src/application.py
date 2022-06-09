import argparse
from typing import Tuple, List

import torch
import torchvision.models.vgg

from functions import (
    cosine_similarity_scores_of_layers,
    array_to_tensor,
    get_report_of_channels,
)
from opencv_functions import draw_squares, open_and_crop_images
from neural_network import NeuralNetwork
from vgg16 import populate_images_with_features, load_vgg16
from classes import Cases, Category, HyperParameters as HP
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
    return round(svm_before, 1), round(svm_after, 1)


def main(case_num: int) -> None:
    case = Cases(case_num)

    save_cropped_image_before = case.cropped_path(category=Category.BEFORE)
    save_cropped_image_after = case.cropped_path(category=Category.AFTER)

    before, extract_before, before_images_for_color = open_and_crop_images(
        path=case.before_path, desktop_path=save_cropped_image_before
    )
    after, extract_after, after_images_for_color = open_and_crop_images(
        path=case.after_path, desktop_path=save_cropped_image_after
    )

    model: torchvision.models.vgg.VGG = load_vgg16()

    populate_images_with_features(model=model, lst=before)
    populate_images_with_features(model=model, lst=after)

    before_results: List[float] = cosine_similarity_scores_of_layers(before)
    channels_before: float = get_report_of_channels(images=before_images_for_color)
    after_results: List[float] = cosine_similarity_scores_of_layers(after)
    channels_after: float = get_report_of_channels(images=after_images_for_color)

    # add the colour difference of before and after
    before_results.append(channels_before)
    after_results.append(channels_after)

    before_results: torch.Tensor = array_to_tensor(before_results)
    after_results: torch.Tensor = array_to_tensor(after_results)

    result_before, result_after = network(before=before_results, after=after_results)

    # svm
    # regressor = initialize_svr(kernel='rbf')
    # result_before = regressor.predict([before_results])[0]
    # result_after = regressor.predict([after_results])[0]

    draw_squares(case.before_path, extract_before, round(result_before, 1))
    draw_squares(case.after_path, extract_after, round(result_after, 1))


if __name__ == "__main__":
    main(case_num=args.case)
