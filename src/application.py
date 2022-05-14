import argparse
from typing import Tuple

import torch

from func import (
    array_to_tensor,
    results,
    calculate_similarities,
)
from ImageFunctions import draw_squares, get_images
from net import Net
from Vgg16 import get_features_from_lst, load_vgg16
from config import Cases
from SVM import init_svm

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
    net = Net(inFeats=3, outFeats=1, fHidden=100, sHidden=100)
    net.load_state_dict(torch.load("../models/net.pt"))
    net.eval()

    prediction_before = net(before).item()
    prediction_after = net(after).item()
    return round(prediction_before, 1), round(prediction_after, 1)


def svm(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    svm_reg = init_svm("rbf")
    svm_before = svm_reg.predict([list(before)])[0]
    svm_after = svm_reg.predict([list(after)])[0]
    return round(svm_before, 2), round(svm_after, 2)


def main(case_num: int) -> None:
    case = Cases(case_num)

    before, extract_before, before_images_for_color = get_images(case.before_path)
    after, extract_after, after_images_for_color = get_images(case.after_path)

    model = load_vgg16()

    get_features_from_lst(model=model, lst=before)
    get_features_from_lst(model=model, lst=after)

    layer28_bef = calculate_similarities(
        before[0].layer28.flatten(),
        before[1].layer28.flatten(),
        before[2].layer28.flatten(),
    )

    layer28_aft = calculate_similarities(
        after[0].layer28.flatten(),
        after[1].layer28.flatten(),
        after[2].layer28.flatten(),
    )

    print("layer 28")
    print(f"before: {layer28_bef:.3f}")
    print(f"after: {layer28_aft:.3f}")

    before_results = array_to_tensor(results(before))
    after_results = array_to_tensor(results(after))

    result_before, result_after = network(before=before_results, after=after_results)

    draw_squares(case.before_path, extract_before, result_before)
    draw_squares(case.after_path, extract_after, result_after)


if __name__ == "__main__":
    main(case_num=args.case)
