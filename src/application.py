import argparse
from typing import Tuple, Any

import torch

from func import array_to_tensor, convert_number, results
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


def network(before: torch.Tensor, after: torch.Tensor) -> Tuple[int, int]:
    # load nn model
    net = Net(inFeats=3, outFeats=1, fHidden=100, sHidden=50)
    net.load_state_dict(torch.load("../models/net.pt"))
    net.eval()

    before_prediction = round(net(before).item(), 3)
    after_prediction = round(net(after).item(), 3)
    res_before = convert_number(before_prediction, range1=(0, 1), range2=(1, 8)),
    res_after = convert_number(after_prediction, range1=(0, 1), range2=(1, 8)),
    # TODO check why convert_number returns a tuple
    return res_before[0], res_after[0]


def svm(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    svm_reg = init_svm("rbf")
    svm_before = svm_reg.predict([list(before)])[0]
    svm_after = svm_reg.predict([list(after)])[0]
    return round(svm_before, 1), round(svm_after, 1)


def main(case_num: int) -> None:
    case = Cases(case_num)

    before, extract_before = get_images(case.before_path)
    after, extract_after = get_images(case.after_path)

    model = load_vgg16()

    get_features_from_lst(model=model, lst=before)
    get_features_from_lst(model=model, lst=after)

    before_results = array_to_tensor(results(before))
    after_results = array_to_tensor(results(after))

    result_before, result_after = network(before=before_results, after=after_results)

    draw_squares(case.before_path, extract_before, result_before)
    draw_squares(case.after_path, extract_after, result_after)


if __name__ == "__main__":
    main(case_num=args.case)
