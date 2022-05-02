import argparse
from typing import List

import torch

from func import array_to_tensor, convert_number, results
from ImageFunctions import Image, get_images
from net import Net
from Vgg16 import get_features_from_lst, load_vgg16
from config import Cases

parser = argparse.ArgumentParser(description="Evaluating AK treatment")
parser.add_argument(
    "--case",
    metavar="case_num",
    type=int,
    help="Enter the number of the case",
    required=True,
)
args = parser.parse_args()


def main(case_num: int) -> None:
    case = Cases(case_num)

    before: List[Image] = get_images(case.before_path)
    after: List[Image] = get_images(case.after_path)

    model = load_vgg16()

    get_features_from_lst(model=model, lst=before)
    get_features_from_lst(model=model, lst=after)

    before_results = array_to_tensor(results(before))
    after_results = array_to_tensor(results(after))

    # ********************************************
    #       Use Net to predict the score
    # ********************************************

    # load nn model
    net = Net(inFeats=3, outFeats=1, fHidden=100, sHidden=50)
    net.load_state_dict(torch.load("../models/net.pt"))
    net.eval()
    # svm = init_svm("rbf")

    # predict
    before_prediction = round(net(before_results).item(), 3)
    after_prediction = round(net(after_results).item(), 3)

    print(
        "Sigmoid: Before: ",
        convert_number(before_prediction, range1=(0, 1), range2=(1, 8)),
        end=" ------- ",
    )

    print("After: ", convert_number(after_prediction, range1=(0, 1), range2=(1, 8)))

    # svm_before = svm.predict([list(before_results)])[0]
    # svm_after = svm.predict([list(after_results)])[0]
    #
    # print(
    #     "Sigmoid: Before: ",
    #     convert_number(before_prediction, range1=(0, 1), range2=(1, 8)),
    #     end=" ------- ",
    # )
    #
    # print("After: ", convert_number(after_prediction, range1=(0, 1), range2=(1, 8)))
    #
    # print(f"svm: before: {round(svm_before, 1)} ----- after: {round(svm_after, 1)}")


if __name__ == "__main__":
    main(case_num=args.case)
