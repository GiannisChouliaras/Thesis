import numpy as np
import torch

from func import array_to_tensor
from func import convert_number
from net import Net


def main() -> None:
    # load model
    model = Net(inFeats=3, outFeats=1, fHidden=10, sHidden=5)
    model.load_state_dict(torch.load("../models/net.pt"))
    model.eval()

    actual = 2
    data = np.array([0.604, 0.807, 0.898])
    data = array_to_tensor(data)

    prediction = round(model(data).item(), 3)
    print(f"actual is {actual} and prediction is :", end=" ")
    print(convert_number(prediction, range1=(0, 1), range2=(1, 8)))


if __name__ == "__main__":
    main()
