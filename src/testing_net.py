import numpy as np
import torch

from func import array_to_tensor
from func import convert_number
# from net_crossentropy import SoftNet
from net import Net


def main() -> None:
    # load model
    # model = SoftNet(inputs=3, outputs=8, hidden_size=10)
    model = Net(inFeats=3, outFeats=1, fHidden=10, sHidden=5)
    model.load_state_dict(torch.load("../models/net.pt"))
    model.eval()

    actual = 8
    data = np.array([0.3240, 0.3832, 0.5903])
    data = array_to_tensor(data)

    prediction = model(data)
    print(f"actual is {actual} and prediction is :", end=" ")
    print(convert_number(prediction, range1=(0, 1), range2=(1, 8)))
    # print(torch.argmax(prediction).item() + 1)


if __name__ == "__main__":
    main()
