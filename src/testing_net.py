import numpy as np
import torch
from func import convert_number, array_to_tensor
from net import Net


def main() -> None:
    # load model
    model = Net(inFeats=3, outFeats=1, fHidden=10, sHidden=5)
    model.load_state_dict(torch.load("../models/net.pth"))
    model.eval()

    F1 = 0.3690
    F2 = 0.5530
    F3 = 0.7310
    actual = 0.286

    data = np.array([F1, F2, F3])
    data = array_to_tensor(data)

    prediction = model(data).item()
    print(round(prediction, 3))
    print(f"Result is: {convert_number(number=round(prediction, 3), range1=(0, 1), range2=(1, 8))}")
    print(f"actual is: {convert_number(number=actual, range1=(0, 1), range2=(1, 8))}")


if __name__ == "__main__":
    main()
