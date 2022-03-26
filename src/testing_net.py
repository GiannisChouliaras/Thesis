import numpy as np
import torch

from func import array_to_tensor
from net_crossentropy import SoftNet


def main() -> None:
    # load model
    model = SoftNet(inputs=3, outputs=8, hidden_size=10)
    model.load_state_dict(torch.load("../models/net_ce.pt"))
    model.eval()

    F1 = 0.2970
    F2 = 0.5880
    F3 = 0.5270
    actual = 7

    data = np.array([F1, F2, F3])
    data = array_to_tensor(data)

    prediction = model(data)
    print(f"actual is {actual} and prediction is :", end=" ")
    print(torch.argmax(prediction).item() + 1)


if __name__ == "__main__":
    main()
