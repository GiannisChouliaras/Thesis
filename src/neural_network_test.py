import numpy as np
import torch

from neural_network import NeuralNetwork
from classes import HyperParameters as HP
from sklearn.metrics import (
    explained_variance_score as var,
    r2_score as r2,
    mean_squared_error as mse,
)


def main() -> None:

    # load model
    model = NeuralNetwork(
        inFeats=HP.INPUT.value,
        outFeats=HP.OUTPUT.value,
        fHidden=HP.FIRST_HIDDEN.value,
        sHidden=HP.SECOND_HIDDEN.value,
    )
    model.load_state_dict(torch.load("../models/net_reg.pt"))
    model.eval()

    data = np.load("../data/arrays/data.npy")
    target = list(np.load("../data/arrays/targets.npy"))
    data = torch.tensor(data, dtype=torch.float32)

    predicted = []
    for index, item in enumerate(data):
        prediction = model(item).item()
        predicted.append(int(round(prediction, 0)))

    print("prediction: ", predicted, sep="\t")
    print("target: ", target, sep="\t\t")

    print(f"var score: \t{var(target, predicted):.3f}")
    print(f"R2 score: \t{r2(target, predicted):.3f}")
    print(f"MSE score: \t{mse(target, predicted):.3f}")


if __name__ == "__main__":
    main()
