import numpy as np
import pandas as pd
import pingouin as pg
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
        predicted.append(round(prediction, 1))

    df = pd.DataFrame({'Q1': target,
                       'Q2': predicted})

    print(f"Cronbach's alpha is: {pg.cronbach_alpha(data=df)}")

    print(f"var score: \t{var(target, predicted):.3f}")
    print(f"R2 score: \t{r2(target, predicted):.3f}")
    print(f"MSE score: \t{mse(target, predicted):.3f}")


if __name__ == "__main__":
    main()
