import numpy as np
import torch

from net import Net
from sklearn.metrics import explained_variance_score as var, r2_score as r2, mean_squared_error as mse


def main() -> None:
    # load model
    model = Net(inFeats=3, outFeats=1, fHidden=100, sHidden=100)
    model.load_state_dict(torch.load("../models/net.pt"))
    model.eval()

    data = np.load("../data/arrays/data.npy")
    target = list(np.load("../data/arrays/targets.npy"))
    data = torch.tensor(data, dtype=torch.float32)

    predicted = []
    for index, item in enumerate(data):
        prediction = model(item).item()
        predicted.append(round(prediction, 1))

    print('pred: ', predicted, sep='\t')
    print('targ: ', target, sep='\t')

    print(f'var score: \t{var(target, predicted):.3f}')
    print(f'R2 score: \t{r2(target, predicted):.3f}')
    print(f'MSE score: \t{mse(target, predicted):.3f}')


if __name__ == "__main__":
    main()
