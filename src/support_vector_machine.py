import numpy as np
import pandas as pd
import pingouin as pg
from sklearn import svm
from typing import List
from sklearn.metrics import explained_variance_score as var, r2_score as r2, mean_squared_error as mse


def convert_colour_range(predictions: np.ndarray, targets: List[int]) -> None:
    predictions = list(predictions)

    convert_range = lambda l: l*10
    predictions = [round(convert_range(entry), 1) for entry in predictions]

    def check_number(number: float) -> float:
        if number > 4.0:
            return 4.0

        if number < 1.0:
            return 1.0

        return number

    predictions = [check_number(prediction) for prediction in predictions]

    df = pd.DataFrame({'Q1': targets,
                       'Q2': predictions})
    print(predictions)
    print(f"Cronbach's alpha is: {pg.cronbach_alpha(data=df)}")


def initialize_svr(kernel='rbf') -> svm.SVR:
    """Using the kernel @param kernel, initialize and @return the regressor."""

    data = np.load("../data/arrays/data.npy")
    target = list(np.load("../data/arrays/targets.npy"))

    # transform each entry from np.array to list
    transform = lambda l: list(l)
    data = [transform(entry) for entry in data]

    # initialize svm_regressor -> regressor
    clf = svm.SVR(kernel=kernel)

    # fit data and return the regressor
    clf.fit(data, target)

    print(f'Coefficient of determination of the prediction: {clf.score(data, target):.3f}')
    return clf


def main() -> None:

    data = np.load("../data/arrays/data.npy")
    target = list(np.load("../data/arrays/targets.npy"))

    regressor = initialize_svr(kernel='rbf')

    # transform each entry from np.array to list
    transform = lambda l: list(l)
    data = [transform(entry) for entry in data]

    predicted = list()

    for index, image in enumerate(data):
        res = regressor.predict([data[index]])[0]
        predicted.append(round(res, 1))

    df = pd.DataFrame({'Q1': target,
                       'Q2': predicted})

    print(f"Cronbach's alpha is: {pg.cronbach_alpha(data=df)}")

    print(f'var score: \t{var(target, predicted):.3f}')
    print(f'R2 score: \t{r2(target, predicted):.3f}')
    print(f'MSE score: \t{mse(target, predicted):.3f}')

    # print(f'Coefficient of determination of the prediction: {regressor.score(data, target):.3f}')


if __name__ == '__main__':
    main()

