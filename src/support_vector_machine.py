import numpy as np
from sklearn import svm
from sklearn.metrics import explained_variance_score as var, r2_score as r2, mean_squared_error as mse


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
    return clf


def main(case_number: int) -> None:

    data = np.load("../data/arrays/data.npy")
    target = np.load("../data/arrays/targets.npy")
    
    regressor = initialize_svr(kernel='linear')

    predicted = list()

    for index, image in enumerate(data):
        res = regressor.predict([data[index]])[0]
        predicted.append(round(res, 2))

    print(predicted[case_number], target[case_number], sep='\n')
    print(f'var score: \t{var(target, predicted):.3f}')
    print(f'R2 score: \t{r2(target, predicted):.3f}')
    print(f'MSE score: \t{mse(target, predicted):.3f}')


if __name__ == '__main__':
    main(case_number=18)
