import numpy as np
from sklearn import svm
from sklearn.metrics import explained_variance_score as var, r2_score as r2, mean_squared_error as mse
from sklearn.preprocessing import StandardScaler


def init_svm(ker='rbf') -> svm.SVR:
    # load data
    data = np.load("../data/arrays/data.npy")
    target = list(np.load("../data/arrays/targets.npy"))
    transform = lambda l: list(l)
    data = [transform(entry) for entry in data]

    # init svc
    clf = svm.SVR(kernel=ker)
    # fit
    clf.fit(data, target)

    return clf


def main() -> None:
    data = np.load("../data/arrays/data.npy")
    target = np.load("../data/arrays/targets.npy")

    transform = lambda l: list(l)
    data = [transform(entry) for entry in data]

    clf = svm.SVR(kernel='rbf')
    clf.fit(data, target)

    predicted = list()

    for index, image in enumerate(data):
        res = clf.predict([data[index]])[0]
        predicted.append(round(res, 2))

    number = 4
    print(predicted[number], target[number], sep='\n')
    print(f'var score: \t{var(target, predicted):.3f}')
    print(f'R2 score: \t{r2(target, predicted):.3f}')
    print(f'MSE score: \t{mse(target, predicted):.3f}')


if __name__ == '__main__':
    main()
