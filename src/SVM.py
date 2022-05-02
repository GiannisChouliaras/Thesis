import numpy as np
from sklearn import svm


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
    target = list(np.load("../data/arrays/targets.npy"))
    transform = lambda l: list(l)
    data = [transform(entry) for entry in data]

    clf = svm.SVR(kernel='rbf')
    clf.fit(data, target)

    print(type(clf))

    test = 38
    print("actual: ", target[test], " predicted: ", end=" ")
    print(round(clf.predict([data[test]])[0], 2))


if __name__ == '__main__':
    main()
