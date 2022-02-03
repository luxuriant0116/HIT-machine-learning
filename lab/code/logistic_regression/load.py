import numpy as np
import random
import re

random.seed(0)
np.random.seed(0)


def load_iris():
    data = []
    with open('uci_data/iris.data') as f:
        for line in f:
            l = line.strip('\n').split(',')
            x = [float(scalar) for scalar in l[:-1]]
            y = [0, 0, 0]
            if l[-1] == 'Iris-setosa':
                y[0] = 1
            elif l[-1] == 'Iris-versicolor':
                y[1] = 1
            elif l[-1] == 'Iris-virginica':
                y[2] = 1
            data.append([x, y])
    f.close()
    return data


def load_seeds():
    data = []
    with open('uci_data/seeds_dataset.txt') as f:
        for line in f:
            l = re.split(r"\t+", line.strip("\n"))
            x = [float(scalar) for scalar in l[:-1]]
            y = [0, 0, 0]
            if l[-1] == '1':
                y[0] = 1
            elif l[-1] == '2':
                y[1] = 1
            elif l[-1] == '3':
                y[2] = 1
            data.append([x, y])
    f.close()
    return data


def split_data(data):
    total = len(data)
    train_data = []
    test_data = []
    group = total // 10
    for g in range(group):
        for i in range(9):
            if g * 10 + i < total:
                train_data.append(data[g * 10 + i])
            else:
                break
        if g * 10 + 9 < total:
            test_data.append(data[g * 10 + 9])
    return train_data, test_data


def process_train(train):
    random.shuffle(train)
    X = []
    Y = []
    for d in train:
        X.append(d[0])
        Y.append(d[1])
    X = normalize(np.array(X))
    return X, np.array(Y)


def process_test(test):
    random.shuffle(test)
    X = []
    target = []
    for d in test:
        X.append(d[0])
        target.append(np.argmax(d[1]))
    X = normalize(np.array(X))
    return X, np.array(target)


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data


if __name__ == '__main__':
    data = load_iris()
    train, test = split_data(data)
    print(train)
