import numpy as np
import random


random.seed(0)


def load_iris():
    data = []
    with open('iris.data') as f:
        for line in f:
            l = line.strip('\n').split(',')
            x = [float(scalar) for scalar in l[:-1]]
            if l[-1] == 'Iris-setosa':
                x.append(0)
            elif l[-1] == 'Iris-versicolor':
                x.append(1)
            elif l[-1] == 'Iris-virginica':
                x.append(2)
            data.append(x)
    f.close()
    return data


def process_train(train):
    random.shuffle(train)
    X = []
    Y = []
    for d in train:
        X.append(d[:-1])
        Y.append(d[-1])
    X = normalize(np.array(X))
    return X, np.array(Y)


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data


if __name__ == '__main__':
    data = load_iris()
    print(data)
