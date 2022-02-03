import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1219)
np.random.seed(1219)


def generate_bernoulli_data(num_sample):
    """
    generate data that satisfy bernoulli distribution
    :param num_sample: the total number of samples to generate
    :return: the number of samples in two classes
    """
    num_class_1 = random.randint(1, num_sample - 1)
    num_class_2 = num_sample - num_class_1
    return num_class_1, num_class_2


def generate_gauss_data(mu, sigma, dim):
    """
    generate data that satisfy gauss distribution
    :param mu: mean
    :param sigma: variance
    :param dim: dimension of parameter x
    :return: generated data x
    """
    x = []
    for j in range(dim):
        x.append(random.gauss(mu[j], sigma[j]))
    return x


def generate_independent_data(mu, sigma, num_sample):
    """
    Generate data of (X, y) pairs, in which every dimension of X satisfies an independent Gaussian Distribution,
    and y satisfies a Bernoulli Distribution with the number of classes at 2.
    :param mu: a matrix of (2, x_dim)
    :param sigma: a matrix of (2, x_dim)
    :param num_sample: the number of samples to generate
    :return: data
    """
    assert len(mu) == len(sigma)
    assert len(mu[0]) == 2
    assert len(mu[1]) == 2
    x_dim = len(mu)

    num_class_1, num_class_2 = generate_bernoulli_data(num_sample)
    # num_class_1, num_class_2 = int(0.6 * num_sample), int(0.4 * num_sample)

    data = []
    for i in range(num_class_1):
        x = generate_gauss_data(mu[0], sigma[0], x_dim)
        data.append([x, [0]])

    for i in range(num_class_2):
        x = generate_gauss_data(mu[1], sigma[1], x_dim)
        data.append([x, [1]])
    return data


def generate_multi_variate_gauss_data(mu, cov):
    """
    generate data that satisfy multi-variate gauss
    :param mu: mean
    :param cov: variance
    :return: generated data
    """
    assert len(mu) == len(cov)
    assert len(cov) == len(cov[0])
    x_dim = len(mu)

    x = np.random.multivariate_normal(mu, cov).tolist()
    return x


def generate_correlated_data(mu, cov, num_sample):
    """
    Generate data of (X, y) pairs, in which X satisfies a multi-variate Gaussian Distribution,
    and y satisfies a Bernoulli Distribution with the number of classes at 2.
    :param mu: the mean of multi-variate Gaussian Distribution
    :param cov: the covariance matrix of multi-variate Gaussian Distribution
    :param num_sample: the number of samples to generate
    :return: data
    """
    assert len(mu) == 2
    assert len(cov) == 2

    num_class_1, num_class_2 = generate_bernoulli_data(num_sample)
    data = []
    for i in range(num_class_1):
        x = generate_multi_variate_gauss_data(mu[0], cov[0])
        data.append([x, [0]])

    for i in range(num_class_2):
        x = generate_multi_variate_gauss_data(mu[1], cov[1])
        data.append([x, [1]])

    return data


def shuffle_data(data, shuffle=True):
    """
    shuffle data randomly
    :param data: input data
    :param shuffle: conduct shuffle operation or not
    :return: processed data
    """
    if shuffle:
        random.shuffle(data)
    X = []
    Y = []
    for d in data:
        X.append(d[0])
        Y.append(d[1][0])
    return np.array(X), np.array(Y)


# def shuffle_data_multi(data):
#     random.shuffle(data)
#     X = []
#     Y = []
#     for d in data:
#         X.append(d[0])
#         if d[1][0] == 0:
#             Y.append([1, 0])
#         else:
#             Y.append([0, 1])
#     return np.array(X), np.array(Y)


def visualize_data(data, mode='train'):
    x_class_1, y_class_1, x_class_2, y_class_2 = [], [], [], []
    for d in data:
        if d[1][0] == 0:
            x_class_1.append(d[0][0])
            y_class_1.append(d[0][1])
        if d[1][0] == 1:
            x_class_2.append(d[0][0])
            y_class_2.append(d[0][1])
    if mode == 'train':
        plt.plot(x_class_1, y_class_1, 'ro', color='pink')
        plt.plot(x_class_2, y_class_2, 'ro', color='wheat')
    if mode == 'test':
        plt.plot(x_class_1, y_class_1, 'ro', color='red')
        plt.plot(x_class_2, y_class_2, 'ro', color='yellow')


def visualize_result_2d(param, x_min, x_max, y_min, y_max):
    w = param[:-1]
    b = param[-1]
    x = np.arange(x_min, x_max, 0.01)
    y = -(b + w[0] * x) / w[1]
    y_min = min(y_min, y[0], y[-1])
    y_max = max(y_max, y[0], y[-1])
    plt.plot(x, y, color='thistle')
    plt.fill_between(x, y, y_max, color='cornsilk', alpha=0.3)
    plt.fill_between(x, y_min, y, color='mistyrose', alpha=0.3)


# def visualize_result_2d_multi(param, x_min, x_max, y_min, y_max):
#     w = param.T[:-1].T
#     b = param.T[-1]
#     x = np.arange(x_min, x_max, 0.01)
#     y = -(b + w[0][0] * x) / w[0][1]
#     y_min = min(y_min, y[0], y[-1])
#     y_max = max(y_max, y[0], y[-1])
#     plt.plot(x, y, color='thistle')
#     plt.fill_between(x, y, y_max, color='mistyrose', alpha=0.3)
#     plt.fill_between(x, y_min, y, color='cornsilk', alpha=0.3)


def evaluate(labels, logits):
    """
    evaluate the accuracy
    :param labels: true
    :param logits: prediction
    :return: accuracy
    """
    total = len(labels)
    correct = 0
    for i in range(total):
        correct += (labels[i] == logits[i])
    acc = correct / total
    return acc


if __name__ == '__main__':
    # ------------------------ test ------------------------
    mu = [[0, 1], [0.3, 0.1]]
    sigma = [[0.1, 0.2], [0.1, 0.2]]
    # cov = [[[1, 1], [1, 2]], [[1, 1], [1, 2]]]
    data_ind = generate_independent_data(mu, sigma, 100)
    # data_cov = generate_correlated_data(mu, cov, 100)
    # visualize_data(data_ind)
    # plt.show()
    X = [1, 2, 3]
    Y = [1, 3, 3]
    print(len([x for x in X if x in Y]))
