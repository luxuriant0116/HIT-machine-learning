import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(116)
random.seed(116)


def generate_gauss_data(mu, sigma, dim):
    """
    generate data that satisfy gauss distribution
    :param mu: mean
    :param sigma: variance
    :param dim: dimension of parameter x
    :return: generated data
    """
    x = []
    for j in range(dim):
        x.append(random.gauss(mu[j], sigma[j]))
    return x


def generate_data(mu, sigma, num_class, num_sample):
    """
    generate data of k classes, and data in each class satisfy gauss distribution
    :param mu: mean
    :param sigma: variance
    :param num_class: number of classes (k)
    :param num_sample: number of samples in each class
    :return: generated data
    """
    assert len(mu) == len(sigma)
    assert len(mu[0]) == len(sigma[0])
    assert len(sigma) == num_class
    assert len(num_sample) == len(sigma)
    class_id = 0
    x_dim = len(mu[0])
    data = []
    for num in num_sample:
        for i in range(num):
            x = generate_gauss_data(mu[class_id], sigma[class_id], x_dim)
            x.append(class_id)
            data.append(x)
        class_id += 1
    return data


def visualize_2d_data(data, num_class, color=True, color_list=None):
    if color_list is None:
        color_list = ['tomato', 'orange', 'gold',  'palegreen', 'turquoise', 'skyblue', 'mediumpurple', 'hotpink', 'pink', ]
    classify = []
    for i in range(num_class):
        classify.append([])
    for r in data:
        classify[int(r[-1])].append(r[:-1])

    class_id = 0
    for c in classify:
        c = np.array(c)
        if color:
            plt.plot(c[:, 0], c[:, 1], 'ro', color=color_list[class_id])
        else:
            plt.plot(c[:, 0], c[:, 1], 'ro', color=color_list[-1])
        class_id += 1
    plt.show()


def process_data(mu, sigma, num_sample, is_shuffle=True):
    num_class = len(num_sample)
    data = generate_data(mu, sigma, num_class, num_sample)
    plt.title('original')
    visualize_2d_data(data, num_class)
    random.shuffle(data)
    data = np.array(data)[:, :-1]
    return data


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
        if labels[i] == 0 and logits[i] == 2:
            correct += 1
        elif labels[i] == 1 and logits[i] == 1:
            correct += 1
        elif labels[i] == 2 and logits[i] == 0:
            correct += 1
    acc = correct / total
    return acc


if __name__ == '__main__':
    mu = [[1, 1], [0.5, 2], [4, 1.5], [4.5, 4]]
    sigma = [[1, 1], [1, 1], [1, 1], [1, 1]]
    num_sample = [650, 625, 675, 650]
    num_class = len(num_sample)
    data = generate_data(mu, sigma, num_class, num_sample)
    visualize_2d_data(data, num_class)
    plt.show()
