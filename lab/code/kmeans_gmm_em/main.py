import numpy as np
from matplotlib import pyplot as plt

import utils
import kmeans
import gmm
import load

np.random.seed(0)

def test_kmeans():
    # mu = [[1, 1], [1.5, 1.5]]
    # sigma = [[3, 3], [0.5, 0.5]]
    num_sample = [500, 500, 500, 500]
    mu = [[0, 2], [1, 3], [3, 2], [3, 4], [4, 5]]
    sigma = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    # mu = [[1, 1],  [6, 4], [10, 10]]
    # sigma = [[3, 3], [2, 2], [2.5, 2]]
    num_class = len(num_sample)
    data = utils.process_data(mu, sigma, num_sample)
    model = kmeans.KMeans(num_class, data)
    model.forward(1000)
    model.visualize()


def init_gmm_randomly(data, num_class):
    mu_init = data[np.random.choice(data.shape[0], size=num_class, replace=False)]
    sigma_init = []
    for i in range(num_class):
        sigma_init.append(np.cov(data, rowvar=False))
    sigma_init = np.array(sigma_init)
    pi_init = np.random.rand(num_class)
    pi_init = pi_init / np.sum(pi_init)
    return mu_init, sigma_init, pi_init


def init_gmm_with_kmeans(center, cluster, num_class, num_sample):
    mu_init = center
    sigma_init = []
    pi_init = np.zeros(num_class)
    classify = []
    for i in range(num_class):
        classify.append([])
    for r in cluster:
        classify[int(r[-1])].append(r[:-1])
    for k in range(num_class):
        if len(classify[k]) == 1:
            dim = len(classify[k][0])
            sigma_init.append(np.zeros((dim, dim)) + 1e-6)  # add a regular term
        else:
            sigma_init.append(np.cov(classify[k], rowvar=False))
        pi_init[k] = len(classify[k]) / num_sample
    sigma_init = np.array(sigma_init)
    return mu_init, sigma_init, pi_init


def test_gmm():
    # mu = [[1, 1], [1, 6], [8, 1], [10, 10]]
    # sigma = [[0.5, 0.5], [0.5, 0.5], [0.6, 0.6], [0.5, 0.7]]
    # sigma = [[0.2, 0.2], [0.5, 0.5], [0.2, 0.2], [0.3, 0.3]]
    mu = [[1, 2], [1, 3], [3, 2], [3, 4]]
    sigma = [[1, 1], [1, 1], [1, 1], [1, 1]]
    # mu = [[1, 1], [1.5, 1.5], [10, 10]]
    # sigma = [[3, 3], [1, 1], [2, 2]]
    num_sample = [500, 500, 500, 500]
    num_class = len(num_sample)
    data = utils.process_data(mu, sigma, num_sample)
    kMeans = kmeans.KMeans(num_class, data)
    kMeans.forward(1000)
    kMeans.visualize()

    # mu_init, sigma_init, pi_init = init_gmm_randomly(data, num_class)
    mu_init, sigma_init, pi_init = init_gmm_with_kmeans(kMeans._center, kMeans._cluster, num_class, data.shape[0])
    GMM = gmm.GaussianMixtureModel(num_class, mu_init, sigma_init, pi_init, data)
    GMM.forward(epoch=5)
    GMM.visualize_distribution(np.array(mu), np.array(sigma))
    GMM.visualize()


def test_uci():
    data = load.load_iris()
    data, label = load.process_train(data)
    num_class = max(label) + 1
    kMeans = kmeans.KMeans(num_class, data)
    kMeans.forward(1000)
    mu_init, sigma_init, pi_init = init_gmm_with_kmeans(kMeans._center, kMeans._cluster, num_class, data.shape[0])
    GMM = gmm.GaussianMixtureModel(num_class, mu_init, sigma_init, pi_init, data)
    GMM.forward(epoch=10)
    pred = GMM.predict()
    print("truth: " + str(label))
    print("prediction: " + str(pred))
    acc = utils.evaluate(label, pred)
    print("\nAccuracy on Iris: " + str(acc))


if __name__ == '__main__':
    # test_kmeans()
    test_gmm()
    # test_uci()
