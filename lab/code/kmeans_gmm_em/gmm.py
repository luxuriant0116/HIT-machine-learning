import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from tqdm import trange
import matplotlib.pyplot as plt
from utils import visualize_2d_data


class GaussianMixtureModel:
    def __init__(self, k, mu, sigma, pi, data):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        self.data = data

    def expectation_step(self):
        prob = []
        for i in range(self.k):
            prob.append(self.probability(i))
        prob = np.transpose(prob)
        numerator = self.pi * prob
        denominator = np.sum(numerator, axis=-1)
        denominator = np.repeat(denominator, self.k, axis=0).reshape((-1, self.k))
        responsible_matrix = numerator / denominator
        return responsible_matrix

    def maximization_step(self, responsible_matrix):
        valid_num = np.sum(responsible_matrix, axis=0)

        # update mu
        for i in range(self.k):
            responsibility = np.repeat(responsible_matrix[:, i], self.data.shape[1], axis=0).reshape(self.data.shape[0],
                                                                                                     -1)
            self.mu[i] = np.sum(self.data * responsibility, axis=0) / valid_num[i]

        # update sigma
        for i in range(self.k):
            self.sigma[i] = np.zeros((self.data.shape[1], self.data.shape[1]))
            bias = np.matrix(self.data - self.mu[i])
            for j in range(self.data.shape[0]):
                self.sigma[i] += np.dot(bias[j].T, bias[j]) * responsible_matrix[j][i]
            self.sigma[i] /= valid_num[i]

        # update pi
        self.pi = valid_num / self.data.shape[0]

    def log_likelihood(self):
        prob = []
        for i in range(self.k):
            prob.append(self.probability(i))
        prob = np.sum(np.transpose(prob) * self.pi, axis=-1)
        likelihood = np.mean(np.log(prob))
        return likelihood

    def forward(self, epoch, log=True):
        for e in trange(epoch):
            response = self.expectation_step()
            self.maximization_step(response)
            if log:
                print('\nlog-likelihood: ' + str(self.log_likelihood()))

    def probability(self, class_id):
        x_dim = self.data.shape[1]
        denominator = ((2 * np.pi) ** (x_dim / 2)) * ((np.linalg.det(self.sigma[class_id])) ** (1 / 2))
        bias = self.data - self.mu[class_id]
        numerator = np.exp(np.diagonal(- 1 / 2 * np.dot(np.dot(bias, np.linalg.inv(self.sigma[class_id])), bias.T)))
        prob = numerator / denominator
        # prob = np.diagonal(prob)
        return prob

    def predict(self):
        prob = []
        for i in range(self.k):
            prob.append(self.probability(i))
        prob = np.transpose(prob)
        pred = np.argmax(prob, axis=-1)
        return pred

    def visualize_distribution(self, mu_true=None, sigma_true=None):
        plt.title('N=500')
        plt.scatter(self.data[:, 0], self.data[:, 1], c='mistyrose')
        ax = plt.gca()
        for i in range(self.k):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': 'violet', 'ls': ':'}
            ellipse = Ellipse(self.mu[i], 3 * np.sqrt(self.sigma[i][0][0]), 3 * np.sqrt(self.sigma[i][1][1]), **plot_args)
            ax.add_patch(ellipse)

        if (mu_true is not None) and (sigma_true is not None):
            for i in range(self.k):
                plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': 'purple'}
                ellipse = Ellipse(mu_true[i], 3 * np.sqrt(sigma_true[i][0]), 3 * np.sqrt(sigma_true[i][1]), **plot_args)
                ax.add_patch(ellipse)
        plt.show()

    def visualize(self):
        pred = self.predict()
        cluster = np.c_[self.data, pred]
        plt.title('N=500')
        color_list = ['skyblue', 'mediumpurple', 'hotpink', 'pink', 'tomato', 'orange', 'gold', 'palegreen', 'turquoise',]
        visualize_2d_data(cluster, self.k, color_list=color_list)

if __name__ == '__main__':
    data = np.array([[1, 2], [0, 0], [1, 1]])
    mu = np.array([[0, 1], [0, 1]])
    sigma = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    a = np.array([[1, 3], [2, 3], [4, 5]])
    b = np.array([1, 2, 3])
    de = np.sum(a, axis=-1)
    a[0] = np.array([1, 2])
    print(a)
