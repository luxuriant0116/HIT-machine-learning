import numpy as np
import random
import matplotlib.pyplot as plt
import numpy.random
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import math


random.seed(0)
numpy.random.seed(0)

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


def generate_data(mu, sigma, num_sample):
    """
    generate data
    :param mu: mean
    :param sigma: variance
    :param num_sample: number of samples in each class
    :return: generated data
    """
    x_dim = len(mu)
    data = []
    for i in range(num_sample):
        x = generate_gauss_data(mu, sigma, x_dim)
        data.append(x)
    return np.array(data)


def rotate_mat(axis, radian):
    """
    compute rotate matrix
    :param axis: axis
    :param radian: angle
    :return: rotate matrix
    """
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


def rotate_2d_data(data, angle):
    rotate_matrix = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotate_data = np.dot(data, rotate_matrix)
    return rotate_data



def rotate_3d_data(data, axis):
    radian = math.pi / 3
    rotate_matrix = rotate_mat(axis, radian)
    rotate = np.dot(data, rotate_matrix)
    return rotate


def visualize_2d_data(data):
    assert data.shape[1] == 2
    plt.plot(data[:, 0], data[:, 1], 'ro', color='violet')
    plt.xlim([min(data[:, 0]) - 0.5, max(data[:, 0]) + 0.5])
    # plt.ylim([min(data[:, 0]) + 1.5, max(data[:, 0]) - 1.5])
    plt.ylim([min(data[:, 0]) + 1, max(data[:, 0]) - 1])
    plt.show()


def visualize_3d_data(data):
    assert data.shape[1] == 3

    # projection in xOy, xOz, and yOz
    for elev, azim, title in zip([0, 0, 90], [0, 90, 0], ["yOz", "xOz", "xOy"]):
        fig = plt.figure()
        fig.suptitle(title)
        ax = Axes3D(fig)
        ax.view_init(elev=elev, azim=azim)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.show()

    # 3d image
    fig = plt.figure()
    fig.suptitle('3D')
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


def visualize_difference_3d(data, recover):
    fig = plt.figure()
    fig.suptitle('3D')
    ax = Axes3D(fig)
    ax.scatter(recover[:, 0], recover[:, 1], recover[:, 2], c='r')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


def visualize_difference(data, recover):
    for elev, azim, title in zip([0, 0, 90], [0, 90, 0], ["yOz", "xOz", "xOy"]):
        fig = plt.figure()
        fig.suptitle(title)
        ax = Axes3D(fig)
        ax.view_init(elev=elev, azim=azim)
        ax.scatter(recover[:, 0], recover[:, 1], recover[:, 2], c='r')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.show()


def visualize_difference_2d(data, recover):
    plt.plot(data[:, 0], data[:, 1], 'ro', color='violet')
    plt.plot(recover[:, 0], recover[:, 1], 'ro', color='pink')
    plt.xlim([min(data[:, 0]) - 0.5, max(data[:, 0]) + 0.5])
    plt.ylim([min(data[:, 0]) + 1, max(data[:, 0]) - 1])
    plt.show()


def compute_psnr(origin, compress):
    assert origin.shape[0] == compress.shape[0]
    assert origin.shape[1] == compress.shape[1]
    r = origin.shape[0]
    c = origin.shape[1]
    mse = np.sum(np.square(origin - compress)) / (r * c)
    psnr = 20 * math.log10(255 / math.sqrt(mse))
    return psnr


if __name__ == '__main__':
    # mu = [1, 1, 1]
    # sigma = [1, 2, 0.01]
    mu = [0, 0]
    sigma = [1, 0.05]
    num_sample = 100
    data = generate_data(mu, sigma, num_sample)
    # axis = np.array([1, 2])
    angle = np.pi / 12
    # data = rotate_2d_data(data, angle)
    # data = rotate_3d_data(data, axis)
    # visualize_3d_data(data)
    visualize_2d_data(data)
    # mean = np.mean(data, axis=0)
    # print(mean)
    # print(data - mean)
