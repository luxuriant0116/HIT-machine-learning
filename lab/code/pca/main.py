import random

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

from pca import PCA
import utils
from load import load_mnist
from load import load_fer

random.seed(0)
np.random.seed(0)


def testPCA_3d():
    mu = [0, 0, 0]
    sigma = [1, 2, 0.5]
    num_sample = 100
    data = utils.generate_data(mu, sigma, num_sample)
    # axis = np.array([1, 1, 2])
    # data = utils.rotate_3d_data(data, axis)
    utils.visualize_3d_data(data)
    expected_dim = 2
    model = PCA(data, expected_dim)
    recover = model.recover_dim()
    utils.visualize_difference_3d(data, recover)
    utils.visualize_difference(data, recover)


def testPCA_2d():
    mu = [0, 0]
    sigma = [1, 0.05]
    num_sample = 100
    data = utils.generate_data(mu, sigma, num_sample)
    angle = np.pi / 12
    data = utils.rotate_2d_data(data, angle)
    utils.visualize_2d_data(data)
    expected_dim = 1
    model = PCA(data, expected_dim)
    recover = model.recover_dim()
    utils.visualize_difference_2d(data, recover)


def face_compression():
    faces = []
    for i in range(6):
        path = './chalemet/' + str(i + 1) + '.png'
        # path = './yalefaces/subject0' + str(i + 1) + '.happy'
        face = img.imread(path)[:, :, 0]
        # print(face.shape)
        plt.subplot(2, 3, i + 1)
        plt.imshow(face)
        face = np.array(face).reshape(50 * 50)
        face = face.astype(np.float64)
        faces.append(face)
    plt.show()
    faces = np.array(faces)

    model = PCA(faces, 6)
    compress = model.recover_dim()

    for i in range(6):
        image = compress[i].reshape((50, 50))
        origin_image = faces[i].reshape((50, 50))
        psnr = utils.compute_psnr(image, origin_image)
        plt.subplot(2, 3, i+1)
        plt.title('PSNR: %.2f' % psnr)
        plt.imshow(image)
    plt.show()


def compression_mnist():
    imgs, labels = load_mnist()
    imgs = imgs[:9]
    id = 1
    for img in imgs:
        plt.subplot(3, 3, id)
        id += 1
        image = img.reshape((28, 28))
        plt.imshow(image)
    plt.show()

    imgs = np.array(imgs)
    model = PCA(imgs, 2)
    compress = model.recover_dim()
    for i in range(9):
        image = compress[i].reshape((28, 28))
        origin_image = imgs[i].reshape((28, 28))
        psnr = utils.compute_psnr(image, origin_image)
        plt.subplot(3, 3, i + 1)
        plt.title('PSNR = %.2f' % psnr)
        plt.imshow(image)
    plt.show()


def compression_fer():
    faces = load_fer()
    faces = faces[1:10]
    id = 1
    for face in faces:
        plt.subplot(3, 3, id)
        id += 1
        image = face.reshape((48, 48))
        plt.imshow(image)
    plt.show()

    model = PCA(faces, 6)
    compress = model.recover_dim()
    for i in range(9):
        image = compress[i].reshape((48, 48))
        origin_image = faces[i].reshape((48, 48))
        psnr = utils.compute_psnr(image, origin_image)
        plt.subplot(3, 3, i + 1)
        plt.title('PSNR = %.2f' % psnr)
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    # testPCA_2d()
    # testPCA_3d()
    # face_compression()
    # compression_mnist()
    compression_fer()
