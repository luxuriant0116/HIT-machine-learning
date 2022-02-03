import random

import numpy as np
from random import gauss
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)


def generate_data(N, mu=0, sigma=0.5):
    # generate true values
    x = np.linspace(0, 1, N, endpoint=True)
    y = np.sin(2 * np.pi * x)

    # add gaussian noise
    for i in range(x.size):
        y[i] += gauss(mu, sigma)

    return x, y


def analyse_without_penalty(order, x, y):
    vander = np.vander(x, order + 1)  # Vandermonde matrix of x
    w = np.dot(np.matrix(np.dot(vander.T, vander)).I, np.dot(vander.T, y))  # the solution of w
    # w = np.dot(np.linalg.inv(np.matrix(np.dot(vander.T, vander))), np.dot(vander.T, y))
    # 改精度? float64?
    return w


def analyse_with_penalty(order, x, y, lambd):
    vander = np.vander(x, order + 1)  # Vandermonde matrix of x
    w = np.dot(np.matrix(np.dot(vander.T, vander) + lambd * np.identity(order + 1)).I, np.dot(vander.T, y))
    return w


def test_lambd():
    x_continuous = np.arange(-40, -15, 1)
    x, y = generate_data(10)
    w = []
    for i in range(x_continuous.size):
        w.append(analyse_with_penalty(9, x, y, np.exp(x_continuous[i])).tolist()[0])
    rms = []
    x, y = generate_data(10)
    vander = np.vander(x, 10)
    for i in range(x_continuous.size):
        logits = np.dot(vander, w[i])  # prediction
        loss = np.sum(np.square(logits - y)) / 2 + np.exp(x_continuous[i])/ 2 * np.dot(w[i], w[i])
        # the value of loss function
        rms.append(np.sqrt(2*loss/10))
    plt.plot(x_continuous, rms)
    plt.show()


def gradient_descent_per_epoch(params, grads, learning_rate):
    params -= grads * learning_rate
    return params


def gradient_descent_without_penalty(order, epoch, x, y, learning_rate=0.001, plot_loss=False):
    w = np.random.randn(order + 1)
    vander = np.vander(x, order + 1)

    epochs = []
    losses = []
    for e in range(epoch):
        logits = np.dot(vander, w)  # prediction
        loss = np.sum(np.square(logits - y)) / 2  # the value of loss function
        grads = np.sum((logits - y) * vander.T, axis=-1)  # compute the gradient

        # w -= grads * learning_rate
        w = gradient_descent_per_epoch(w, grads, learning_rate)  # update

        # save epoch and related loss in order to plot later
        if e % 100 == 0:
            epochs.append(e)
            losses.append(loss)
        # epochs.append(e)
        # losses.append(loss)

    # plot the tendency of loss
    if plot_loss:
        visualize_loss(losses, epochs)

    return w


def gradient_descent_with_penalty(order, epoch, x, y, lambd, learning_rate=0.001, plot_loss=False):
    w = np.random.randn(order + 1)
    vander = np.vander(x, order + 1)

    epochs = []
    losses = []
    for e in range(epoch):
        logits = np.dot(vander, w)  # prediction
        loss = np.sum(np.square(logits - y)) / 2 + lambd / 2 * np.dot(w, w)  # value of loss function
        grads = np.sum((logits - y) * vander.T, axis=-1) + lambd * w  # compute the gradient
        w = gradient_descent_per_epoch(w, grads, learning_rate)  # update

        if e % 100 == 0:
            epochs.append(e)
            losses.append(loss)

    if plot_loss:
        visualize_loss(losses, epochs)

    return w


def conjugate_gradient(A, b, w):
    r = b - np.dot(A, w)
    p = r
    while np.dot(r, r) > 1e-10:
        a = np.dot(r, r) / np.dot(np.dot(p, A), p)
        w += a * p
        r_pre = r.copy()
        r -= a * np.dot(A, p)
        b = np.dot(r, r) / np.dot(r_pre, r_pre)
        p = r + b * p
    return w


def conjugate_gradient_without_penalty(order, x, y):
    w = np.random.randn(order + 1)
    vander = np.vander(x, order + 1)
    # AX = b
    A = np.dot(vander.T, vander)
    b = np.dot(vander.T, y)
    return conjugate_gradient(A, b, w)


def conjugate_gradient_with_penalty(order, x, y, lambd):
    w = np.random.randn(order + 1)
    vander = np.vander(x, order + 1)
    A = np.dot(vander.T, vander) + lambd * np.identity(order + 1)
    b = np.dot(vander.T, y)
    return conjugate_gradient(A, b, w)


# visualize the curve: origin/with penalty/without penalty
def visualize(type, order, x=None, y=None, w=None):
    x_continuous = np.arange(0, 1, 0.01)

    if type == 'naive':
        y_fit = np.dot(np.vander(x_continuous, order + 1), w.T)
        plt.plot(x_continuous, y_fit, color='orchid', label='fitting curve without penalty')

    elif type == 'origin':
        y_sin = np.sin(2 * np.pi * x_continuous)
        plt.scatter(x, y, color='mistyrose')
        plt.plot(x_continuous, y_sin, color='pink', label='original curve')

    elif type == 'penalty':
        y_fit = np.dot(np.vander(x_continuous, order + 1), w.T)
        plt.plot(x_continuous, y_fit, color='purple', label='fitting curve with penalty')

    plt.legend()


# visualize the tendency of loss
def visualize_loss(loss, epoch):
    plt.plot(epoch, loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.show()


# --------------------- test ---------------------
def test_order_analyse():
    x, y = generate_data(10)
    for order in range(11):
        visualize('origin', order, x=x, y=y)
        w1 = analyse_without_penalty(order, x, y)
        visualize('naive', order, w=w1)
        w2 = analyse_with_penalty(order, x, y, lambd=np.exp(-18))
        visualize('penalty', order, w=w2)
        plt.show()


def test_order_gd():
    x, y = generate_data(10)
    for order in range(10):
        visualize('origin', order, x=x, y=y)
        w1 = gradient_descent_without_penalty(order, 1000000, x, y)
        visualize('naive', order, w=w1)
        w2 = gradient_descent_with_penalty(order, 1000000, x, y, lambd=np.exp(-18))
        visualize('penalty', order, w=w2)
        plt.show()


def test_order_cg():
    x, y = generate_data(10)
    for order in range(10):
        visualize('origin', order, x=x, y=y)
        w1 = conjugate_gradient_without_penalty(order, x, y)
        visualize('naive', order, w=w1)
        w2 = conjugate_gradient_with_penalty(order, x, y, lambd=np.exp(-18))
        visualize('penalty', order, w=w2)
        plt.show()


def test_data_analyse():
    order = 9
    for num in [10, 50, 100, 500, 1000]:
        x, y = generate_data(num)
        plt.title('N = ' + str(num))
        visualize('origin', order, x, y)
        w1 = analyse_without_penalty(order, x, y)
        visualize('naive', order, w=w1)
        w2 = analyse_with_penalty(order, x, y, lambd=np.exp(-18))
        visualize('penalty', order, w=w2)

        plt.legend()
        plt.show()


def test_data_gd():
    order = 9
    for num in [10, 50, 100, 500, 1000]:
        x, y = generate_data(num)
        plt.title('N = ' + str(num))
        visualize('origin', order, x, y)
        w1 = gradient_descent_without_penalty(order, 1000000, x, y)
        visualize('naive', order, w=w1)
        w2 = gradient_descent_with_penalty(order, 1000000, x, y, lambd=np.exp(-18))
        visualize('penalty', order, w=w2)

        plt.legend()
        plt.show()


def test_data_cg():
    order = 9
    for num in [10, 50, 100, 500, 1000]:
        x, y = generate_data(num)
        plt.title('N = ' + str(num))
        visualize('origin', order, x, y)
        w1 = conjugate_gradient_without_penalty(order, x, y)
        visualize('naive', order, w=w1)
        w2 = conjugate_gradient_with_penalty(order, x, y, lambd=np.exp(-18))
        visualize('penalty', order, w=w2)

        plt.legend()
        plt.show()


def test_lr_gd(lr):
    order = 3
    x, y = generate_data(10)
    plt.title('learning rate = ' + str(lr))
    visualize('origin', order, x, y)
    w1 = gradient_descent_without_penalty(order, 1000000, x, y, learning_rate=lr)
    visualize('naive', order, w=w1)
    w2 = gradient_descent_with_penalty(order, 1000000, x, y, lambd=np.exp(-18), learning_rate=lr)
    visualize('penalty', order, w=w2)
    plt.show()


def test_epoch_gd(epoch):
    order = 9
    x, y = generate_data(10)
    plt.title('epoch = ' + str(epoch))
    visualize('origin', order, x, y)
    w1 = gradient_descent_without_penalty(order, epoch, x, y, learning_rate=1e-3)
    visualize('naive', order, w=w1)
    w2 = gradient_descent_with_penalty(order, epoch, x, y, lambd=np.exp(-18), learning_rate=1e-3)
    visualize('penalty', order, w=w2)
    plt.show()

    # w1 = gradient_descent_without_penalty(order, epoch, x, y, learning_rate=1e-3, plot_loss=True)


def additive_test():
    order = 50
    num = 100
    x, y = generate_data(num)
    # plt.title('order = ' + str(order) + ', N = ' + str(num) + ', method: conjugate')
    # plt.ylim((-2, 2))
    # visualize('origin', order, x, y)
    # w1 = conjugate_gradient_without_penalty(order, x, y)
    # visualize('naive', order, w=w1)
    # w2 = conjugate_gradient_with_penalty(order, x, y, lambd=np.exp(-18))
    # visualize('penalty', order, w=w2)
    # plt.show()

    plt.title('order = ' + str(order) + ', N = ' + str(num) + ', method: analyse')
    plt.ylim((-2, 2))
    visualize('origin', order, x=x, y=y)
    w3 = analyse_without_penalty(order, x, y)
    visualize('naive', order, w=w3)
    w4 = analyse_with_penalty(order, x, y, lambd=np.exp(-18))
    visualize('penalty', order, w=w4)
    plt.show()



if __name__ == '__main__':
    # test_order_analyse()
    # test_order_cg()
    # test_order_gd()
    # test_data_analyse()
    # test_data_gd()
    # test_data_cg()
    # test_epoch_gd(1000000000)
    # test_lambd()
    additive_test()

    # x, y = generate_data(10)
    # w = gradient_descent_without_penalty(3, 1000000, x, y, plot_loss=True)

    # x, y = generate_data(10)
    # visualize('origin', 9, x, y)
    # plt.title('lambd = 1')
    # w1 = analyse_without_penalty(9, x, y)
    # visualize('naive', 9, w=w1)
    # w = analyse_with_penalty(9, x, y, lambd=1)
    # visualize('penalty', 9, w=w)
    # plt.show()

    # x, y = generate_data(10)
    # w1 = analyse_without_penalty(9, x, y)
    # w2 = analyse_with_penalty(9, x, y, lambd=np.exp(-18))
    # print('without penalty:')
    # print(w1)
    # print('with penalty:')
    # print(w2)

    # w = analyse_with_penalty(order, x, y, np.exp(-18))
    # w = conjugate_gradient_without_penalty(order, 10, x, y)
