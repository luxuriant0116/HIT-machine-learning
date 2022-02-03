import matplotlib.pyplot as plt

from logistic_regression import logistic_regression_classifier
from logistic_regression import multi_logistic_regression_classifier
import utils
import numpy as np

import load


def test_independent_data(mu, sigma, x_dim):
    data_ind = utils.generate_independent_data(mu, sigma, 10)
    data_test = utils.generate_independent_data(mu, sigma, 1000)
    plt.title('Class 1: mean(x1) = ' + str(mu[0][0]) + ', mean(x2) = ' + str(mu[0][1]) + '; var(x1) = ' + str(sigma[0][0])
              + ', var(x2) = ' + str(sigma[0][1]) + '\nClass 2: mean(x1) = ' + str(mu[1][0]) + ', mean(x2) = ' + str(mu[1][1]) + '; var(x1) = '
              + str(sigma[1][0]) + ', var(x2) = ' + str(sigma[1][1]) +
              '\nWith penalty')

    utils.visualize_data(data_ind)
    # utils.visualize_data(data_test, mode='test')
    # print(data_ind)
    X, Y = utils.shuffle_data(data_ind)
    X_test, Y_test = utils.shuffle_data(data_test, shuffle=False)
    x_min, x_max, y_min, y_max = min(X_test.T[0]), max(X_test.T[0]), min(X_test.T[1]), max(X_test.T[1])
    classifier = logistic_regression_classifier(x_dim)
    # classifier.train_gradient_descent(X, Y, epoch=100000)
    classifier.train_gradient_descent(X, Y, epoch=100000, penalty=True, lambd=1e-3)

    print('Without penalty:')
    print('param: ' + str(classifier.param))

    predict = classifier.predict(X_test)
    acc = utils.evaluate(Y_test, predict)
    plt.xlabel('Accuracy on test set: ' + str(acc))
    # print("Accuracy on test set: ")
    # print(utils.evaluate(Y_test, predict))

    utils.visualize_result_2d(classifier.param, x_min, x_max, y_min, y_max)

    plt.show()


def test_correlated_data(mu, cov, x_dim):
    plt.title('Cov(+) = ' + str(cov[0][0][1]) + ', Cov(-) = ' + str(cov[1][0][1]))
    data_cov = utils.generate_correlated_data(mu, cov, 5)
    utils.visualize_data(data_cov)
    data_test = utils.generate_correlated_data(mu, cov, 50)
    utils.visualize_data(data_test, mode='test')
    # print(data_ind)
    X, Y = utils.shuffle_data(data_cov)
    X_test, Y_test = utils.shuffle_data(data_test, shuffle=False)
    x_min, x_max, y_min, y_max = min(X_test.T[0]), max(X.T[0]), min(X.T[1]), max(X.T[1])
    classifier = logistic_regression_classifier(x_dim)

    classifier.train_gradient_descent(X, Y, epoch=100000)
    # classifier.train_gradient_descent(X, Y, epoch=100000, penalty=True, lambd=np.exp(-18))
    utils.visualize_result_2d(classifier.param, x_min, x_max, y_min, y_max)

    predict = classifier.predict(X_test)
    acc = utils.evaluate(Y_test, predict)
    plt.xlabel('Accuracy on test set: ' + str(acc))

    plt.show()


def test_uci(dataset='iris'):
    if dataset == 'iris':
        data = load.load_iris()
    elif dataset == 'seeds':
        data = load.load_seeds()
    train, test = load.split_data(data)
    X_train, Y = load.process_train(train)
    X_test, target = load.process_test(test)
    _, train_target = load.process_test(train)
    classifier = multi_logistic_regression_classifier(X_train.shape[1], Y.shape[1])
    classifier.train_gradient_descent(X_train, Y, epoch=10000, log=True)

    X_valid, target_train = load.process_test(train)
    predict_train = classifier.predict(X_valid)
    print("Accuracy on train set: ")
    print(utils.evaluate(target_train, predict_train))
    predict = classifier.predict(X_test)
    print("Accuracy on test set: ")
    print(utils.evaluate(target, predict))


if __name__ == '__main__':
    mu = [[1, 1], [3, 3]]
    # sigma = [[0.8, 0.8], [0.9, 0.9]]
    sigma = [[0.4, 0.4], [0.3, 0.3]]
    # test_independent_data(mu, sigma, 2)
    # cov = [[[3, 2], [2, 3]], [[3, 2.5], [2.5, 3]]]
    # test_correlated_data(mu, cov, 2)

    test_uci('seeds')
