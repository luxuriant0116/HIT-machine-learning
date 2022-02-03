import numpy as np
from tqdm import trange
import optimizer


class logistic_regression_classifier:
    def __init__(self, x_dim):
        self.W = np.random.randn(x_dim)
        self.b = np.random.randn(1)

    @property
    def param(self):
        param = np.concatenate((self.W, self.b))
        return param

    def mcle_loss(self, X, Y):
        linear = self.b + np.dot(X, self.W.T)
        loss = - np.sum(Y * linear - np.log(1 + np.exp(linear)))
        return loss

    def mcle_loss_with_penalty(self, X, Y, lambd):
        linear = self.b + np.dot(X, self.W.T)
        loss = - np.sum(Y * linear - np.log(1 + np.exp(linear))) + lambd / 2 * np.dot(self.param, self.param.T)
        return loss

    def gradient(self, X, Y):
        linear = self.b + np.dot(X, self.W.T)
        sigmoid_linear = self.sigmoid(linear)
        w_grad = - (np.sum((Y - sigmoid_linear) * X.T, axis=1))
        b_grad = - (np.sum(Y - sigmoid_linear))
        grad = np.concatenate((w_grad, [b_grad]))
        # print(grad)
        return grad

    def train_gradient_descent(self, X, Y, lr=0.001, epoch=1000, log=False, penalty=False, lambd=None):
        loss = 0
        for e in trange(epoch):
            param = self.param
            if penalty and lambd is not None:
                grad = self.gradient_with_penalty(X, Y, lambd)
            else:
                grad = self.gradient(X, Y)
            param = optimizer.gradient_descent(param, grad, lr)
            self.W, self.b = param[:-1], [param[-1]]

            if penalty and lambd is not None:
                loss_new = self.mcle_loss_with_penalty(X, Y, lambd)
            else:
                loss_new = self.mcle_loss(X, Y)

            # if np.abs(loss_new - loss) < 1e-5:
            #     lr /= 2
            # loss = loss_new

            if log and e % 10000 == 0:
                print('\nMCLE Loss: ' + str(loss_new))
                print('gradient: ' + str(grad))

    @staticmethod
    def sigmoid(X):
        res = np.zeros(len(X))
        for i in range(len(X)):
            if X[i] >= 0:
                res[i] = 1 / (1 + np.exp(-X[i]))
            else:
                res[i] = np.exp(X[i]) / (1 + np.exp(X[i]))
        return res

    # with penalty
    def gradient_with_penalty(self, X, Y, lambd):
        linear = self.b + np.dot(X, self.W.T)
        sigmoid_linear = self.sigmoid(linear)
        w_grad = - (np.sum((Y - sigmoid_linear) * X.T, axis=1)) + lambd * self.W
        b_grad = - (np.sum(Y - sigmoid_linear)) + lambd * self.b[0]
        grad = np.concatenate((w_grad, [b_grad]))
        # print(grad)
        return grad

    def predict(self, X):
        logits = np.exp(np.dot(X, self.W.T) + self.b)
        logits = np.c_[logits, np.ones(logits.shape[0])]
        label = np.argmin(logits, axis=1)
        return label


class multi_logistic_regression_classifier:
    def __init__(self, x_dim, n_class):
        self.W = np.random.randn(n_class - 1, x_dim)
        self.b = np.random.randn(n_class - 1)

    def show_w(self):
        print(self.W)

    def show_b(self):
        print(self.b)

    @property
    def param(self):
        param = np.c_[self.W, self.b]
        return param

    def mcle_loss(self, X, Y):
        tmp = np.c_[np.exp(np.dot(X, self.W.T) + self.b), np.ones(X.shape[0])]
        sum_exp = np.sum(tmp, axis=1)
        sum_exp = np.reshape(sum_exp, (sum_exp.shape[0], 1))
        loss = - np.sum(Y * np.log(tmp / sum_exp)) / X.shape[0]
        return loss

    def gradient(self, X, Y):
        exp = np.exp(np.dot(X, self.W.T) + self.b)
        sum_exp = np.sum(exp, axis=1) + 1
        sum_exp = np.reshape(sum_exp, (sum_exp.shape[0], 1))
        X = np.expand_dims(np.c_[X, np.ones(X.shape[0])], axis=1)
        X = np.repeat(X, Y.shape[1] - 1, axis=1)
        tmp = np.expand_dims(Y.T[:-1].T, axis=-1) * X
        ratio = np.expand_dims(exp / sum_exp, axis=-1)
        grad = - np.sum(tmp - ratio * X, axis=0)
        return grad

    def train_gradient_descent(self, X, Y, lr=0.001, epoch=1000, log=True):
        for e in trange(epoch):
            param = self.param
            grad = self.gradient(X, Y)
            param = optimizer.gradient_descent(param, grad, lr)
            self.W = param.T[:-1].T
            self.b = param.T[-1]

            if log and e % 100 == 0:
                print(self.mcle_loss(X, Y))

    def predict(self, X):
        logits = np.exp(np.dot(X, self.W.T) + self.b)
        logits = np.c_[logits, np.ones(logits.shape[0])]
        label = np.argmax(logits, axis=1)
        return label


if __name__ == '__main__':
    x = np.array([[1, 2], [1, 2]])
    y = np.array([1, 0])
    print(x * y)
