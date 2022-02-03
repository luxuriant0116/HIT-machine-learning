def gradient_descent(param, grad, lr=0.000001):
    param -= lr * grad
    return param
