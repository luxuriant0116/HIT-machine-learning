import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(0)


class KMeans:
    def __init__(self, num_class, data):
        self.data = data
        self.k = num_class
        self.init = self.initialize_center()
        self.center = self.init
        self.cluster = np.zeros(self.data.shape[0])

    @property
    def _center(self):
        return self.center

    @property
    def _cluster(self):
        return np.c_[self.data, self.cluster]

    def initialize_center(self):
        random_id = np.random.choice(self.data.shape[0], size=self.k, replace=False)

        return self.data[random_id]

    def update_center(self):
        num_sample = self.data.shape[0]
        x_dim = self.data.shape[1]
        center = np.repeat(self.center, num_sample, axis=0).reshape((self.k, num_sample, -1))
        distance = self.compute_distance(self.data, center)
        prev_cluster = self.cluster.view()
        self.cluster = np.argmin(distance, axis=0)

        self.center = np.zeros((self.k, x_dim))
        total = np.zeros(self.k)
        for i in range(num_sample):
            self.center[self.cluster[i]] += self.data[i]
            total[self.cluster[i]] += 1
        total = np.repeat(total, x_dim).reshape((self.k, x_dim))
        self.center /= total

        return (prev_cluster == self.cluster).all()

    def forward(self, epoch):
        for e in range(epoch):
            flag = self.update_center()
            # self.visualize()
            if flag:
                break

    def visualize(self):
        plt.title('K-Means')
        color_list = ['gold', 'palegreen', 'turquoise', 'skyblue', 'mediumpurple', 'hotpink', 'tomato', 'orange', ]
        x, y = [], []
        for i in range(self.k):
            x.append([])
            y.append([])
        for i in range(self.data.shape[0]):
            x[self.cluster[i]].append(self.data[i][0])
            y[self.cluster[i]].append(self.data[i][1])
        for i in range(self.k):
            plt.plot(x[i], y[i], 'ro', color=color_list[i])
        for center in self.center:
            plt.plot(center[0], center[1], marker='+', color='blue')
        # for center_init in self.init:
        #     plt.plot(center_init[0], center_init[1], marker='*', color='red')

        plt.show()

    @staticmethod
    def compute_distance(a, b):
        return np.sqrt(np.sum(np.square(a - b), axis=-1))


if __name__ == '__main__':
    pass
