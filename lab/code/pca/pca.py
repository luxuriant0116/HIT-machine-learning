import numpy as np


class PCA:
    def __init__(self, data, dim):
        self.raw_data = data
        self.mean_data = np.mean(data, axis=0)
        self.data = data - self.mean_data
        self.pre_dim = data.shape[1]
        self.target_dim = dim

    def compute_cov_matrix(self):
        cov = np.cov(self.data, rowvar=False)
        return cov

    def compute_direction(self):
        cov = self.compute_cov_matrix()
        cov.astype(np.float64)
        feature_value, feature_vector = np.linalg.eig(cov)
        # column feature_vector[:, i] is eigenvector corresponding to the eigenvalue feature_value[i].
        index = np.argsort(feature_value)[::-1]
        return feature_vector[:, index[:self.target_dim]]

    def reduce_dim(self):
        u = self.compute_direction()
        u = np.array(u, dtype=np.float64)
        target = np.dot(self.data, u)
        return target

    def recover_dim(self):
        u = self.compute_direction()
        u = np.array(u, dtype=np.float64)
        z = self.reduce_dim()
        recover = np.dot(z, u.T) + self.mean_data
        return recover
