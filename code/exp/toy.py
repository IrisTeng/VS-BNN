from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


__all__ = [
    'x3_toy',
    'x3_gap_toy',
    'sin_toy',
]


class toy_dataset(object):
    def __init__(self, name=''):
        self.name = name

    def train_samples(self):
        raise NotImplementedError

    def test_samples(self):
        raise NotImplementedError


class rbf_toy(toy_dataset):
    def __init__(self, variance=1.0, lengthscale=0.4, name='rbf'):
        self.variance = variance
        self.lengthscale = lengthscale
        self.x_min = 0
        self.x_max = 1
        #self.y_std = np.sqrt(.01)
        self.y_std = np.sqrt(.001)
        super(rbf_toy, self).__init__(name)

    def sample_f(self, n_train_max, n_test, seed=0):

        X_train_1 = np.random.uniform(0, 0.4, (n_train_max // 2, 1))
        X_train_2 = np.random.uniform(0.6, 1, (n_train_max // 2, 1))
        self.X_train_max = np.concatenate([X_train_1, X_train_2], axis=0)
        # self.X_train_max = np.random.uniform(self.x_min, self.x_max, (n_train_max, 1))
        self.X_test = np.linspace(self.x_min, self.x_max, n_test).reshape(-1,1)
        self.X = np.concatenate([self.X_train_max, self.X_test], axis=0)
        K = self.variance*np.exp(-0.5*(self.X.reshape(-1,1)-self.X.reshape(1,-1))**2 / self.lengthscale**2)
        f = np.random.multivariate_normal(np.zeros(self.X.shape[0]), K)
        self.f_train_max = f[:n_train_max].reshape(-1,1)
        self.f_test = f[-n_test:].reshape(-1,1)


    def train_samples(self, n_data=20, seed=0):
        np.random.seed(seed)
        idx_sample = np.random.choice(self.X_train_max.shape[0], n_data)
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = self.f_train_max[idx_sample] + epsilon
        return self.X_train_max[idx_sample], y_train

    def test_samples(self, n_data=None, seed=0):
        np.random.seed(seed)
        epsilon = np.random.normal(0, self.y_std, (self.f_test.shape[0], 1))
        y_test = self.f_test + epsilon
        return self.X_test, y_test
