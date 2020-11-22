import GPy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../')
import exp.util as util
import exp.kernelized as kernel_layers


class GPyVarImportance(object):
    def __init__(self, X, Y, sig2, opt_sig2=True, opt_kernel_hyperparam=True, lengthscale=1.0, variance=1.0):
        super().__init__()

        self.dim_in = X.shape[1]
        self.kernel = GPy.kern.RBF(input_dim=self.dim_in)
        self.model = GPy.models.GPRegression(X,Y,self.kernel)
        self.model.Gaussian_noise.variance = sig2

        self.opt_sig2 = opt_sig2
        self.opt_kernel_hyperparam = opt_kernel_hyperparam
        
        if not opt_sig2:
            self.model.Gaussian_noise.fix()

        if not opt_kernel_hyperparam:
            self.model.kern.lengthscale.fix()
            self.model.kern.variance.fix()
        
    def train(self):
        self.model.optimize_restarts(num_restarts = 10, verbose=False)
    
    def estimate_psi(self, X, n_samp=1000):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        grad_mu, grad_var = self.model.predict_jacobian(X, full_cov=True) # mean and variance of derivative, (N*, d)
        #psi = np.mean(grad_mu[:,:,0]**2, axis=0)
        psi_mean = np.zeros(self.dim_in)
        psi_var = np.zeros(self.dim_in)
        
        for l in range(self.dim_in):
            grad_samp = np.random.multivariate_normal(grad_mu[:,l,0], grad_var[:,:,l,l], size=n_samp) # (n_samp, N*)
            psi_samp = np.mean(grad_samp**2,1)
            psi_mean[l] = np.mean(psi_samp)
            psi_var[l] = np.var(psi_samp)
            
        return psi_mean, psi_var


class RffVarImportance(object):
    def __init__(self, X, sig2=1., rff_scale=1., rff_dim=1200, seed=None):
        super().__init__()
        self.dim_in = X.shape[1]

        self.model_graph = tf.Graph()
        self.model_sess = tf.Session(graph=self.model_graph)
        self.sig2 = sig2
        self.rff_scale = rff_scale
        self.rff_dim = rff_dim
        self.seed = seed

        # Define model graph.
        with self.model_graph.as_default():
            self.X_tr = tf.placeholder(dtype=tf.float64, shape=[None, self.dim_in])
            self.Y_true = tf.placeholder(dtype=tf.float64, shape=[None, 1])
            self.H_inv = tf.placeholder(dtype=tf.float64, shape=[self.rff_dim, self.rff_dim])
            self.Phi_y = tf.placeholder(dtype=tf.float64, shape=[self.rff_dim, 1])

            self.rff_layer = kernel_layers.RandomFourierFeatures(
                output_dim=self.rff_dim,
                kernel_initializer='gaussian',
                scale=self.rff_scale,
                seed=self.seed)

            ## define model
            self.rff_output = tf.cast(self.rff_layer(self.X_tr) * np.sqrt(2. / self.rff_dim),
                                      dtype=tf.float64)
            self.weight_cov = util.minibatch_woodbury_update(self.rff_output, self.H_inv)
            self.covl_xy = util.minibatch_interaction_update(self.Phi_y, self.rff_output, self.Y_true)

            self.random_feature_weight = self.rff_layer.kernel
            self.random_feature_bias = self.rff_layer.bias

    def train(self, X, Y, batch_size=16, epochs=1):
        ### Training and Evaluation ###
        X_batches = util.split_into_batches(X, batch_size) * epochs
        Y_batches = util.split_into_batches(Y, batch_size) * epochs

        num_steps = X_batches.__len__()
        num_batch = int(num_steps / epochs)

        with self.model_sess as sess:
            sess.run(tf.global_variables_initializer())
            self.RFF_weight, self.RFF_bias = sess.run(
                [self.random_feature_weight, self.random_feature_bias])

            rff_1 = sess.run(self.rff_output,
                             feed_dict={self.X_tr: X_batches[0]})
            weight_cov_val = util.compute_inverse(rff_1, sig_sq=self.sig2 ** 2)
            covl_xy_val = np.matmul(rff_1.T, Y_batches[0])

            for batch_id in range(1, num_batch):
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                try:
                    weight_cov_val, covl_xy_val = sess.run(
                        [self.weight_cov, self.covl_xy],
                        feed_dict={self.X_tr: X_batch,
                                   self.Y_true: Y_batch,
                                   self.H_inv: weight_cov_val,
                                   self.Phi_y: covl_xy_val})
                except:
                    print("\n================================\n"
                          "Problem occurred at Step {}\n"
                          "================================".format(batch_id))

        self.beta = np.matmul(weight_cov_val, covl_xy_val)

        self.weight_cov_val = weight_cov_val
        self.covl_xy_val = covl_xy_val
        self.Sigma_beta = weight_cov_val * self.sig2 ** 2

    def make_rff_feature(self, X):
        return np.sqrt(2. / self.rff_dim) * np.cos(
            np.matmul(X, self.RFF_weight) + self.RFF_bias)

    def predict(self, X):
        D = self.rff_dim
        rff_new = np.sqrt(2. / D) * np.cos(np.matmul(X, self.RFF_weight) +
                                           self.RFF_bias)
        pred_mean = np.matmul(rff_new, self.beta)
        pred_cov = np.matmul(np.matmul(rff_new, self.Sigma_beta), rff_new.T)

        return pred_mean.reshape((-1, 1)), pred_cov

    def estimate_psi(self, X, n_samp=1000):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        nD_mat = np.sin(np.matmul(X, self.RFF_weight) + self.RFF_bias)
        n, d = X.shape
        D = self.RFF_weight.shape[1]
        der_array = np.zeros((n, d, n_samp))
        beta_samp = np.random.multivariate_normal(self.beta,
                                                  self.Sigma_beta,
                                                  size=n_samp).T
        # (D, n_samp)
        for r in range(n):
            cur_mat = np.diag(nD_mat[r, :])
            cur_mat_W = np.matmul(self.RFF_weight, cur_mat)  # (d, D)
            cur_W_beta = np.matmul(cur_mat_W, beta_samp)  # (d, n_samp)
            der_array[r, :, :] = cur_W_beta

        der_array = der_array * np.sqrt(2. / D)
        psi_mean = np.zeros(self.dim_in)
        psi_var = np.zeros(self.dim_in)

        for l in range(self.dim_in):
            grad_samp = der_array[:, l, :].T  # (n_samp, n)
            psi_samp = np.mean(grad_samp ** 2, 1)
            psi_mean[l] = np.mean(psi_samp)
            psi_var[l] = np.var(psi_samp)

        return psi_mean, psi_var

