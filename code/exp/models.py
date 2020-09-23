import GPy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys
from math import pi, log, sqrt

sys.path.append('../')
import exp.util as util
import exp.kernelized as kernel_layers

# for horseshoe model
import torch
import torch.nn.functional as F
sys.path.append('./horseshoe')
from horseshoe.networks import RffHs, Rff
from horseshoe.networks import train as train_rffhs
torch.set_default_dtype(torch.float64)

# for bkmr
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

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

    def sample_f_post(self, x):
        # inputs and outputs are numpy arrays
        return self.model.posterior_samples_f(x, size=1)

class RffVarImportance(object):
    def __init__(self, X):
        super().__init__()
        self.dim_in = X.shape[1]


    def train(self, X, Y, sig2, rff_dim=1200, batch_size=16, epochs=16):

        model_graph = tf.Graph()
        model_sess = tf.Session(graph=model_graph)

        with model_graph.as_default():
            X_tr = tf.placeholder(dtype=tf.float64, shape=[None, self.dim_in])
            Y_true = tf.placeholder(dtype=tf.float64, shape=[None, 1])
            H_inv = tf.placeholder(dtype=tf.float64, shape=[rff_dim, rff_dim])
            Phi_y = tf.placeholder(dtype=tf.float64, shape=[rff_dim, 1])

            rff_layer = kernel_layers.RandomFourierFeatures(output_dim=rff_dim,
                                                            kernel_initializer='gaussian',
                                                            trainable=True)

            ## define model
            rff_output = tf.cast(rff_layer(X_tr) * np.sqrt(2. / rff_dim), dtype=tf.float64)

            weight_cov = util.minibatch_woodbury_update(rff_output, H_inv)

            covl_xy = util.minibatch_interaction_update(Phi_y, rff_output, Y_true)

            random_feature_weight = rff_layer.kernel

            random_feature_bias = rff_layer.bias

        ### Training and Evaluation ###
        X_batches = util.split_into_batches(X, batch_size) * epochs
        Y_batches = util.split_into_batches(Y, batch_size) * epochs

        num_steps = X_batches.__len__()
        num_batch = int(num_steps / epochs)

        with model_sess as sess:
            sess.run(tf.global_variables_initializer())

            rff_1 = sess.run(rff_output, feed_dict={X_tr: X_batches[0]})
            weight_cov_val = util.compute_inverse(rff_1, sig_sq=sig2**2)
            covl_xy_val = np.matmul(rff_1.T, Y_batches[0])

            rff_weight, rff_bias = sess.run([random_feature_weight, random_feature_bias])

            for batch_id in range(1, num_batch):
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                try:
                    weight_cov_val, covl_xy_val = sess.run([weight_cov, covl_xy],
                                                           feed_dict={X_tr: X_batch,
                                                                      Y_true: Y_batch,
                                                                      H_inv: weight_cov_val,
                                                                      Phi_y: covl_xy_val})
                except:
                    print("\n================================\n"
                          "Problem occurred at Step {}\n"
                          "================================".format(batch_id))

        self.beta = np.matmul(weight_cov_val, covl_xy_val)[:,0]

        self.Sigma_beta = weight_cov_val * sig2**2

        self.RFF_weight = rff_weight  # (d, D)

        self.RFF_bias = rff_bias  # (D, )



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
        beta_samp = np.random.multivariate_normal(self.beta, self.Sigma_beta, size=n_samp).T
        # (D, n_samp)
        for r in range(n):
            cur_mat = np.diag(nD_mat[r,:])
            cur_mat_W = np.matmul(self.RFF_weight, cur_mat)  # (d, D)
            cur_W_beta = np.matmul(cur_mat_W, beta_samp)  # (d, n_samp)
            der_array[r,:,:] = cur_W_beta

        der_array = der_array * np.sqrt(2. / D)
        psi_mean = np.zeros(self.dim_in)
        psi_var = np.zeros(self.dim_in)

        for l in range(self.dim_in):
            grad_samp = der_array[:,l,:].T  # (n_samp, n)
            psi_samp = np.mean(grad_samp ** 2, 1)
            psi_mean[l] = np.mean(psi_samp)
            psi_var[l] = np.var(psi_samp)

        return psi_mean, psi_var

    def sample_f_post(self, x):
        # inputs and outputs are numpy arrays
        D = self.RFF_weight.shape[1]
        phi = np.sqrt(2. / D) * np.cos(np.matmul(x, self.RFF_weight) + self.RFF_bias)
        beta_samp = np.random.multivariate_normal(self.beta, self.Sigma_beta, size=1).T
        return np.matmul(phi, beta_samp)


class RffVarImportancePytorch(object):
    def __init__(self, X, Y, sig2_inv, dim_in=1, dim_out=1, dim_hidden=50):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        self.model = Rff(dim_in=X.shape[1], dim_out=Y.shape[1], sig2_inv=sig2_inv, dim_hidden=dim_hidden)
        
    def train(self):
        self.model.fixed_point_updates(self.X, self.Y)

    def estimate_psi(self, X=None, n_samp=1000):
        '''
        Use automatic gradients

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        X = torch.from_numpy(X)
        X.requires_grad = True

        psi_mean = np.zeros(self.model.dim_in)
        psi_var = np.zeros(self.model.dim_in)

        psi_samp = torch.zeros((n_samp, self.model.dim_in))
        for i in range(n_samp):

            f = self.model(X, weights_type='sample_post')
            torch.sum(f).backward()
            psi_samp[i,:] = torch.mean(X.grad**2,0)
            X.grad.zero_()

        psi_mean = torch.mean(psi_samp, 0)
        psi_var = torch.var(psi_samp, 0)

        return psi_mean.numpy(), psi_var.numpy()


    def sample_f_post(self, x):
        # inputs and outputs are numpy arrays
        return self.model(torch.from_numpy(x), weights_type='sample_post').detach().numpy()

class RffHsVarImportance(object):
    def __init__(self, 
        X, Y, \
        sig2_inv, \
        dim_in=1, dim_out=1, dim_hidden=50, \
        infer_noise=False, sig2_inv_alpha_prior=None, sig2_inv_beta_prior=None, \
        linear_term=False, linear_dim_in=None,\
        **model_kwargs):
        super().__init__()

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

        self.model = RffHs(dim_in=X.shape[1], dim_out=Y.shape[1], dim_hidden=dim_hidden, \
            infer_noise=infer_noise, sig2_inv=sig2_inv, sig2_inv_alpha_prior=sig2_inv_alpha_prior, sig2_inv_beta_prior=sig2_inv_beta_prior, \
            linear_term=linear_term, linear_dim_in=linear_dim_in, **model_kwargs)

        
    def train(self, lr=.001, n_epochs=100):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.reinit_parameters(self.X, self.Y, n_reinit=10) 
        elbo = -train_rffhs(self.model, optimizer, self.X, self.Y, n_epochs=n_epochs, n_rep_opt=25, print_freq=1)


    def estimate_psi(self, X=None, n_samp=1000):
        '''
        Use automatic gradients

        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        X = torch.from_numpy(X)
        X.requires_grad = True

        psi_mean = np.zeros(self.model.dim_in)
        psi_var = np.zeros(self.model.dim_in)

        psi_samp = torch.zeros((n_samp, self.model.dim_in))
        for i in range(n_samp):

            #f = self.model(X, weights_type_layer_in='sample_post', weights_type_layer_out='sample_post')
            f = self.model.sample_posterior_predictive(x_test=X, x_train=self.X, y_train=self.Y)
            
            torch.sum(f).backward()
            psi_samp[i,:] = torch.mean(X.grad**2,0)
            X.grad.zero_()

        psi_mean = torch.mean(psi_samp, 0)
        psi_var = torch.var(psi_samp, 0)

        return psi_mean.numpy(), psi_var.numpy()


    def estimate_psi2(self, X=None, n_samp=1000):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''
        with torch.no_grad():

            #breakpoint()

            dist_nu = torch.distributions.log_normal.LogNormal(loc=self.model.layer_in.lognu_mu, 
                                                               scale=self.model.layer_in.lognu_logsig2.exp().sqrt())
            
            dist_eta = torch.distributions.log_normal.LogNormal(loc=self.model.layer_in.logeta_mu, 
                                                                scale=self.model.layer_in.logeta_logsig2.exp().sqrt())
            
            dist_beta = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.model.layer_out.mu, 
                                                                                   covariance_matrix=self.model.layer_out.sig2)

            psi_mean = np.zeros(self.model.dim_in)
            psi_var = np.zeros(self.model.dim_in)
            for l in range(self.model.dim_in):

                # TO DO: replace loop for efficiency
                grad_samp = torch.zeros((n_samp, X.shape[0]))
                for i in range(n_samp):

                    samp_nu = dist_nu.sample()
                    samp_eta = dist_eta.sample()
                    samp_beta = dist_beta.sample()

                    nu_eta_w = samp_nu*samp_eta*self.model.layer_in.w

                    grad_samp[i,:] = (-sqrt(2/self.model.dim_out) \
                                     *torch.sin(F.linear(torch.from_numpy(X), nu_eta_w, self.model.layer_in.b)) \
                                     @ torch.diag(nu_eta_w[:,l]) \
                                     @ samp_beta.T).reshape(-1)

                psi_samp = torch.mean(grad_samp**2,1)
                psi_mean[l] = torch.mean(psi_samp)
                psi_var[l] = torch.var(psi_samp)
        breakpoint()

        return psi_mean.numpy(), psi_var.numpy()

    def dist_scales(self):
        '''
        returns mean and variance parameters of input-specific scale eta (not log eta)
        '''

        logeta_mu = self.model.layer_in.logeta_mu.detach()
        logeta_sig2 = self.model.layer_in.logeta_logsig2.exp().detach()

        eta_mu = torch.exp(logeta_mu + logeta_sig2/2)
        eta_sig2 = (torch.exp(logeta_sig2)-1)*torch.exp(2*logeta_mu+logeta_sig2)

        return eta_mu.numpy(), eta_sig2.numpy()

    #def sample_f_post(self, x):
    #    # inputs and outputs are numpy arrays
    #    return self.model(torch.from_numpy(x), weights_type_layer_in='sample_post', weights_type_layer_out='sample_post').detach().numpy()

    def sample_f_post(self, x_test):
        # inputs and outputs are numpy arrays
        with torch.no_grad():
            return self.model.sample_posterior_predictive(x_test=torch.from_numpy(x_test), x_train=self.X, y_train=self.Y).numpy().reshape(-1)



class BKMRVarImportance(object):
    def __init__(self, Z, Y, sig2):
        super().__init__()

        self.bkmr = importr('bkmr') 
        self.base = importr('base') 
        self.sigsq_true = robjects.FloatVector([sig2])

        Zvec = robjects.FloatVector(Z.reshape(-1))
        self.Z = robjects.r.matrix(Zvec, nrow=Z.shape[0], ncol=Z.shape[1], byrow=True)

        Yvec = robjects.FloatVector(Y.reshape(-1))
        self.Y = robjects.r.matrix(Yvec, nrow=Y.shape[0], ncol=Y.shape[1], byrow=True)
        
    def train(self, n_samp=5000):

        self.base.set_seed(robjects.FloatVector([1]))
        self.fitkm = self.bkmr.kmbayes(y = self.Y, Z = self.Z, iter = robjects.IntVector([n_samp]), verbose = robjects.vectors.BoolVector([False]), varsel = robjects.vectors.BoolVector([True]))

    def estimate_psi(self, X=None, n_samp=None):
        '''
        estimates mean and variance of variable importance psi
        X:  inputs to evaluate gradient
        n_samp:  number of MC samples
        '''

        out = self.bkmr.ExtractPIPs(self.fitkm)
        
        pip = np.ascontiguousarray(out.rx2('PIP'))
        pip_var = np.zeros_like(pip)
        print('pip:', pip)
        return pip, pip_var
