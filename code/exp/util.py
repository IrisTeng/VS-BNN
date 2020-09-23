import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import tensorflow as tf
import GPy
import seaborn as sns
import os
import sys

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def standardize_data(original_x_train, original_y_train, 
                     original_x_test=None, original_y_test=None,
                     mean_x=None, std_x=None, mean_y=None, std_y=None):
    if mean_x is None and std_x is None:
        mean_x, std_x = np.mean(original_x_train,0), np.std(original_x_train,0)

    if mean_y is None and std_y is None:
        mean_y, std_y = np.mean(original_y_train,0), np.std(original_y_train,0)

    train_x = (original_x_train - mean_x) / std_x
    train_y = ((original_y_train - mean_y) / std_y).reshape(-1, 1)

    if original_x_test is not None and original_y_test is not None:
        test_x = ((original_x_test - mean_x) / std_x).reshape(-1, 1)
        test_y = ((original_y_test - mean_y) / std_y).reshape(-1, 1)
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y

## toy data

def sin_toy(n_obs, dim_in, sig2=.01, seed=0):
    r = np.random.RandomState(seed)  
    Z = r.uniform(-5,5,(n_obs, dim_in))
    f = lambda x: np.sin(x)
    Y = (f(Z[:,0]) + r.normal(0,sig2,n_obs)).reshape(-1,1)
    
    return Z, Y.reshape(-1,1), sig2

def rbf_toy(n_obs, dim_in, sig2=.01, seed_f=8, seed_zy=0):

    r_f = np.random.RandomState(seed_f)  
    r_zy = np.random.RandomState(seed_zy)  
    
    # sample 1D function
    n_obs_sample = 500
    sig2_f = .1
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    Z = r_f.uniform(-5,5,(n_obs_sample,1))
    f = r_f.multivariate_normal(np.zeros(n_obs_sample), kernel.K(Z), 1).reshape(-1,1)
    Y = f + r_f.normal(0,sig2_f,(n_obs_sample,1))
    
    # fit gp to this function
    m = GPy.models.GPRegression(Z,Y,kernel)
    m.Gaussian_noise.variance = sig2_f
    
    # sample Z data that is actually used (not using same random state)
    # and get Y data by predicting using fitted GP
    Z = r_zy.uniform(-5, 5, (n_obs, dim_in))
    Y,_ = m.predict(Z[:,0].reshape(-1,1)) + r_zy.normal(0,sig2,(n_obs,1))

    return Z, Y.reshape(-1,1), sig2

def sparse_linear_toy(n_obs=200, dim_in=5, sig2=.1**2, n_nonzero=2, seed_x=0, seed_beta=0, seed_sig2=0):
    '''
    Based on Matlab example: https://www.mathworks.com/help/stats/lasso-regularization.html
    '''
    r_x = np.random.RandomState(seed_x) 
    r_beta = np.random.RandomState(seed_beta)  
    r_sig2 = np.random.RandomState(seed_sig2)  

    X = np.zeros((n_obs, dim_in))
    for ii in range(dim_in):
        X[:,ii] = r_x.exponential(1/(ii+1), size=n_obs)

    if n_nonzero==2:
        # use same coefficients as matlab function
        beta = np.zeros(dim_in)
        beta[1] = 2
        beta[3] = -3
    else:
        # otherwise random (integer) coefficients
        beta = r_beta.normal(scale=2,size=dim_in).astype(np.int64)

    Y = X @ beta.reshape(-1,1) + r_sig2.randn(n_obs).reshape(-1,1)*np.sqrt(sig2)

    return X, Y.reshape(-1,1), sig2



def bkmr_toy(n_obs, dim_in, sig2=.5, seed=0):
    '''
    generates toy data by running SimData function from bkrm R package
    requires rpy2 to be installed
    Note: random seed not implemented
    '''
    #r = robjects.r
    bkmr = importr('bkmr') 
    base = importr('base') 

    n = robjects.IntVector([n_obs])
    M = robjects.IntVector([dim_in])
    sigsq_true = robjects.FloatVector([sig2])

    base.set_seed(robjects.FloatVector([seed]))
    #out = bkmr.SimData(n=n, M=M, beta_true=0, sigsq_true = sigsq_true, Zgen="realistic")
    out = bkmr.SimData(n=n, M=M, beta_true=0, sigsq_true = sigsq_true)

    Z = np.ascontiguousarray(out.rx2['Z'])
    Y = np.ascontiguousarray(out.rx2['y'])

    return Z, Y.reshape(-1,1), sig2

def workshop_data(n_obs_samp=None, dim_in_samp=None, dir_in='../data/workshop', seed=0):
    '''
    randomly samples observations and mixture components. 
    by default uses all observations and mixture components.

    raw data has 1003 observations and 18 features for X and Z

    used in bkmr as:
    kmbayes(y=lnLTL_z, Z=lnmixture_z, X=covariates)
    ''' 

    Z=pd.read_csv(os.path.join(dir_in, 'Z.csv'), header=0).to_numpy() # lnmixture_z
    X=pd.read_csv(os.path.join(dir_in, 'X.csv'), header=0).to_numpy() # covariates
    Y=pd.read_csv(os.path.join(dir_in, 'y.csv'), header=0).to_numpy() # lnLTL_z

    # randomly sample observations
    if n_obs_samp is not None:
        r = np.random.RandomState(seed) 
        obs_keep = r.choice(Z.shape[0], n_obs_samp, replace=False)
        Z = Z[obs_keep,:]
        X = X[obs_keep,:]
        Y = Y[obs_keep]

    # randomly sample mixture components
    if dim_in_samp is not None:
        r = np.random.RandomState(seed+1) 
        Z = Z[:,r.choice(Z.shape[1], dim_in_samp, replace=False)]

    return Z, X, Y.reshape(-1,1)
    

def load_data(toy_name, n_obs, dim_in, sig2=None, seed=0):
    '''
    inputs:
        toy_name: 
        n_obs: number of observations
        dim_in: number of inputs
        sig2: observation variance (set to None to use default)
        seed: random seed

    Note that not all toys use all inputs or use all inputs in the same way 
    (e.g. 'workshop' has unknown sig2 and randomly samples features and observations if less than max)

    returns:
        Z: mixture components
        X: covariates
        Y: outcome
        sig2: observation variance
    '''
    if toy_name == 'sin':
        Z, Y, sig2 = sin_toy(n_obs, dim_in, sig2, seed)
        X = None
    elif toy_name == 'rbf':
        Z, Y, sig2 = rbf_toy(n_obs, dim_in, sig2, seed_zy=seed)
        X = None
    elif toy_name == 'sparse_linear':
        Z, Y, sig2 = sparse_linear_toy(n_obs, dim_in, sig2, seed_x=seed)
        X = None
    elif toy_name == 'bkmr':
        Z, Y, sig2 = bkmr_toy(n_obs, dim_in, sig2, seed)
        X = None
    elif toy_name == 'workshop':
        Z, X, Y = workshop_data(n_obs, dim_in, seed=seed)
        sig2 = None
    else:
        print('toy not recognized')

    return Z, X, Y, sig2

## posterior metrics

def rmse(f, f_pred):
    # predictive root mean squared error (RMSE)
    return np.sqrt(np.mean((f - f_pred) ** 2))


def picp(f, f_pred_lb, f_pred_ub):
    # prediction interval coverage (PICP)
    return np.mean(np.logical_and(f >= f_pred_lb, f <= f_pred_ub))


def mpiw(f_pred_lb, f_pred_ub):
    # mean prediction interval width (MPIW)
    return np.mean(f_pred_ub - f_pred_lb)


def test_log_likelihood(mean, cov, test_y):
    return multivariate_normal.logpdf(test_y.reshape(-1), mean.reshape(-1), cov)


def minibatch_woodbury_update(X, H_inv):
    """Minibatch update of linear regression posterior covariance
    using Woodbury matrix identity.

    inv(H + X^T X) = H_inv - H_inv X^T inv(I + X H_inv X^T) X H_inv

    Args:
        X: (tf.Tensor) A M x K matrix of batched observation.
        H_inv: (tf.Tensor) A K x K matrix of posterior covariance of rff coefficients.


    Returns:
        H_new: (tf.Tensor) K x K covariance matrix after update.
    """
    batch_size = tf.shape(X)[0]

    M0 = tf.eye(batch_size, dtype=tf.float64) + tf.matmul(X, tf.matmul(H_inv, X, transpose_b=True))
    M = tf.matrix_inverse(M0)
    B = tf.matmul(X, H_inv)
    H_new = H_inv - tf.matmul(B, tf.matmul(M, B), transpose_a=True)

    return H_new


def minibatch_interaction_update(Phi_y, rff_output, Y_batch):
    return Phi_y + tf.matmul(rff_output, Y_batch, transpose_a=True)


def compute_inverse(X, sig_sq=1):
    return np.linalg.inv(np.matmul(X.T, X) + sig_sq * np.identity(X.shape[1]))


def split_into_batches(X, batch_size):
    return [X[i:i + batch_size] for i in range(0, len(X), batch_size)]

## plotting

'''
def plot_results_grid(data, dim_in_list, n_obs_list, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,data.shape[2], figsize=(16,4), sharex=True, sharey=True)
    
    vmax = np.nanmax(data)
    ax[0].set_ylabel('num. obs.')
    for i in range(data.shape[2]):
        pcm=ax[i].imshow(data[:,:,i],vmin=0, vmax=vmax)
        ax[i].set_title('X_%d'%i)
        ax[i].set_xlabel('num. inputs')

    plt.xticks(np.arange(len(dim_in_list)), labels=dim_in_list)

    plt.yticks(np.arange(len(n_obs_list)), labels=n_obs_list)

    fig.colorbar(pcm, ax=ax[:], shrink=0.6)

    return fig, ax 
'''

def plot_results_grid(data, x_vals, y_vals, x_lab, y_lab, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,data.shape[2], figsize=(16,4), sharex=True, sharey=True)
    
    vmax = np.nanmax(data)
    ax[0].set_ylabel(y_lab)
    for i in range(data.shape[2]):
        pcm=ax[i].imshow(data[:,:,i],vmin=0, vmax=vmax)
        ax[i].set_title('X_%d'%i)
        ax[i].set_xlabel(x_lab)

    plt.xticks(np.arange(len(x_vals)), labels=x_vals)

    plt.yticks(np.arange(len(y_vals)), labels=y_vals)

    fig.colorbar(pcm, ax=ax[:], shrink=0.6)

    return fig, ax 


def plot_results_dist(data, dim_in, n_obs_list, data_true=None, fig=None, ax=None):
    '''
    data:   (obs x rep x input) <-- data for models with same input dimension
    data_true:   (input)

    For each combination of n_obs and variable_idx, plots distribution of psi_mean over reps.

    '''

    if fig is None and ax is None:
        fig,ax = plt.subplots(len(n_obs_list),1,figsize=(dim_in*2,16),sharex=True,sharey=True)
    if len(n_obs_list)==1: ax = [ax]

    ax[-1].set_xlabel('variable')
    
    if data_true is not None:
        ax[0].legend([plt.Line2D([0], [0], color='red')],['"truth"'])

    n_rep = data.shape[1]
    for j, n_obs in enumerate(n_obs_list):
        if n_rep>1:
            sns.violinplot(data=data[j,:,:], ax=ax[j])
        else:
            ax[j].bar(np.arange(dim_in), data[j,0,:dim_in])
        for i in range(dim_in):
            ax[j].set_ylabel('%d obs.' % n_obs)
            if data_true is not None:
                ax[j].plot(np.array([i-.25,i+.25]),np.array([data_true[i],data_true[i]]),'red')
        
    ax[0].set_xlim(-.5,dim_in-.5)
    return fig, ax


def plot_slice(f_sampler, x, y, quantile=.5, dim=0, n_samp=500, ax=None):
    '''

    x: (N,D) training inputs
    y: (N,1) or (N,) training outputs
    quantile: Quantile of fixed x variables to use in plot
    dim: dimension of x to plot on x-axis

    Everything should be numpy
    '''

    if ax is None:
        fig, ax = plt.subplots()

    # x-axis
    midx = (x[:,dim].min() + x[:,dim].max())/2
    dx = x[:,dim].max() - x[:,dim].min()
    x_plot = np.linspace(midx - .75*dx, midx + .75*dx, 100)

    x_plot_all = np.quantile(x, q=quantile, axis=0)*np.ones((x_plot.shape[0], x.shape[1]))
    x_plot_all[:, dim] = x_plot

    # sample from model
    f_samp_plot = np.zeros((n_samp, x_plot.shape[0]))
    for i in range(n_samp):
        f_samp_plot[i,:] = f_sampler(x_plot_all).reshape(-1)

    # plot
    ax.scatter(x[:,dim], y) # training data
    ax.plot(x_plot, np.mean(f_samp_plot, 0)) # posterior mean
    for q in [.025, .05, .1]:
        ci = np.quantile(f_samp_plot, [q, 1-q], axis=0)
        ax.fill_between(x_plot_all[:,dim].reshape(-1), ci[0,:], ci[1,:], alpha=.1, color='blue')


def plot_slices(f_sampler, x, y, quantile=.5, n_samp=500, figsize=(4,4)):  
    dim_in = x.shape[1]  
    fig, ax = plt.subplots(1, dim_in, figsize=figsize, sharey=True)

    if dim_in>1:
        ax[0].set_ylabel('y')
    else:
        ax.set_ylabel('y')

    fig.suptitle("Posterior predictive (1d slices)")
    for dim in range(dim_in):
        ax_dim = ax[dim] if dim_in>1 else ax 
        plot_slice(f_sampler, x, y, quantile=quantile, dim=dim, n_samp=n_samp, ax=ax_dim)
        ax_dim.set_xlabel('x'+str(dim))
    return fig, ax

## miscellaneous


def arrange_full(start, stop, step): 
    # so, e.g., np.arrange(1,10,1) returns [1,2,...,10] instead of [1,2,...,9]
    return np.arange(start, stop+((stop-start)%step==0), step) 

def resid_linear_model(X, Y):
    '''
    Regress X on Y and return residual
    '''
    beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return Y - X @ beta_hat



