import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
import numpy as np
import math

from torch.autograd import Variable


def reparam_trick_lognormal(mu, sig2, sample=True):
    '''
    log z ~ N(mu, sig2)
    '''
    if sample:
        # Simulate from normal and exponentiate
        return torch.exp(mu + sig2.sqrt()*torch.randn(sig2.shape))
    else:
        # Mean of lognormal distribution
        return torch.exp(mu + 0.5*sig2)

def local_reparam_trick_normal(a, w_mu, w_sig2, sample=True):
    '''
    Inputs: a
    Weights: w ~ N(w_mu, w_sig2)
    Preactivations: b = a * w ~ N(a * w_mu, a^2 * w_sig2) <-- samples preactivations directly
    '''
    b_mu = F.linear(a, w_mu)

    if sample:
        return b_mu
    else:
        b_sig2 = F.linear(a.pow(2), w_sig2)
        return b_mu + b_sig2.sqrt() * torch.randn(b_sig2.shape)
        

def reparam_trick_normal(a, w_mu, w_sig2, sample=True):
    '''
    Inputs: a
    Weights: w = w_mu + sqrt(w_sig2) * eps, eps ~ N(0,1) <-- samples weights 
    Preactivations: b = a * w
    '''
    if sample:
        # Use sample from varational distribution
        w = w_mu + w_sig2.sqrt() * torch.randn(w_sig2.shape)
    else:
        # Use mean of variational distribution
        w = w_mu

    return F.linear(a, w)

def reparam_trick_noncentered_normal(a, beta_mu, beta_sig2, eta_mu, eta_sig2, sample=True):
    '''
    Inputs: a
    Noncentered weights: beta ~ N(beta_mu, beta_sig2)
    Weight scales: log eta ~ N(eta_mu, eta_sig2)
    Weights: w = eta * beta
    Preactivations: b = a * w

    Assumes beta.shape = (dim_out x dim_in) and eta.shape = (dim_in,)
    '''
    if sample:
        # Use samples from variational distributions
        beta = beta_mu + beta_sig2.sqrt()*torch.randn(beta_sig2.shape)
        eta = torch.exp(eta_mu + eta_sig2.sqrt()*torch.randn(eta_sig2.shape))
    else:
        # Use means of variational distributions
        beta = beta_mu
        eta = torch.exp(eta_mu + 0.5*eta_sig2.sqrt())

    w = eta*beta
    return F.linear(a, w)


def reparam(mu, var):
    eps = torch.FloatTensor(mu.size()).normal_()
    return mu + eps * torch.sqrt(var)

def entropy_invgamma(alpha, beta):
    return torch.sum(alpha + torch.log(beta) + torch.lgamma(alpha) - (1 + alpha) * torch.digamma(alpha))
    #return torch.sum(alpha - torch.log(beta) + torch.lgamma(alpha) + (1-alpha) * torch.digamma(alpha)) # Realized this was incorrect on 6/24/2020

def cross_entropy_invgamma_alt(q_alpha, q_beta, p_alpha, p_beta):
    '''
    H(q, p) = E_q[-Ln p]

    q(x) = InvGamma(q_alpha, q_beta)
    p(x) = InvGamma(p_alpha, p_beta)

    where did I get this from?
    '''
    return -torch.lgamma(q_alpha) - 2 * q_alpha * np.log(q_beta) + (-q_alpha - 1.) * \
        (torch.log(p_beta) - torch.digamma(p_alpha)) - (1. / q_beta ** 2) * (p_alpha / p_beta)

def cross_entropy_invgamma_new(q_alpha, q_beta, p_alpha, p_beta):
    '''
    H(q, p) = E_q[-Ln p]

    q(x) = InvGamma(q_alpha, q_beta)
    p(x) = InvGamma(p_alpha, p_beta)

    Derived on 6/25/2020
    '''
    return torch.sum(-p_alpha*torch.log(p_beta) + torch.lgamma(p_alpha) + (p_alpha+1)*(torch.log(q_beta) - torch.digamma(q_alpha)) + p_beta*q_alpha/q_beta)


def cross_entropy_cond_lognormal_invgamma_new(q_mu, q_sig2, q_alpha, q_beta, p_alpha):
    '''
    H(q, p) = E_q[-Ln p(z | x)] 

    p(z | x) = InvGamma(p_alpha, 1/x)

    q(z,x) = q(z)q(x)
    q(z) = LogNormal(q_mu, q_sig2) 
    q(x) = InvGamma(q_alpha, q_beta)

    Derived on 6/25/2020
    '''

    return torch.sum(p_alpha*(torch.log(q_beta) - torch.digamma(q_alpha)) + torch.lgamma(p_alpha) + (p_alpha+1)*q_mu + q_alpha / q_beta * torch.exp(-q_mu + 0.5*q_sig2))


def cross_entropy_invgamma(alpha_p, beta_p, alpha_q, beta_q):
    '''
    H(p,q) = E_p[-Ln q] = E_p[Ln(p/q)] + E_p[-Ln p] = KL(p,q) + H(p,p)
    '''
    return kl_gammma(alpha_p, beta_p, alpha_q, beta_q) + entropy_invgamma(alpha_p, beta_p)

def kl_gamma(alpha_p, beta_p, alpha_q, beta_q):
    '''
    KL(p,q) = E_p[Ln(p/q)] = -E_p[-Ln p] + E_p[-Ln q] = -H(p,p) + H(p,q)

    p(x) = Gamma(alpha_p, beta_p)
    q(x) = Gamma(alpha_q, beta_q)
    '''
    return (alpha_p - alpha_q)*torch.digamma(alpha_p) - torch.lgamma(alpha_p) + torch.lgamma(alpha_p) \
     + alpha_q*(torch.log(beta_p) - torch.log(beta_q)) + alpha_p * (beta_q - beta_p) / beta_p

def kl_invgamma(alpha_p, beta_p, alpha_q, beta_q):
    '''
    KL(p,q) = E_p[Ln(p/q)] = -E_p[-Ln p] + E_p[-Ln q] = -H(p,p) + H(p, q)

    p(x) = InvGamma(alpha_p, beta_p)
    q(x) = InvGamma(alpha_q, beta_q)
    '''
    return kl_gamma(alpha_p, beta_p, alpha_q, beta_q)

def cross_entropy_lognormal_invgamma(mu_p, sig2_p, alpha_q, beta_q):
    '''
    H(p, q) = E_p[-Ln q]

    p(x) = LogNormal(mu_p, sig2_p)
    q(x) = InvGamma(alpha_q, mu_q)
    '''
    return torch.sum(-alpha_q*torch.log(beta_q) + torch.lgamma(alpha_q) \
        - (-alpha_q-1)*mu_p + beta_q * torch.exp(-2*mu_p + 0.5*sig2_p))
    

def E_tau_lambda(logtau_a_prior, \
                lambda_a_prior, lambda_b_prior, \
                logtau_mu, logtau_logsig2, \
                lambda_a, lambda_b):
    '''E_q[ p(ln tau | lambda)] + E_q[ln p(lambda)]'''

    E_tau_given_lambda = -torch.lgamma(logtau_a_prior) - logtau_a_prior * (torch.log(lambda_b) - torch.digamma(lambda_a)) + (
                            -logtau_a_prior - 1.) * logtau_mu - torch.exp(-logtau_mu + 0.5 * logtau_logsig2.exp()) * (lambda_a /
                                               lambda_b)
    
    E_lambda = -torch.lgamma(lambda_a_prior) - 2 * lambda_a_prior * torch.log(lambda_b_prior) + (-lambda_a_prior - 1.) * (
            torch.log(lambda_b) - torch.digamma(lambda_a)) - (1. / lambda_b_prior ** 2) * (lambda_a / lambda_b)

    return -torch.sum(E_tau_given_lambda) - torch.sum(E_lambda) # BC: Is the sign right?


def EPw_Gaussian(mu, sig2):
    """"int q(z) log p(z) dz, assuming gaussian q(z) and p(z)"""
    wD = mu.shape[0]
    a = 0.5*wD *math.log(2*math.pi)  + 0.5*(torch.sum(mu**2)+torch.sum(sig2)) # BC: Is the sign right?
    return a


def diag_gaussian_entropy(log_std, D):
    return 0.5 * D * (1.0 + math.log(2*math.pi)) + torch.sum(log_std)

def lognormal_entropy(log_std, mu, D):
    return torch.sum(log_std + mu + 0.5) + (D / 2.0) * math.log(2 * math.pi)

def invgamma_entropy(a, b):
    return torch.sum(a + torch.log(b) + torch.lgamma(a) - (1 + a) * torch.digamma(a)) # BC: Is this wrong?


def gen_toy_data(num_train=100, num_val=10, seed=0, x_true=None):
    """
    :param add_noise:
    :param num_tasks:
    :param seed:
    :param num_train:
    :param num_val:
    :param generate_data: if True, generate data. If false only generate task characteristics, i.e., phase and amplitude.
    :return:
    """
    np.random.seed(seed)
    period = 0.1 * np.random.randn() + 1#np.linspace(0.5, 5, np.sqrt(num_tasks))
    phase = np.pi
    amp = 1
    x = np.sort(np.random.uniform(-5, 5, num_train)).astype(np.float32)
    x_val = np.sort(np.random.uniform(-5, 5, num_val)).astype(np.float32)
    x = torch.from_numpy(x).view(-1, 1)
    x_val = torch.from_numpy(x_val).view(-1, 1)
    y_neg = Variable(amp * torch.sin(period * x[x<0] + phase)).view(-1,1) \
            + 0.1 * torch.randn(x[x<0].view(-1,1).size())
    y_pos = Variable(amp * torch.sin(4*period * x[x>=0] + phase)).view(-1,1) \
            + 0.1 * torch.randn(x[x>=0].view(-1,1).size())
    y = torch.cat((y_neg,y_pos))
    #y_val = amp * torch.sin(period * x_val + phase) + 0.1 * torch.randn(x_val.size())
    y_val_neg = amp * torch.sin(period * x_val[x_val<0] + phase).view(-1,1) \
            + 0.1 * torch.randn(x_val[x_val<0].view(-1,1).size())
    y_val_pos = amp * torch.sin(4*period * x_val[x_val>=0] + phase).view(-1,1) \
            + 0.1 * torch.randn(x_val[x_val>=0].view(-1,1).size())
    y_val = torch.cat((y_val_neg,y_val_pos))
    y_scale = torch.std(y)

    if x_true is not None:
        y_true_neg = amp * torch.sin(period * x_true[x_true<0] + phase).view(-1,1)
        y_true_pos = amp * torch.sin(4*period * x_true[x_true>=0] + phase).view(-1,1)
        y_true = torch.cat((y_true_neg,y_true_pos))
        return x, x_val, y, y_val, y_scale, y_true

    return x, x_val, y, y_val, y_scale

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

