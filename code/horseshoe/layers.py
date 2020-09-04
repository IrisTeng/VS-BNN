import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.beta import Beta
from distributions import LogitNormal

import numpy as np
from math import pi, log, sqrt
import numpy as np

import util


class LinearLayer(nn.Module):
    """
    Linear model layer

    Assumes 1d outputs for now
    """
    def __init__(self, dim_in, prior_mu=0., prior_sig2=1., sig2_y=.1, **kwargs):
        super(LinearLayer, self).__init__()

        self.dim_in = dim_in
        self.prior_mu = prior_mu
        self.prior_sig2 = prior_sig2
        self.sig2_y = sig2_y

        self.register_buffer('mu', torch.empty(1, dim_in))
        self.register_buffer('sig2', torch.empty(dim_in, dim_in))

        self.init_parameters()
        self.sample_weights(store=True)

    def init_parameters(self):
        self.mu.normal_(0,1)
        self.sig2 = self.prior_sig2*torch.eye(self.dim_in)

    def sample_weights(self, store=False, prior=False):
        if prior:
            mu = self.prior_mu*torch.ones(1, self.dim_in)
            sig2 = self.prior_sig2*torch.eye(self.dim_in)
        else:
            mu = self.mu
            sig2 = self.sig2

        try:
            m = MultivariateNormal(mu, sig2)
            w = m.sample()
        except:
            print('Using np.random.multivariate_normal')
            w = torch.from_numpy(np.random.multivariate_normal(mu.reshape(-1).numpy(), sig2.numpy())).float()
        
        if store: self.w = w
        return w
    
    def fixed_point_updates(self, x, y):
        # conjugate updates

        prior_sig2inv_mat = 1/self.prior_sig2*torch.eye(self.dim_in)
        prior_mu_vec = torch.ones(self.dim_in,1)*self.prior_mu

        try:
            self.sig2 = torch.pinverse(prior_sig2inv_mat + x.transpose(0,1)@x/self.sig2_y)
            self.mu = (self.sig2 @ (prior_sig2inv_mat@prior_mu_vec + x.transpose(0,1)@y/self.sig2_y)).transpose(0,1)
        except:
            print('Error: cannot update LinearLayer, skipping update')
            pass
        
    def forward(self, x, weights_type='sample_post'):
        '''
        weights_type = 'mean': 
        weights_type = 'sample': 
        weights_type = 'stored': 
        '''
        if weights_type == 'mean_prior':
            w = self.prior_mu

        elif weights_type == 'mean_post':
            w = self.mu

        elif weights_type == 'sample_prior':
            w = self.sample_weights(store=False, prior=True)

        elif weights_type == 'sample_post':
            w = self.sample_weights(store=False, prior=False)

        elif weights_type == 'stored':
            w = self.w

        return F.linear(x, w)


class RffLayer(nn.Module):
    """
    Random features layer
    """
    def __init__(self, dim_in, dim_out, **kwargs):
        super(RffLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.register_buffer('w', torch.empty(dim_out, dim_in))
        self.register_buffer('b', torch.empty(dim_out))

        self.sample_features()

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1)
        self.b.uniform_(0, 2*pi)

    def forward(self, x):
        return sqrt(2/self.dim_in)*torch.cos(F.linear(x, self.w, self.b))


class RffHsLayer(nn.Module):
    """
    Random features with horseshoe
    """
    def __init__(self, dim_in, dim_out, b_g=1, b_0=1, infer_nu=True, nu=None, **kwargs):
        super(RffHsLayer, self).__init__()
        '''
        nu is the layer scale, eta is the unit-specific scale

        b_g: nu ~ C^+(b_g)
        b_0: eta ~ C^+(b_0)
        infer_nu: whether to infer nu
        nu: fixed value for nu (only used if infer_nu=True)
        '''

        self.b_g = b_g
        self.b_0 = b_0

        self.infer_nu = infer_nu

        ### architecture
        self.dim_in = dim_in
        self.dim_out = dim_out

        # noncentered random weights
        self.register_buffer('w', torch.empty(dim_out, dim_in))
        self.register_buffer('b', torch.empty(dim_out))

        ### variational parameters

        # layer scales
        if self.infer_nu:
            self.lognu_mu = nn.Parameter(torch.empty(1))
            self.lognu_logsig2 = nn.Parameter(torch.empty(1))

            self.register_buffer('vtheta_a', torch.empty(1))
            self.register_buffer('vtheta_b', torch.empty(1))
        
        else:
            self.register_buffer('nu', torch.tensor(nu))

        # input unit scales
        self.logeta_mu = nn.Parameter(torch.empty(self.dim_in))
        self.logeta_logsig2 = nn.Parameter(torch.empty(self.dim_in))

        self.register_buffer('psi_a', torch.empty(self.dim_in))
        self.register_buffer('psi_b', torch.empty(self.dim_in))

        ### priors

        # layer scales
        if self.infer_nu:
            self.lognu_a_prior = torch.tensor(0.5) # shouldn't this be called "nu_a_prior"?

            self.vtheta_a_prior = torch.tensor(0.5)
            self.vtheta_b_prior = torch.tensor(1/(b_g**2))

            self.nu_dist_prior = HalfCauchy(scale=b_g*torch.ones(1))

        # input unit scales
        self.logeta_a_prior = torch.tensor(0.5) # shouldn't this be called "eta_a_prior"?

        self.psi_a_prior = torch.tensor(0.5)
        self.psi_b_prior = torch.tensor(1/(b_0**2))

        self.eta_dist_prior = HalfCauchy(scale=b_0*torch.ones(self.dim_in))

        ### init params
        self.init_parameters()
        self.sample_features()

    def init_parameters(self):
        # initialize variational parameters

        # layer scale
        if self.infer_nu:
            #self.lognu_mu.data = torch.log(np.abs(self.b_g*torch.randn(1) / torch.randn(1)))
            self.lognu_mu.data = torch.log(1+1e-2*torch.randn(self.lognu_mu.shape))
            self.lognu_logsig2.data.normal_(-9, 1e-2)

        # input unit scales
        #self.logeta_mu.data = torch.log(np.abs(self.b_0*torch.randn(self.logeta_mu.shape) / torch.randn(self.logeta_mu.shape)))
        self.logeta_mu.data = torch.log(1+1e-2*torch.randn(self.logeta_mu.shape))
        self.logeta_logsig2.data.normal_(-9, 1e-2)

        self.fixed_point_updates()

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1)
        self.b.uniform_(0, 2*pi)

    def forward(self, x, weights_type='sample_post'):
        '''
        Note: 'mean_prior' doesn't exist (since HalfCauchy)
        '''

        if weights_type == 'mean_post':
            if self.infer_nu:
                nu = util.reparam_trick_lognormal(self.lognu_mu, self.lognu_logsig2.exp(), sample=False)
            else:
                nu = self.nu
            eta = util.reparam_trick_lognormal(self.logeta_mu, self.logeta_logsig2.exp(), sample=False)

        elif weights_type == 'sample_prior':
            if self.infer_nu:
                nu = self.nu_dist_prior.sample()
            else:
                nu = self.nu
            eta = self.eta_dist_prior.sample()

        elif weights_type == 'sample_post':
            # regular reparameterization trick on scales
            if self.infer_nu:
                nu = util.reparam_trick_lognormal(self.lognu_mu, self.lognu_logsig2.exp(), sample=True)
            else:
                nu = self.nu
            eta = util.reparam_trick_lognormal(self.logeta_mu, self.logeta_logsig2.exp(), sample=True)

        #nu=1. # TEMP FOR TESTING
        #eta=1. # TEMP FOR TESTING
        #return F.relu(F.linear(x, nu*eta*self.w, self.b)) # relu features TEMP FOR TESTING

        return sqrt(2/self.dim_out)*torch.cos(F.linear(x, nu*eta*self.w, self.b))

    def fixed_point_updates(self):

        # layer scale
        if self.infer_nu:
            self.vtheta_a = torch.tensor([1.]) # torch.ones((self.dim_out,)) # could do this in init
            self.vtheta_b = torch.exp(-self.lognu_mu + 0.5*self.lognu_logsig2.exp()) + 1/(self.b_g**2)

        # input unit scales
        self.psi_a = torch.ones((self.dim_in,)) # could do this in init
        self.psi_b = torch.exp(-self.logeta_mu + 0.5*self.logeta_logsig2.exp()) + 1/(self.b_0**2)


    def entropy(self):
        '''H(q,q) = E_q[ -q(z)]'''
        e=0

        # layer scale
        if self.infer_nu:
            e += util.lognormal_entropy(self.lognu_logsig2.exp().sqrt().log(), self.lognu_mu, 1)
            e += util.entropy_invgamma(self.vtheta_a, self.vtheta_b)

        # input unit scales
        e += util.lognormal_entropy(self.logeta_logsig2.exp().sqrt().log(), self.logeta_mu, self.dim_in)
        e += util.entropy_invgamma(self.psi_a, self.psi_b)

        return e

    def cross_entropy(self):
        '''H(q,p) = E_q[ -p(z) ]'''

        '''
        ce = 0

        # layer scale
        ce += util.E_tau_lambda(self.lognu_a_prior, \
            self.vtheta_a_prior, self.vtheta_b_prior, \
            self.lognu_mu, self.lognu_logsig2, \
            self.vtheta_a, self.vtheta_b)

        # input unit scales
        ce += util.E_tau_lambda(self.logeta_a_prior, \
            self.psi_a_prior, self.psi_b_prior, \
            self.logeta_mu, self.logeta_logsig2, \
            self.psi_a, self.psi_b)

        return ce
        '''

        ce = 0

        if self.infer_nu:
            ce += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.lognu_mu, q_sig2=self.lognu_logsig2.exp(), q_alpha=self.vtheta_a, q_beta=self.vtheta_b, p_alpha=self.lognu_a_prior) 
            ce += util.cross_entropy_invgamma_new(q_alpha=self.vtheta_a, q_beta=self.vtheta_b, p_alpha=self.vtheta_a_prior, p_beta=self.vtheta_b_prior)

        ce += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.logeta_mu, q_sig2=self.logeta_logsig2.exp(), q_alpha=self.psi_a, q_beta=self.psi_b, p_alpha=self.logeta_a_prior) 
        ce += util.cross_entropy_invgamma_new(q_alpha=self.psi_a, q_beta=self.psi_b, p_alpha=self.psi_a_prior, p_beta=self.psi_b_prior)

        return ce

    def kl_divergence(self):
        return self.cross_entropy() - self.entropy()



class RffHsLayer2(nn.Module):
    """
    Random features with horseshoe

    INCLUDES INDICATOR PARAMETER S
    """
    def __init__(self, dim_in, dim_out, b_g=1, b_0=1, infer_nu=True, nu=None, **kwargs):
        super(RffHsLayer2, self).__init__()
        '''
        nu is the layer scale, eta is the unit-specific scale

        b_g: nu ~ C^+(b_g)
        b_0: eta ~ C^+(b_0)
        infer_nu: whether to infer nu
        nu: fixed value for nu (only used if infer_nu=True)
        '''

        self.b_g = b_g
        self.b_0 = b_0

        self.infer_nu = infer_nu

        ### architecture
        self.dim_in = dim_in
        self.dim_out = dim_out

        # noncentered random weights
        self.register_buffer('w', torch.empty(dim_out, dim_in))
        self.register_buffer('b', torch.empty(dim_out))

        ### variational parameters

        # layer scales
        if self.infer_nu:
            self.lognu_mu = nn.Parameter(torch.empty(1))
            self.lognu_logsig2 = nn.Parameter(torch.empty(1))

            self.register_buffer('vtheta_a', torch.empty(1))
            self.register_buffer('vtheta_b', torch.empty(1))
        
        else:
            self.register_buffer('nu', torch.tensor(nu))

        # input unit scales
        self.logeta_mu = nn.Parameter(torch.empty(self.dim_in))
        self.logeta_logsig2 = nn.Parameter(torch.empty(self.dim_in))

        self.register_buffer('psi_a', torch.empty(self.dim_in))
        self.register_buffer('psi_b', torch.empty(self.dim_in))

        # input unit indicators
        self.s_mu = nn.Parameter(torch.empty(self.dim_in))
        self.s_logsig2 = nn.Parameter(torch.empty(self.dim_in))

        ### priors

        # layer scales
        if self.infer_nu:
            self.lognu_a_prior = torch.tensor(0.5) # shouldn't this be called "nu_a_prior"?

            self.vtheta_a_prior = torch.tensor(0.5)
            self.vtheta_b_prior = torch.tensor(1/(b_g**2))

            self.nu_dist_prior = HalfCauchy(scale=b_g*torch.ones(1))

        # input unit scales
        self.logeta_a_prior = torch.tensor(0.5) # shouldn't this be called "eta_a_prior"?

        self.psi_a_prior = torch.tensor(0.5)
        self.psi_b_prior = torch.tensor(1/(b_0**2))

        self.eta_dist_prior = HalfCauchy(scale=b_0*torch.ones(self.dim_in))

        # input unit indicators
        self.s_mu_prior = torch.tensor(1.) # what do I want this to be? should it be 1?
        self.s_sig2_prior = torch.tensor(1.)
        self.s_dist_prior = Normal(self.s_mu_prior*torch.ones(self.dim_in), self.s_sig2_prior.sqrt()*torch.ones(self.dim_in))

        ### init params
        self.init_parameters()
        self.sample_features()

    def init_parameters(self):
        # initialize variational parameters

        # layer scale
        if self.infer_nu:
            #self.lognu_mu.data = torch.log(np.abs(self.b_g*torch.randn(1) / torch.randn(1)))
            self.lognu_mu.data = torch.log(1+1e-2*torch.randn(self.lognu_mu.shape))
            self.lognu_logsig2.data.normal_(-9, 1e-2)

        # input unit scales
        #self.logeta_mu.data = torch.log(np.abs(self.b_0*torch.randn(self.logeta_mu.shape) / torch.randn(self.logeta_mu.shape)))
        self.logeta_mu.data = torch.log(1+1e-2*torch.randn(self.logeta_mu.shape))
        self.logeta_logsig2.data.normal_(-9, 1e-2)

        # input unit indicators
        self.s_mu.data.normal_(self.s_mu_prior, 1e-2)
        self.s_logsig2.data.normal_(-9, 1e-2)

        self.fixed_point_updates()

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1)
        self.b.uniform_(0, 2*pi)

    def forward(self, x, weights_type='sample_post'):
        '''
        Note: 'mean_prior' doesn't exist (since HalfCauchy)
        '''

        if weights_type == 'mean_post':
            if self.infer_nu:
                nu = util.reparam_trick_lognormal(self.lognu_mu, self.lognu_logsig2.exp(), sample=False)
            else:
                nu = self.nu
            eta = util.reparam_trick_lognormal(self.logeta_mu, self.logeta_logsig2.exp(), sample=False)
            s = self.s_mu

        elif weights_type == 'sample_prior':
            if self.infer_nu:
                nu = self.nu_dist_prior.sample()
            else:
                nu = self.nu
            eta = self.eta_dist_prior.sample()
            s = self.s_dist_prior.sample()

        elif weights_type == 'sample_post':
            # regular reparameterization trick on scales
            if self.infer_nu:
                nu = util.reparam_trick_lognormal(self.lognu_mu, self.lognu_logsig2.exp(), sample=True)
            else:
                nu = self.nu
            eta = util.reparam_trick_lognormal(self.logeta_mu, self.logeta_logsig2.exp(), sample=True)
            s = util.reparam(self.s_mu, self.s_logsig2.exp())

        #nu=1. # TEMP FOR TESTING
        #eta=1. # TEMP FOR TESTING
        #return F.relu(F.linear(x, nu*eta*self.w, self.b)) # relu features TEMP FOR TESTING

        return sqrt(2/self.dim_out)*torch.cos(F.linear(x, s*nu*eta*self.w, self.b))

    def fixed_point_updates(self):

        # layer scale
        if self.infer_nu:
            self.vtheta_a = torch.tensor([1.]) # torch.ones((self.dim_out,)) # could do this in init
            self.vtheta_b = torch.exp(-self.lognu_mu + 0.5*self.lognu_logsig2.exp()) + 1/(self.b_g**2)

        # input unit scales
        self.psi_a = torch.ones((self.dim_in,)) # could do this in init
        self.psi_b = torch.exp(-self.logeta_mu + 0.5*self.logeta_logsig2.exp()) + 1/(self.b_0**2)


    def _entropy_scales(self):
        '''H(q,q) = E_q[ -q(z)]'''
        e=0

        # layer scale
        if self.infer_nu:
            e += util.lognormal_entropy(self.lognu_logsig2.exp().sqrt().log(), self.lognu_mu, 1)
            e += util.entropy_invgamma(self.vtheta_a, self.vtheta_b)

        # input unit scales
        e += util.lognormal_entropy(self.logeta_logsig2.exp().sqrt().log(), self.logeta_mu, self.dim_in)
        e += util.entropy_invgamma(self.psi_a, self.psi_b)

        return e

    def _cross_entropy_scales(self):
        '''H(q,p) = E_q[ -p(z) ]'''
        ce = 0

        if self.infer_nu:
            ce += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.lognu_mu, q_sig2=self.lognu_logsig2.exp(), q_alpha=self.vtheta_a, q_beta=self.vtheta_b, p_alpha=self.lognu_a_prior) 
            ce += util.cross_entropy_invgamma_new(q_alpha=self.vtheta_a, q_beta=self.vtheta_b, p_alpha=self.vtheta_a_prior, p_beta=self.vtheta_b_prior)

        ce += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.logeta_mu, q_sig2=self.logeta_logsig2.exp(), q_alpha=self.psi_a, q_beta=self.psi_b, p_alpha=self.logeta_a_prior) 
        ce += util.cross_entropy_invgamma_new(q_alpha=self.psi_a, q_beta=self.psi_b, p_alpha=self.psi_a_prior, p_beta=self.psi_b_prior)

        return ce
    def _kl_divergence_scales(self):
        return self._cross_entropy_scales() - self._entropy_scales()

    def _kl_divergence_indicators(self):
        return util.kl_diag_normal(self.s_mu, self.s_logsig2.exp(), self.s_mu_prior, self.s_sig2_prior)

    def kl_divergence(self):
        return self._kl_divergence_scales() + self._kl_divergence_indicators()





class RffBetaLayer(nn.Module):
    """
    Random features with beta distributed indicator s
    """
    def __init__(self, dim_in, dim_out, a_prior=0.5, b_prior=0.5, **kwargs):
        super(RffBetaLayer, self).__init__()
        '''
        s_d ~ beta(a,b)
        '''

        ### architecture
        self.dim_in = dim_in
        self.dim_out = dim_out

        # noncentered random weights
        self.register_buffer('w', torch.empty(dim_out, dim_in))
        self.register_buffer('b', torch.empty(dim_out))

        ### variational parameters

        # input unit indicators
        self.s_a_trans = nn.Parameter(torch.empty(self.dim_in))
        self.s_b_trans = nn.Parameter(torch.empty(self.dim_in))

        ### priors

        # input unit indicators
        self.s_a_prior = torch.tensor(a_prior)
        self.s_b_prior = torch.tensor(b_prior)
        self.s_dist_prior = Beta(self.s_a_prior, self.s_b_prior)
    
        ### other stuff
        self.untransform = nn.Softplus() # ensure correct range

        ### init params
        self.init_parameters()
        self.sample_features()


    def init_parameters(self):
        # initialize variational parameters

        # input unit indicators
        self.s_a_trans.data = 1e-2*torch.randn(self.dim_in)
        self.s_b_trans.data = 1e-2*torch.randn(self.dim_in)

        self.sample_variational(store=True)

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1)
        self.b.uniform_(0, 2*pi)

    def sample_prior(self, shape=torch.Size([]), store=False):
        s = self.s_dist_prior.sample()
        if store: self.s = s
        return s

    def get_variational(self):
        a = self.untransform(self.s_a_trans)
        b = self.untransform(self.s_b_trans)
        return Beta(a, b)

    def sample_variational(self, store=False):
        # sample from the variational family or prior
        q = self.get_variational()
        s = q.rsample()
        if store: self.s = s
        return s

    def fixed_point_updates(self):
        pass

    def forward(self, x, weights_type='sample_post'):

        if weights_type == 'mean_prior':
            s = self.s_dist_prior.mean() # will this work?

        elif weights_type == 'mean_post':
            q = self.get_variational()
            s = q.mean() # will this work?

        elif weights_type == 'sample_prior':
            s = self.sample_prior(shape=self.dim_in)

        elif weights_type == 'sample_post':
            s = self.sample_variational(store=True) # STORE = TRUE!!!!!!!!

        elif weights_type == 'stored':
            s = self.s

        return sqrt(2/self.dim_out)*torch.cos(F.linear(x, s*self.w, self.b))

    def log_prob_variational(self):
        # evaluates log prob of variational distribution at stored variational param values
        q = self.get_variational()
        return torch.sum(q.log_prob(self.s))

    def kl_divergence(self):
        q = self.get_variational()
        return torch.sum(torch.distributions.kl_divergence(q, self.s_dist_prior))




class RffLogitNormalLayer(nn.Module):
    """
    Random features with beta distributed indicator s
    """
    def __init__(self, dim_in, dim_out, s_loc_prior=0.0, s_scale_prior=1.0, **kwargs):
        super(RffLogitNormalLayer, self).__init__()
        '''
        s_d ~ LogitNormal(mu,sig2) <-- parameters of underlying normal distribution
        '''

        ### architecture
        self.dim_in = dim_in
        self.dim_out = dim_out

        # noncentered random weights
        self.register_buffer('w', torch.empty(dim_out, dim_in))
        self.register_buffer('b', torch.empty(dim_out))

        ### variational parameters

        # input unit indicators
        self.s_loc = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal
        self.s_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal

        ### priors

        # input unit indicators
        self.s_loc_prior = torch.tensor(s_loc_prior) # of underlying normal
        self.s_scale_prior = torch.tensor(s_scale_prior) # of underlying normal
        self.s_dist_prior = LogitNormal(self.s_loc_prior, self.s_scale_prior)

        ### other stuff
        self.transform = nn.Softplus() # ensure correct range
        self.untransform = util.softplus_inverse

        self.s = torch.empty(self.dim_in)

        ### init params
        self.init_parameters()
        self.sample_features()

    def init_parameters(self):
        # initialize variational parameters

        # input unit indicators
        self.s_loc.data.normal_(0, 1e-2)
        self.s_scale_untrans.data.normal_(1e-4, 1e-2)

        self.sample_variational(store=True)

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1)
        self.b.uniform_(0, 2*pi)

    def sample_prior(self, shape=torch.Size([]), store=False):
        s = self.s_dist_prior.sample()
        if store: self.s = s
        return s

    def get_variational(self):
        return LogitNormal(loc=self.s_loc, scale=self.transform(self.s_scale_untrans))

    def sample_variational(self, shape=torch.Size([]), store=False):
        # sample from the variational family or prior
        q = self.get_variational()
        s = q.rsample(shape)
        if store: self.s = s
        return s

    def fixed_point_updates(self):
        pass

    def forward(self, x, weights_type='sample_post'):

        if weights_type == 'mean_prior':
            raise NotImplementedError # No analytical solution

        elif weights_type == 'mean_post':
            raise NotImplementedError # No analytical solution

        elif weights_type == 'sample_prior':
            s = self.sample_prior(self.dim_in)

        elif weights_type == 'sample_post':
            s = self.sample_variational()

        elif weights_type == 'stored':
            s = self.s

        return sqrt(2/self.dim_out)*torch.cos(F.linear(x, s*self.w, self.b))

    def log_prob_variational(self):
        # evaluates log prob of variational distribution at stored variational param values
        q = self.get_variational()
        return torch.sum(q.log_prob(self.s))

    def kl_divergence(self):
        q = self.get_variational()
        return torch.sum(torch.distributions.kl_divergence(q, self.s_dist_prior))


def get_layer(name):
    if name == 'LinearLayer':
        return LinearLayer
    elif name == 'RffLayer':
        return RffLayer
    elif name == 'RffHsLayer':
        return RffHsLayer
    elif name == 'RffHsLayer2':
        return RffHsLayer2
    elif name == 'RffBetaLayer':
        return RffBetaLayer
    elif name == 'RffLogitNormalLayer':
        return RffLogitNormalLayer






