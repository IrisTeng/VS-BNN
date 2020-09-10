import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.beta import Beta
from torch.distributions.log_normal import LogNormal
from distributions import LogitNormal, ProductDistribution, InvGamma, PointMass
from torch.distributions import kl_divergence

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

        ### architecture
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.register_buffer('w', torch.empty(dim_out, dim_in))
        self.register_buffer('b', torch.empty(dim_out))

        self.sample_features()

        self.act = lambda z: sqrt(2/self.dim_out)*torch.cos(z)

    def sample_features(self):
        # sample random weights for RFF features
        self.w.normal_(0, 1)
        self.b.uniform_(0, 2*pi)

    def forward(self, x):
        return self.act(F.linear(x, self.w, self.b))


class _RffVarSelectLayer(RffLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__(dim_in, dim_out)
        '''
        '''

        ### scale parameters
        self.s = torch.empty(self.dim_in)

    def init_parameters(self):
        raise NotImplementedError

    def get_prior(self):
        raise NotImplementedError

    def get_variational(self):
        raise NotImplementedError

    def sample_prior(self, shape=torch.Size([]), store=False):
        p = self.get_prior()
        s = p.sample(shape)
        if store: self.s = s
        return s

    def sample_variational(self, shape=torch.Size([]), store=False):
        # sample from the variational family or prior
        q = self.get_variational()
        s = q.rsample(shape)
        if store: self.s = s
        return s

    def log_prob_variational(self):
        # evaluates log prob of variational distribution at stored variational param values
        q = self.get_variational()
        return torch.sum(q.log_prob(self.s))

    def kl_divergence(self):
        p = self.get_prior()
        q = self.get_variational()
        return torch.sum(torch.distributions.kl_divergence(q, p))

    def fixed_point_updates(self):
        pass

    def forward(self, x, weights_type='sample_post', n_samp=None):
        '''
        if taking samples:
            if n_samp == None, output is (n_obs, dim_hidden)
            otherwise output is (n_obs, n_samp, dim_hidden)
        '''
        s_shape = torch.Size([]) if n_samp is None else (n_samp,)

        if weights_type == 'mean_prior':
            p = self.get_prior()
            s = p.mean()

        elif weights_type == 'mean_post':
            q = self.get_variational()
            s = q.mean()

        elif weights_type == 'sample_prior':
            s = self.sample_prior(s_shape)

        elif weights_type == 'sample_post':
            s = self.sample_variational(s_shape)

        elif weights_type == 'stored':
            s = self.s

        if weights_type == 'mean_prior' or weights_type == 'mean_prior' or n_samp is None:
            return self.act(F.linear(x, s*self.w, self.b)) # (n_obs, dim_hidden)
        else:
            xs = x.unsqueeze(1) * s.unsqueeze(0) # (n_obs, n_samp, dim_hidden)
            return self.act(F.linear(xs, self.w, self.b))

class RffVarSelectHsLayer(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, b_g=1, b_0=1, infer_nu=True, nu=None):
        super().__init__(dim_in, dim_out)
        '''
        '''
        self.b_g = b_g
        self.b_0 = b_0

        self.infer_nu = infer_nu

        ### variational parameters

        # layer scales
        if self.infer_nu:
            self.nu_loc = nn.Parameter(torch.empty(1)) # of underlying normal
            self.nu_scale_untrans = nn.Parameter(torch.empty(1)) # of underlying normal

            self.register_buffer('vtheta_a', torch.empty(1))
            self.register_buffer('vtheta_b', torch.empty(1))
        
        else:
            self.register_buffer('nu', torch.tensor(nu))

        # input unit scales
        self.eta_loc = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal
        self.eta_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal

        self.register_buffer('psi_a', torch.empty(self.dim_in))
        self.register_buffer('psi_b', torch.empty(self.dim_in))

        # input unit indicators
        self.s_loc = nn.Parameter(torch.empty(self.dim_in))
        self.s_scale_untrans = nn.Parameter(torch.empty(self.dim_in))

        ### priors

        # layer scales
        if self.infer_nu:
            self.nu_a_prior = torch.tensor(0.5)

            self.vtheta_a_prior = torch.tensor(0.5)
            self.vtheta_b_prior = torch.tensor(1/(b_g**2))

        # input unit scales
        self.eta_a_prior = torch.tensor(0.5) 

        self.psi_a_prior = torch.tensor(0.5)
        self.psi_b_prior = torch.tensor(1/(b_0**2))

        # input unit indicators
        self.s_loc_prior = torch.tensor(1.) # what do I want this to be? should it be 1?
        self.s_scale_prior = torch.tensor(1.)

        ### other stuff
        self.transform = nn.Softplus() # ensure correct range

        self.init_parameters()

    def init_parameters(self):
        # initialize variational parameters

        # layer scale
        if self.infer_nu:
            self.nu_loc.data.normal_(0, 1e-2)
            self.nu_scale_untrans.data.normal_(1e-4, 1e-2)

        # input unit scales
        self.eta_loc.data.normal_(0, 1e-2)
        self.eta_scale_untrans.data.normal_(1e-4, 1e-2)

        # input unit indicators
        self.s_loc.data.normal_(self.s_loc_prior, 1e-2)
        self.s_scale_untrans.data.normal_(1e-4, 1e-2)

        self.fixed_point_updates()

    def fixed_point_updates(self):

        # layer scale
        if self.infer_nu:
            self.vtheta_a = torch.tensor([1.]) # torch.ones((self.dim_out,)) # could do this in init
            self.vtheta_b = torch.exp(-self.nu_loc + 0.5*self.transform(self.nu_scale_untrans)) + 1/(self.b_g**2)

        # input unit scales
        self.psi_a = torch.ones((self.dim_in,)) # could do this in init
        self.psi_b = torch.exp(-self.eta_loc + 0.5*self.transform(self.eta_scale_untrans)) + 1/(self.b_0**2)

    def _get_prior_all(self):
        # returns all priors. includes aux variables
        s_dist = Normal(self.s_loc_prior*torch.ones(self.dim_in), self.s_scale_prior*torch.ones(self.dim_in))
        eta_dist = HalfCauchy(scale=self.b_0*torch.ones(self.dim_in)) # really should be inverse gamma, but then I'd need to sample conditional on psi
        psi_dist = InvGamma(self.psi_a_prior*torch.ones(self.dim_in), self.psi_b_prior*torch.ones(self.dim_in))

        if self.infer_nu:
            nu_dist = HalfCauchy(scale=self.b_g*torch.ones(1))
            vtheta_dist = InvGamma(self.vtheta_a_prior, self.vtheta_b_prior)
        else:
            nu_dist = PointMass(self.nu)
            vtheta_dist = None

        return s_dist, eta_dist, psi_dist, nu_dist, vtheta_dist

    def get_prior(self):
        s_dist, eta_dist, _, nu_dist, _ = self._get_prior_all()
        return ProductDistribution([s_dist, eta_dist, nu_dist])
        
    def _get_variational_all(self):
        # returns all variational distributions. includes aux variables
        s_dist = Normal(self.s_loc, self.transform(self.s_scale_untrans))
        eta_dist = LogNormal(loc=self.eta_loc, scale=self.transform(self.eta_scale_untrans))
        psi_dist = InvGamma(self.psi_a, self.psi_b)

        if self.infer_nu:
            nu_dist = LogNormal(loc=self.nu_loc, scale=self.transform(self.nu_scale_untrans))
            vtheta_dist = InvGamma(self.vtheta_a, self.vtheta_b)
        else:
            nu_dist = PointMass(self.nu)
            vtheta_dist = None

        return s_dist, eta_dist, psi_dist, nu_dist, vtheta_dist

    def get_variational(self):
        s_dist, eta_dist, _, nu_dist, _ = self._get_variational_all()
        return ProductDistribution([s_dist, eta_dist, nu_dist])

    def kl_divergence(self):
        '''
        overwrites parent class so aux variables included
        '''

        q_s, q_eta, q_psi, q_nu, q_vtheta = self._get_variational_all()
        p_s, p_eta, p_psi, p_nu, p_vtheta = self._get_variational_all()
        
        kl = 0.0

        # unit indicators (normal-normal)
        kl += torch.sum(kl_divergence(q_s, p_s))

        # eta
        kl += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.eta_loc, 
                                                             q_sig2=self.transform(self.eta_scale_untrans).pow(2), 
                                                             q_alpha=self.psi_a, 
                                                             q_beta=self.psi_b, 
                                                             p_alpha=self.eta_a_prior) 
        kl += -self.q_eta.entropy()

        # psi
        kl += torch.sum(kl_divergence(q_psi, p_psi))


        if torch.infer_nu:
            # nu
            kl += util.cross_entropy_cond_lognormal_invgamma_new(q_mu=self.nu_loc, 
                                                                 q_sig2=self.transform(self.nu_scale_untrans).pow(2), 
                                                                 q_alpha=self.vtheta_a, 
                                                                 q_beta=self.vtheta_b, 
                                                                 p_alpha=self.nu_a_prior) 
            kl += -self.q_nu.entropy()

            # vtheta
            kl += torch.sum(kl_divergence(q_vtheta, p_vtheta))

        return kl

        

class RffVarSelectBetaLayer(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, s_a_prior=1.0, s_b_prior=1.0):
        super().__init__(dim_in, dim_out)
        '''
        '''

        ### variational parameters

        # input unit indicators
        self.s_a_trans = nn.Parameter(torch.empty(self.dim_in))
        self.s_b_trans = nn.Parameter(torch.empty(self.dim_in))

        ### priors

        # input unit indicators
        self.s_a_prior = torch.tensor(s_a_prior)
        self.s_b_prior = torch.tensor(s_b_prior)
    
        ### other stuff
        self.untransform = nn.Softplus() # ensure correct range

        ### init params
        self.init_parameters()

    def init_parameters(self):
        # initialize variational parameters

        # input unit indicators
        self.s_a_trans.data.normal_(1, 1e-2)
        self.s_b_trans.data.normal_(1, 1e-2)

        self.sample_variational(store=True)

    def get_prior(self):
        return Beta(self.s_a_prior*torch.ones(self.dim_in), self.s_b_prior*torch.ones(self.dim_in))

    def get_variational(self):
        return Beta(self.untransform(self.s_a_trans), self.untransform(self.s_b_trans))

class RffVarSelectLogitNormalLayer(_RffVarSelectLayer):
    '''
    '''
    def __init__(self, dim_in, dim_out, s_loc_prior=0.0, s_scale_prior=1.0):
        super().__init__(dim_in, dim_out)
        '''
        '''
        ### variational parameters

        # input unit indicators
        self.s_loc = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal
        self.s_scale_untrans = nn.Parameter(torch.empty(self.dim_in)) # of underlying normal

        ### priors

        # input unit indicators
        self.s_loc_prior = torch.tensor(s_loc_prior) # of underlying normal
        self.s_scale_prior = torch.tensor(s_scale_prior) # of underlying normal

        ### other stuff
        self.transform = nn.Softplus() # ensure correct range

        self.init_parameters()

    def init_parameters(self):
        self.s_loc.data.normal_(0, 1e-2)
        self.s_scale_untrans.data.normal_(1e-4, 1e-2)

        self.sample_variational(store=True)

    def get_prior(self):
        return LogitNormal(self.s_loc_prior*torch.ones(self.dim_in), self.s_scale_prior*torch.ones(self.dim_in))

    def get_variational(self):
        return LogitNormal(loc=self.s_loc, scale=self.transform(self.s_scale_untrans))


def get_layer(name):
    if name == 'LinearLayer':
        return LinearLayer

    elif name == 'RffLayer':
        return RffLayer

    elif name == 'RffVarSelectHsLayer':
        return RffVarSelectHsLayer

    elif name == 'RffVarSelectBetaLayer':
        return RffVarSelectBetaLayer

    elif name == 'RffVarSelectLogitNormalLayer':
        return RffVarSelectLogitNormalLayer







