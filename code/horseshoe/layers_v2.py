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
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__(dim_in, dim_out)
        '''
        '''

    def init_parameters(self):
        pass

    def get_prior(self):
        pass

    def get_variational(self):
        pass

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







