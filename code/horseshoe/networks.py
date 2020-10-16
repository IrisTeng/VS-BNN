import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import math
import os

import layers_v2 as layers

def get_activation(activation_type='relu'):
    if activation_type=='relu':
        return F.relu
    elif activation_type=='tanh':
        return torch.tanh
    else:
        print('activation not recognized')

class BLM(nn.Module):
    def __init__(self, dim_in, dim_out, sig2_inv):
        super(BLM, self).__init__()
        '''
        Bayesian linear model with flat prior and known variance.
        Requires precompute to be run whenever changing x. 

        sig2: observation noise
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.sig2_inv = sig2_inv

        self.register_buffer('beta_mu', torch.empty((self.dim_out, self.dim_in))) # BC: should these be parameters?
        self.register_buffer('beta_sig2', torch.empty((self.dim_out, self.dim_in)))

        self.init_parameters()

    def init_parameters(self):
        #self.beta_mu.data.normal_()
        self.beta_mu = torch.tensor([[2.]]) # TEMP
        self.beta_sig2.data.normal_(-9, 1e-2).exp_()

    def precompute(self, x):
        self.xx_inv = torch.inverse(x.transpose(0,1) @ x)
        self.h = self.xx_inv @ x.transpose(0,1)
        
    def forward(self, x, sample=True):
        if sample:
            beta = self.beta_mu + self.beta_sig2.sqrt()*torch.randn(self.beta_sig2.shape)
        else:
            beta = self.beta_mu
        return F.linear(x, beta)

    def fixed_point_updates(self, y):
        self.beta_mu = self.h @ y
        self.beta_sig2 = 1/self.sig2_inv * self.xx_inv

class BNN(nn.Module):
    """
    Fully connected BNN with constant width
    """
    def __init__(self, dim_in, dim_out, dim_hidden=50, n_layers=1, activation_type='relu', \
        infer_noise=False, sig2_inv=None, sig2_inv_alpha_prior=None, sig2_inv_beta_prior=None, \
        linear_term=False, linear_dim_in=None, \
        **kwargs):
        super(BNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.activation_type = activation_type

        self.activation = get_activation(self.activation_type)

        self.infer_noise=infer_noise
        self.linear_term=linear_term
        self.linear_dim_in=linear_dim_in

        # linear term
        if linear_term:
            self.blm = BLM(linear_dim_in, dim_out, sig2_inv)

        # noise
        if self.infer_noise:
            self.sig2_inv_alpha_prior=torch.tensor(sig2_inv_alpha_prior)
            self.sig2_inv_beta_prior=torch.tensor(sig2_inv_beta_prior)
            self.sig2_inv = None

            self.register_buffer('sig2_inv_alpha', torch.empty(1, requires_grad=False))  # For now each output gets same noise
            self.register_buffer('sig2_inv_beta', torch.empty(1, requires_grad=False)) 
        else:
            self.sig2_inv_alpha_prior=None
            self.sig2_inv_beta_prior=None

            self.register_buffer('sig2_inv', torch.tensor(sig2_inv))


        # layers
        self.fc1 = layers.HSLayer_type4(self.dim_in, self.dim_hidden)
        #self.fc1 = layers.BBBLayer(self.dim_in, self.dim_hidden)

        self.fc_hidden = nn.ModuleList([layers.HSLayer(self.dim_hidden, self.dim_hidden) for _ in range(self.n_layers-1)])
        self.fc_out = layers.HSOutputLayer(self.dim_hidden, self.dim_out)
        #self.fc_out = layers.BBBLayer(self.dim_hidden, self.dim_out)


    def precompute(self):
        pass

    def forward(self, x, x_linear=None, sample=True):

        # network
        x = self.fc1(x, sample=sample)
        x = self.activation(x)

        for layer in self.fc_hidden:
            x = layer(x, sample=sample)
            x = self.activation(x)
        ypred_bnn = self.fc_out(x, sample=sample)

        # add linear term if specified
        if self.linear_term and x_linear is not None:
            return ypred_bnn + self.blm(x_linear, sample=sample)
        else:
            return ypred_bnn

    def kl_divergence(self):
        kld = self.fc1.kl_divergence() + self.fc_out.kl_divergence()
        for layer in self.fc_hidden:
            kld += layer.kl_divergence()
        return kld

    def neg_log_prob(self, y_observed, y_pred):
        N = y_observed.shape[0] # BC: Should this be multiplied by output dimension?

        if self.infer_noise:
            sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta # Is this right? i.e. IG vs G
        else:
            sig2_inv = self.sig2_inv

        log_prob = -0.5 * N * math.log(2 * math.pi) + 0.5 * N * torch.log(sig2_inv) - 0.5 * torch.sum((y_observed - y_pred)**2) * sig2_inv

        return -log_prob

    def loss(self, x, y, x_linear=None, temperature=1): # TEMP
        '''negative elbo'''
        y_pred = self.forward(x, x_linear)

        kl_divergence = self.kl_divergence()
        neg_log_prob = self.neg_log_prob(y, y_pred)

        return neg_log_prob + temperature*kl_divergence

    def fixed_point_updates(self, x, y, x_linear=None, temperature=1): 
        self.fc1.fixed_point_updates()
        self.fc_out.fixed_point_updates()
        for layer in self.fc_hidden:
            layer.fixed_point_updates()

        if self.linear_term:
            if self.infer_noise:
                self.blm.sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta 
            
            self.blm.fixed_point_updates(y - self.forward(x, x_linear=None, sample=True)) # Subtract off just the bnn

            #print('beta_mu: ', self.blm.beta_mu)

        if self.infer_noise and temperature > 0: 
            
            sample_y_bnn = self.forward(x, x_linear=None, sample=True) # Sample
            if self.linear_term:
                E_y_linear = F.linear(x_linear, self.blm.beta_mu)
                SSR = torch.sum((y-sample_y_bnn-E_y_linear)**2) + torch.sum(self.blm.xx_inv * self.blm.beta_sig2).sum()
            else:
                SSR = torch.sum((y - sample_y_bnn)**2)

            self.sig2_inv_alpha = self.sig2_inv_alpha_prior + temperature*0.5*x.shape[0] # Can be precomputed
            self.sig2_inv_beta = self.sig2_inv_beta_prior + temperature*0.5*SSR

            #print('E[sig2]:', self.sig2_inv_beta / (self.sig2_inv_alpha-1) )

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.fc1.init_parameters()
        for layer in self.fc_hidden:
            layer.init_parameters()
        self.fc_out.init_parameters()

        if self.infer_noise:
            self.sig2_inv_alpha = self.sig2_inv_alpha_prior
            self.sig2_inv_beta = self.sig2_inv_beta_prior

        if self.linear_term:
            self.blm.init_parameters()

    def reinit_parameters(self, x, y, n_reinit=1):
        seeds = torch.zeros(n_reinit).long().random_(0, 1000)
        losses = torch.zeros(n_reinit)
        for i in range(n_reinit):
            self.init_parameters(seeds[i])
            losses[i] = self.loss(x, y)

        self.init_parameters(seeds[torch.argmin(losses).item()])

    def precompute(self, x=None, x_linear=None):
        # Needs to be run before training
        if self.linear_term:
            self.blm.precompute(x_linear)

    def get_n_parameters(self):
        n_param=0
        for p in self.parameters():
            n_param+=np.prod(p.shape)
        return n_param

    def print_state(self, x, y, epoch=0, n_epochs=0):
        '''
        prints things like training loss, test loss, etc
        '''


        #print('Epoch[{}/{}], log_prob: {:.6f}, kl: {:.6f}, elbo: {:.6f}'\
        #                .format(epoch, n_epochs, -self.neg_log_prob().item(), self.kl_divergence().item(), -self.loss().item()))

        print('Epoch[{}/{}], kl: {:.6f}, elbo: {:.6f}'\
                        .format(epoch, n_epochs, self.kl_divergence().item(), -self.loss(x,y).item()))


class Rff(nn.Module):
    """
    Single layer RFF model
    """
    def __init__(self, dim_in, dim_out, sig2_inv, dim_hidden=50, \
        **kwargs):
        super(Rff, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.register_buffer('sig2_inv', torch.tensor(sig2_inv).clone().detach())

        # layers
        self.layer_in = layers.RffLayer(self.dim_in, self.dim_hidden)
        self.layer_out = layers.LinearLayer(self.dim_hidden, prior_sig2=10/self.dim_in, sig2_y=1/sig2_inv)

    def forward(self, x, x_linear=None, weights_type='sample_post'):
        h = self.layer_in(x)
        return self.layer_out(h, weights_type=weights_type)

    def fixed_point_updates(self, x, y):   
        h = self.layer_in(x) # hidden units
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.layer_in.init_parameters()
        self.layer_out.init_parameters()

    def reinit_parameters(self, x, y, n_reinit=1):
        seeds = torch.zeros(n_reinit).long().random_(0, 1000)
        losses = torch.zeros(n_reinit)
        for i in range(n_reinit):
            self.init_parameters(seeds[i])
            losses[i] = self.loss(x, y)

        self.init_parameters(seeds[torch.argmin(losses).item()])
        
class RffHs(nn.Module):
    """
    RFF model with horseshoe

    Currently only single layer supported
    """
    def __init__(self, 
        dim_in, \
        dim_out, \
        dim_hidden=50, \
        infer_noise=False, sig2_inv=None, sig2_inv_alpha_prior=None, sig2_inv_beta_prior=None, \
        linear_term=False, linear_dim_in=None,
        layer_in_name='RffVarSelectLogitNormalLayer',
        **kwargs):
        super(RffHs, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.infer_noise=infer_noise
        self.linear_term=linear_term
        self.linear_dim_in=linear_dim_in

        # noise
        if self.infer_noise:
            self.sig2_inv_alpha_prior=torch.tensor(sig2_inv_alpha_prior)
            self.sig2_inv_beta_prior=torch.tensor(sig2_inv_beta_prior)
            self.sig2_inv = None

            self.register_buffer('sig2_inv_alpha', torch.empty(1, requires_grad=False))  # For now each output gets same noise
            self.register_buffer('sig2_inv_beta', torch.empty(1, requires_grad=False)) 
        else:
            self.sig2_inv_alpha_prior=None
            self.sig2_inv_beta_prior=None

            self.register_buffer('sig2_inv', torch.tensor(sig2_inv).clone().detach())

        # layers
        #self.layer_in = layers.RffHsLayer2(self.dim_in, self.dim_hidden, **kwargs)
        #self.layer_in = layers.RffLogitNormalLayer(self.dim_in, self.dim_hidden, **kwargs)
        self.layer_in = layers.get_layer(layer_in_name)(self.dim_in, self.dim_hidden, **kwargs)
        self.layer_out = layers.LinearLayer(self.dim_hidden, sig2_y=1/sig2_inv, **kwargs)

    def forward(self, x, x_linear=None, weights_type_layer_in='sample_post', weights_type_layer_out='sample_post', n_samp_layer_in=None):
        '''
        n_samp is number of samples from variational distribution (first layer)
        '''

        # network
        h = self.layer_in(x, weights_type=weights_type_layer_in, n_samp=n_samp_layer_in)
        y = self.layer_out(h, weights_type=weights_type_layer_out)

        # add linear term if specified
        if self.linear_term and x_linear is not None:
            return y + self.blm(x_linear, sample=sample)
        else:
            return y

    def sample_posterior_predictive(self, x_test, x_train, y_train):
        '''
        Need training data in order to get sample from non-variational full conditional distribution (output layer)

        Code duplicates some of forward, not ideal
        '''

        # 1: sample from variational distribution
        self.layer_in.sample_variational(store=True)

        # 2: forward pass of training data with sample from 1
        h = self.layer_in(x_train, weights_type='stored')

        # 3: sample output weights from conjugate (depends on ouput from 2)
        self.layer_out.fixed_point_updates(h, y_train) # conjugate update of output weights 
        self.layer_out.sample_weights(store=True)

        # 4: forward pass of test data using samples from 1 and 3
        return self.forward(x_test, weights_type_layer_in='stored', weights_type_layer_out='stored')

    def kl_divergence(self):
        return self.layer_in.kl_divergence()

    def log_prob(self, y_observed, y_pred):
        '''
        y_observed: (n_obs, dim_out)
        y_pred: (n_obs, n_pred, dim_out)

        averages over n_pred (e.g. could represent different samples), sums over n_obs
        '''
        lik = Normal(y_pred, torch.sqrt(1/self.sig2_inv))
        return lik.log_prob(y_observed.unsqueeze(1)).mean(1).sum(0)

    def loss_original(self, x, y, x_linear=None, temperature=1, n_samp=1):
        '''negative elbo'''
        y_pred = self.forward(x, x_linear, weights_type_layer_in='sample_post', weights_type_layer_out='stored', n_samp_layer_in=n_samp)

        kl_divergence = self.kl_divergence()
        #kl_divergence = 0

        log_prob = self.log_prob(y, y_pred)
        #log_prob = 0
        return -log_prob + temperature*kl_divergence

    def loss(self, x, y, x_linear=None, temperature=1, n_samp=1):
        '''
        Uses sample of weights from full conditional *based on samples of s* to compute likelihood
        '''

        kl_divergence = self.kl_divergence()
        #breakpoint()

        # 1: sample from variational distribution
        self.layer_in.sample_variational(store=True)

        # 2: forward pass of training data with sample from 1
        h = self.layer_in(x, weights_type='stored')

        # 3: sample output weights from conjugate (depends on ouput from 2)
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 
        self.layer_out.sample_weights(store=True)

        # 4: forward pass of test data using samples from 1 and 3
        y_pred = self.forward(x, weights_type_layer_in='stored', weights_type_layer_out='stored', n_samp_layer_in=1)

        log_prob = self.log_prob(y, y_pred)

        #breakpoint()
        return -log_prob + temperature*kl_divergence


    def fixed_point_updates(self, x, y, x_linear=None, temperature=1): 
        self.layer_in.fixed_point_updates() # update horseshoe aux variables

        #### COMMENTING OUT OUTPUT LAYER UPDATES SINCE NOW PART OF LOSS FUNCTION ####
        """
        h = self.layer_in(x, weights_type='sample_post') # hidden units based on sample from variational dist
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 
        self.layer_out.sample_weights(store=True) # sample output weights from full conditional
        """
        ####

        if self.linear_term:
            if self.infer_noise:
                self.blm.sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta # Shouldnt this be a samplle?
            
            self.blm.fixed_point_updates(y - self.forward(x, x_linear=None, sample=True)) # Subtract off just the bnn

        if self.infer_noise and temperature > 0: 
            
            sample_y_bnn = self.forward(x, x_linear=None, sample=True) # Sample
            if self.linear_term:
                E_y_linear = F.linear(x_linear, self.blm.beta_mu)
                SSR = torch.sum((y-sample_y_bnn-E_y_linear)**2) + torch.sum(self.blm.xx_inv * self.blm.beta_sig2).sum()
            else:
                SSR = torch.sum((y - sample_y_bnn)**2)

            self.sig2_inv_alpha = self.sig2_inv_alpha_prior + temperature*0.5*x.shape[0] # Can be precomputed
            self.sig2_inv_beta = self.sig2_inv_beta_prior + temperature*0.5*SSR

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.layer_in.init_parameters()
        self.layer_out.init_parameters()

        if self.infer_noise:
            self.sig2_inv_alpha = self.sig2_inv_alpha_prior
            self.sig2_inv_beta = self.sig2_inv_beta_prior

        if self.linear_term:
            self.blm.init_parameters()

    def reinit_parameters(self, x, y, n_reinit=1):
        seeds = torch.zeros(n_reinit).long().random_(0, 1000)
        losses = torch.zeros(n_reinit)
        for i in range(n_reinit):
            self.init_parameters(seeds[i])
            losses[i] = self.loss(x, y)

        self.init_parameters(seeds[torch.argmin(losses).item()])

    def precompute(self, x=None, x_linear=None):
        # Needs to be run before training
        if self.linear_term:
            self.blm.precompute(x_linear)

    def get_n_parameters(self):
        n_param=0
        for p in self.parameters():
            n_param+=np.prod(p.shape)
        return n_param

    def print_state(self, x, y, epoch=0, n_epochs=0):
        '''
        prints things like training loss, test loss, etc
        '''
        print('Epoch[{}/{}], kl: {:.6f}, likelihood: {:.6f}, elbo: {:.6f}'\
                        .format(epoch, n_epochs, self.kl_divergence().item(), -self.loss(x,y,temperature=0).item(), -self.loss(x,y).item()))


class RffBeta(nn.Module):
    """
    RFF model beta prior on indicators

    Currently only single layer supported
    """
    def __init__(self, 
        dim_in, \
        dim_out, \
        dim_hidden=50, \
        infer_noise=False, sig2_inv=None, sig2_inv_alpha_prior=None, sig2_inv_beta_prior=None, \
        linear_term=False, linear_dim_in=None,
        **kwargs):
        super(RffBeta, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.infer_noise=infer_noise
        self.linear_term=linear_term
        self.linear_dim_in=linear_dim_in

        # noise
        if self.infer_noise:
            self.sig2_inv_alpha_prior=torch.tensor(sig2_inv_alpha_prior)
            self.sig2_inv_beta_prior=torch.tensor(sig2_inv_beta_prior)
            self.sig2_inv = None

            self.register_buffer('sig2_inv_alpha', torch.empty(1, requires_grad=False))  # For now each output gets same noise
            self.register_buffer('sig2_inv_beta', torch.empty(1, requires_grad=False)) 
        else:
            self.sig2_inv_alpha_prior=None
            self.sig2_inv_beta_prior=None

            self.register_buffer('sig2_inv', torch.tensor(sig2_inv).clone().detach())

        # layers
        self.layer_in = layers.RffBetaLayer(self.dim_in, self.dim_hidden, **kwargs)
        self.layer_out = layers.LinearLayer(self.dim_hidden, sig2_y=1/sig2_inv, **kwargs)

    def forward(self, x, x_linear=None, weights_type_layer_in='sample_post', weights_type_layer_out='sample_post'):

        # network
        h = self.layer_in(x, weights_type=weights_type_layer_in)
        y = self.layer_out(h, weights_type=weights_type_layer_out)

        # add linear term if specified
        if self.linear_term and x_linear is not None:
            return y + self.blm(x_linear, sample=sample)
        else:
            return y

    def kl_divergence(self):
        return self.layer_in.kl_divergence()

    def compute_loss_gradients(self, x, y, x_linear=None, temperature=1.):

        # sample from variational dist
        self.layer_in.sample_variational(store=True)

        # compute log likelihood
        y_pred = self.forward(x, x_linear, weights_type_layer_in='stored', weights_type_layer_out='stored')
        log_lik = -self.neg_log_prob(y, y_pred)

        # gradients of score function
        for p in self.layer_in.parameters(): 
            if p.grad is not None:
                p.grad.zero_()

        log_q = self.layer_in.log_prob_variational()
        log_q.backward()

        self.layer_in.s_a_trans_grad_q = self.layer_in.s_a_trans.grad.clone()
        self.layer_in.s_b_trans_grad_q = self.layer_in.s_b_trans.grad.clone()

        # gradients of kl
        for p in self.layer_in.parameters(): p.grad.zero_()

        kl = self.kl_divergence()
        kl.backward()

        self.layer_in.s_a_trans_grad_kl = self.layer_in.s_a_trans.grad.clone()
        self.layer_in.s_b_trans_grad_kl = self.layer_in.s_b_trans.grad.clone()

        # gradients of loss=-elbo
        with torch.no_grad():
            self.layer_in.s_a_trans.grad = -log_lik*self.layer_in.s_a_trans_grad_q + temperature*self.layer_in.s_a_trans_grad_kl
            self.layer_in.s_b_trans.grad = -log_lik*self.layer_in.s_b_trans_grad_q + temperature*self.layer_in.s_b_trans_grad_kl

    def loss(self, x, y, x_linear=None, temperature=1):
        '''negative elbo
        NON DIFFERENTIABLE BECAUSE OF SCORE METHOD
        '''
        y_pred = self.forward(x, x_linear, weights_type_layer_in='sample_post', weights_type_layer_out='stored')

        kl_divergence = self.kl_divergence()
        #kl_divergence = 0

        neg_log_prob = self.neg_log_prob(y, y_pred)
        #neg_log_prob = 0

        return neg_log_prob + temperature*kl_divergence

    def neg_log_prob(self, y_observed, y_pred):
        N = y_observed.shape[0]
        if self.infer_noise:
            sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta # Is this right? i.e. IG vs G
        else:
            sig2_inv = self.sig2_inv
        log_prob = -0.5 * N * math.log(2 * math.pi) + 0.5 * N * torch.log(sig2_inv) - 0.5 * torch.sum((y_observed - y_pred)**2) * sig2_inv
        return -log_prob

    def fixed_point_updates(self, x, y, x_linear=None, temperature=1): 

        h = self.layer_in(x, weights_type='sample_post') # hidden units based on sample from variational dist
        self.layer_out.fixed_point_updates(h, y) # conjugate update of output weights 

        self.layer_out.sample_weights(store=True) # sample output weights from full conditional

        if self.linear_term:
            if self.infer_noise:
                self.blm.sig2_inv = self.sig2_inv_alpha/self.sig2_inv_beta # Shouldnt this be a samplle?
            
            self.blm.fixed_point_updates(y - self.forward(x, x_linear=None, sample=True)) # Subtract off just the bnn

        if self.infer_noise and temperature > 0: 
            
            sample_y_bnn = self.forward(x, x_linear=None, sample=True) # Sample
            if self.linear_term:
                E_y_linear = F.linear(x_linear, self.blm.beta_mu)
                SSR = torch.sum((y-sample_y_bnn-E_y_linear)**2) + torch.sum(self.blm.xx_inv * self.blm.beta_sig2).sum()
            else:
                SSR = torch.sum((y - sample_y_bnn)**2)

            self.sig2_inv_alpha = self.sig2_inv_alpha_prior + temperature*0.5*x.shape[0] # Can be precomputed
            self.sig2_inv_beta = self.sig2_inv_beta_prior + temperature*0.5*SSR

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.layer_in.init_parameters()
        self.layer_out.init_parameters()

        if self.infer_noise:
            self.sig2_inv_alpha = self.sig2_inv_alpha_prior
            self.sig2_inv_beta = self.sig2_inv_beta_prior

        if self.linear_term:
            self.blm.init_parameters()

    def reinit_parameters(self, x, y, n_reinit=1):
        seeds = torch.zeros(n_reinit).long().random_(0, 1000)
        losses = torch.zeros(n_reinit)
        for i in range(n_reinit):
            self.init_parameters(seeds[i])
            losses[i] = self.loss(x, y)

        self.init_parameters(seeds[torch.argmin(losses).item()])

    def precompute(self, x=None, x_linear=None):
        # Needs to be run before training
        if self.linear_term:
            self.blm.precompute(x_linear)

    def get_n_parameters(self):
        n_param=0
        for p in self.parameters():
            n_param+=np.prod(p.shape)
        return n_param

    def print_state(self, x, y, epoch=0, n_epochs=0):
        '''
        prints things like training loss, test loss, etc
        '''
        print('Epoch[{}/{}], kl: {:.6f}, likelihood: {:.6f}, elbo: {:.6f}'\
                        .format(epoch, n_epochs, self.kl_divergence().item(), -self.loss(x,y,temperature=0).item(), -self.loss(x,y).item()))




def train(model, optimizer, x, y, n_epochs, x_linear=None, n_warmup = 0, n_rep_opt=10, print_freq=None, frac_start_save=1, frac_lookback=0.5, path_checkpoint='./'):
    '''
    frac_lookback will only result in reloading early stopped model if frac_lookback < 1 - frac_start_save
    '''

    loss = torch.zeros(n_epochs)
    loss_best = torch.tensor(float('inf'))
    loss_best_saved = torch.tensor(float('inf'))
    saved_model = False
    model.precompute(x, x_linear)

    for epoch in range(n_epochs):

        # TEMPERATURE HARDECODED, NEED TO FIX
        #temperature_kl = 0. if epoch < n_epochs/2 else 1.0
        #temperature_kl = epoch / (n_epochs/2) if epoch < n_epochs/2 else 1.0
        temperature_kl = epoch / (n_epochs/10) if epoch < n_epochs/10 else 1.0
        #temperature_kl = 0. # SET TO ZERO TO IGNORE KL

        for i in range(n_rep_opt):
    
            l = model.loss(x, y, x_linear=x_linear, temperature=temperature_kl)

            # backward
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            optimizer.step()

            ##
            #print('------------- %d -------------' % epoch)
            #print('s     :', model.layer_in.s_loc.data)
            #print('grad  :', model.layer_in.s_loc.grad)

            #model.layer_in.s_loc.grad.zero_()
            #kl = model.layer_in.kl_divergence()
            #kl.backward()
            #if epoch > 500:
            #    breakpoint()
            #print('grad kl:', model.layer_in.s_loc.grad)
            ##

        loss[epoch] = l.item()

        with torch.no_grad():
            model.fixed_point_updates(x, y, x_linear=x_linear, temperature=1)

        # print state
        if print_freq is not None:
            if (epoch + 1) % print_freq == 0:
                model.print_state(x, y, epoch+1, n_epochs)

        # see if improvement made (only used if KL isn't tempered)
        if loss[epoch] < loss_best and temperature_kl==1.0:
            loss_best = loss[epoch]

        # save model
        if epoch > frac_start_save*n_epochs and loss[epoch] < loss_best_saved:
            print('saving mode at epoch = %d' % epoch)
            saved_model = True
            loss_best_saved = loss[epoch]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss[epoch],
            },  os.path.join(path_checkpoint, 'checkpoint.tar'))

        # end training if no improvement made in a while and more than half way done
        epoch_lookback = np.maximum(1, int(epoch - .25*n_epochs)) # lookback is 25% of samples by default
        if epoch_lookback > frac_start_save*n_epochs+1:
            loss_best_lookback = torch.min(loss[epoch_lookback:epoch+1])
            percent_improvement = (loss_best - loss_best_lookback)/torch.abs(loss_best) # positive is better
            if percent_improvement < 0.0:
                print('stopping early at epoch = %d' % epoch)
                break

    # reload best model if saving
    if saved_model:
        checkpoint = torch.load(os.path.join(path_checkpoint, 'checkpoint.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('reloading best model from epoch = %d' % checkpoint['epoch'])
        model.eval()

    return loss[:epoch]


def train_score(model, optimizer, x, y, n_epochs, x_linear=None, n_warmup = 0, n_rep_opt=10, print_freq=None, frac_start_save=1):
    loss = torch.zeros(n_epochs)
    loss_best = 1e9 # Need better way of initializing to make sure it's big enough
    model.precompute(x, x_linear)

    for epoch in range(n_epochs):

        # TEMPERATURE HARDECODED, NEED TO FIX
        #temperature_kl = 0. if epoch < n_epochs/2 else 1
        #temperature_kl = epoch / (n_epochs/2) if epoch < n_epochs/2 else 1
        temperature_kl = 0. # SET TO ZERO TO IGNORE KL

        for i in range(n_rep_opt):

            optimizer.zero_grad()

            model.compute_loss_gradients(x, y, x_linear=x_linear, temperature=temperature_kl)

            # backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

        with torch.no_grad():
            model.fixed_point_updates(x, y, x_linear=x_linear, temperature=1)

        if epoch > frac_start_save*n_epochs and loss[epoch] < loss_best: 
            print('saving...')
            loss_best = loss[epoch]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss[epoch],
            }, 'checkpoint.tar')

        if print_freq is not None:
            if (epoch + 1) % print_freq == 0:
                model.print_state(x, y, epoch+1, n_epochs)

    return loss
