import GPy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import exp.util as util
import exp.models as models

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='rbf')
parser.add_argument('--dir_out', type=str, default='output/')
parser.add_argument('--subtract_covariates', action='store_true', help='use Y - beta*X as outcome')

parser.add_argument('--n_obs_min', type=int, default=10)
parser.add_argument('--n_obs_max', type=int, default=100)
parser.add_argument('--n_obs_step', type=int, default=10)

parser.add_argument('--dim_in_min', type=int, default=2)
parser.add_argument('--dim_in_max', type=int, default=5)
parser.add_argument('--dim_in_step', type=int, default=2)

parser.add_argument('--n_rep', type=int, default=2)

parser.add_argument('--model', type=str, default='GP', help='select "GP" or "RFF"')

parser.add_argument('--compute_risk', action='store_true', help='compute bias, variance, risk')

# GP options
parser.add_argument('--opt_likelihood_variance', action='store_true')
parser.add_argument('--opt_kernel_hyperparam', action='store_true')
parser.add_argument('--kernel_lengthscale', type=float, default=1.0)
parser.add_argument('--kernel_variance', type=float, default=1.0)

# RFF options
parser.add_argument('--rff_dim_min', type=int, default=1000)
parser.add_argument('--rff_dim_max', type=int, default=1000)
parser.add_argument('--rff_dim_step', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=16)

# RffVarSelect options
parser.add_argument('--layer_in_name', type=str, default='RffVarSelectLogitNormalLayer')
parser.add_argument('--s_loc_prior', type=float, default=0.0)
parser.add_argument('--s_scale_prior', type=float, default=1.0)

# BKMR options
parser.add_argument('--bkmr_n_samp', type=int, default=1000)

# observational noise variance (sig2) options
parser.add_argument('--sig2_min', type=float, default=.01)
parser.add_argument('--sig2_max', type=float, default=.01)
parser.add_argument('--sig2_n_step', type=float, default=1)


args = parser.parse_args()

if not os.path.exists(args.dir_out):
    os.makedirs(args.dir_out)

with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
    f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))

# allocate space
n_obs_list = util.arrange_full(args.n_obs_min, args.n_obs_max, args.n_obs_step)
dim_in_list = util.arrange_full(args.dim_in_min, args.dim_in_max, args.dim_in_step)
rff_dim_list = util.arrange_full(args.rff_dim_min, args.rff_dim_max, args.rff_dim_step)
sig2_list = np.linspace(args.sig2_min, args.sig2_max, args.sig2_n_step)

print('running experiments for the following setups')
print('number of observations: ', n_obs_list)
print('number of input dimensions: ', dim_in_list)
print('observational noises: ', sig2_list)

## allocate space for results: obs x dim_in x rff_dim x sig2 x rep x input
res_shape = (len(n_obs_list), len(dim_in_list), len(rff_dim_list), len(sig2_list), args.n_rep, np.max(dim_in_list))

res = {
    'psi_mean': np.full(res_shape, np.nan),
    'psi_var': np.full(res_shape, np.nan),
    'n_obs_list': n_obs_list,
    'dim_in_list': dim_in_list,
    'rff_dim_list': dim_in_list,
    'sig2_list': sig2_list,
    'mse': np.full(res_shape[:-1], np.nan),
    'mse_test': np.full(res_shape[:-1], np.nan)
    }

seed = 0
for i, n_obs in enumerate(n_obs_list):
    for j, dim_in in enumerate(dim_in_list):
        for l, rff_dim in enumerate(rff_dim_list):
            for s, sig2 in enumerate(sig2_list):

                print('n_obs [%d/%d], dim_in [%d/%d], rff_dim [%d/%d], sig2 [%d/%d]' % \
                    (i+1,len(n_obs_list),j+1,len(dim_in_list),l+1,len(rff_dim_list),s+1,len(sig2_list)))
            
                for k in range(args.n_rep):
                    seed += 1

                    Z, X, Y, Z_test, X_test, Y_test, sig2 = util.load_data(args.dataset_name, n_obs=n_obs, dim_in=dim_in, sig2=sig2, seed=seed)

                    if args.subtract_covariates:
                        if X is None:
                            print('error: no covariates to subtract')
                        else:
                            Y = util.resid_linear_model(X,Y)

                    if sig2 is None:
                        sig2 = np.var(Y) # initial guess for sig2, could be improved?

                    if args.model=='GP':
                        m = models.GPyVarImportance(Z, Y, sig2=sig2, \
                            opt_kernel_hyperparam=args.opt_kernel_hyperparam, \
                            opt_sig2=args.opt_likelihood_variance,\
                            lengthscale=args.kernel_lengthscale, variance=args.kernel_variance)

                        m.train()
                
                    elif args.model=='RFF':
                        m = models.RffVarImportance(Z)
                        m.train(Z, Y, sig2, rff_dim=rff_dim, batch_size=args.batch_size, epochs=args.epochs)

                    elif args.model=='RFF-PYTORCH':
                        m = models.RffVarImportancePytorch(Z, Y, dim_hidden=rff_dim, sig2_inv=1/sig2)
                        m.train()

                    elif args.model=='RFFHS':
                        m = models.RffHsVarImportance(Z, Y, dim_hidden=rff_dim, sig2_inv=1/sig2,
                            layer_in_name=args.layer_in_name, 
                            s_loc_prior=args.s_loc_prior,
                            s_scale_prior=args.s_scale_prior)
                        loss = m.train(n_epochs=args.epochs, path_checkpoint=args.dir_out)

                    elif args.model=='BKMR':

                        Y2 = Y.copy()
                        Y = Y2.copy()
                        print('mean1', Y.mean())
                        m = models.BKMRVarImportance(Z, Y2, sig2)    
                        m.train(n_samp=args.bkmr_n_samp)
                        print('mean1', Y.mean())


                    psi_est = m.estimate_psi(Z)
                    res['psi_mean'][i,j,l,s,k,:dim_in] = psi_est[0]
                    res['psi_var'][i,j,l,s,k,:dim_in] = psi_est[1]

                    ## loss if available
                    if 'loss' in locals():
                        fig, ax = plt.subplots()
                        ax.plot(loss.numpy())
                        ax.set_xlabel('iterations')
                        ax.set_ylabel('loss')
                        fig.savefig(os.path.join(args.dir_out, 
                                    'loss-n_obs=%d-dim_in=%d-rff_dim=%d-sig2%.2f-rep=%d.png' % (n_obs_list[i],dim_in_list[j],rff_dim_list[l],sig2_list[s],k)))
                        plt.close('all')

                    ## slices
                    if hasattr(m, 'sample_f_post'):
                        #fig, ax = util.plot_slices(m.sample_f_post, Z, Y, quantile=.5, n_samp=500, figsize=(4*dim_in,4))
                        fig, ax = util.plot_slices(m.sample_f_post, Z, Y, quantile=.5, n_samp=100, figsize=(4*dim_in,4))

                        fig.savefig(os.path.join(args.dir_out, 
                                    'slices-n_obs=%d-dim_in=%d-rff_dim=%d-sig2%.2f-rep=%d.png' % (n_obs_list[i],dim_in_list[j],rff_dim_list[l],sig2_list[s],k)))
                        plt.close('all')

                    ## posterior of s (PLOT CODE SHOULD BE IMPROVED)
                    if args.model=='RFFHS' and args.layer_in_name=='RffVarSelectLogitNormalLayer':
                        eps = .0001
                        x = np.linspace(0+eps,1-eps,100)
                        logit = lambda x: np.log(x/(1-x))
                        def logitnormal_pdf(x, mu=0, sig2=1):
                            return 1/np.sqrt(2*np.pi*sig2)/(x*(1-x))*np.exp(-(logit(x)-mu)**2/(2*sig2))

                        fig, ax = plt.subplots()
                        for d in range(m.model.dim_in):
                            loc = m.model.layer_in.s_loc[d].detach().numpy()
                            scale = m.model.layer_in.transform(m.model.layer_in.s_scale_untrans[d]).detach().numpy()
                            ax.plot(x, logitnormal_pdf(x,loc,scale**2),label='s%d'%d)
                        fig.legend()
                        fig.savefig(os.path.join(args.dir_out, 
                                    'sposterior-n_obs=%d-dim_in=%d-rff_dim=%d-sig2%.2f-rep=%d.png' % (n_obs_list[i],dim_in_list[j],rff_dim_list[l],sig2_list[s],k)))
                        plt.close('all')

                    if args.compute_risk:
                        # could clean up code
                        n_samp_risk = 10
                        y_hat = np.zeros((n_samp_risk, Z.shape[0]))
                        y_hat_test = np.zeros((n_samp_risk, Z_test.shape[0]))
                        for ii in range(n_samp_risk):
                            y_hat[ii,:] = model.sample_f_post(Z)
                            y_hat_test[ii,:] = model.sample_f_post(Z_test)

                        # posterior predictive mean
                        y_hat = np.mean(y_hat, 0).reshape(-1,1)
                        y_hat_test = np.mean(y_hat_test, 0).reshape(-1,1)

                        # risk 
                        res['mse'][i,j,l,s,k] = np.mean((Y - y_hat)**2)
                        res['mse_test'][i,j,l,s,k] = np.mean((Y_test - y_hat_test)**2)
                        


np.save(os.path.join(args.dir_out, 'results.npy'), res)

# visualization
# res has dimensions: obs x dim_in x rff_dim x sig2 x rep x input
res = np.load(os.path.join(args.dir_out, 'results.npy'), allow_pickle=True).item()

# average over reps
psi_mean_mean = np.mean(res['psi_mean'], 4) # mean over psi samples, mean over reps
psi_mean_med = np.median(res['psi_mean'], 4) # mean over psi samples, median over reps

psi_var_mean = np.mean(res['psi_var'], 4) # variance over psi samples, mean over reps
psi_var_med = np.median(res['psi_var'], 4) # variance over psi samples, median over reps


## grid plots

# dim_in vs n_obs
idx_rff_dim = 0 # slice of rff_dim to plot
idx_sig2 = 0 # slice of sig2 to plot
fig, ax = util.plot_results_grid(psi_mean_mean[:,:,idx_rff_dim,idx_sig2,:], res['dim_in_list'], res['n_obs_list'], 'num. inputs', 'num. obs')
fig.savefig(os.path.join(args.dir_out, 'estimated_psi_dim_in_vs_n_obs.png'))

# rff_dim vs n_obs
idx_dim_in = -1 # slice of dim_in to plot
idx_sig2 = 0 # slice of sig2 to plot
fig, ax = util.plot_results_grid(psi_mean_mean[:,idx_dim_in,:,idx_sig2,:], res['rff_dim_list'], res['n_obs_list'], 'num. rff', 'num. obs')
fig.savefig(os.path.join(args.dir_out, 'estimated_psi_rff_dim_vs_n_obs.png'))

# n_obs vs rff_dim
idx_n_obs = 0 # slice of n_obs to plot
idx_sig2 = 0 # slice of sig2 to plot
fig, ax = util.plot_results_grid(psi_mean_mean[idx_n_obs,:,:,idx_sig2,:], res['rff_dim_list'], res['dim_in_list'], 'num. rff', 'num. inputs')
fig.savefig(os.path.join(args.dir_out, 'estimated_psi_rff_dim_vs_dim_in.png'))

# sig2 vs n_obs
idx_dim_in = -1 # slice of dim_in to plot
idx_rff_dim = 0 # slice of rff_dim to plot
fig, ax = util.plot_results_grid(psi_mean_mean[:,idx_dim_in,idx_rff_dim,:,:], res['sig2_list'], res['n_obs_list'], 'sig2', 'num. obs')
fig.savefig(os.path.join(args.dir_out, 'estimated_psi_sig2_vs_n_obs.png'))

## violin plots
idx_rff_dim = 0 # slice of rff_dim to plot
idx_sig2 = 0 # slice of sig2 to plot
for idx_dim_in, dim_in in enumerate(res['dim_in_list']):
    fig, ax = util.plot_results_dist(res['psi_mean'][:,idx_dim_in,idx_rff_dim,idx_sig2,:,:], dim_in, res['n_obs_list'], data_true=None)
    ax[0].set_title('distribution of variable importance $\psi$ \n(rbf_opt input, %d variables)' % dim_in)
    fig.savefig(os.path.join(args.dir_out, 'psi_dist_dim_in=%d.png' % dim_in))

