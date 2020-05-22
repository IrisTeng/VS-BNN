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

parser.add_argument('--n_obs_true', type=int, default=200)

parser.add_argument('--dim_in_min', type=int, default=2)
parser.add_argument('--dim_in_max', type=int, default=5)
parser.add_argument('--dim_in_step', type=int, default=2)

parser.add_argument('--n_rep', type=int, default=2)

parser.add_argument('--model', type=str, default='GP', help='select "GP" or "RFF"')

# GP options
parser.add_argument('--opt_likelihood_variance', action='store_true')
parser.add_argument('--opt_kernel_hyperparam', action='store_true')
parser.add_argument('--kernel_lengthscale', type=float, default=1.0)
parser.add_argument('--kernel_variance', type=float, default=1.0)

# RFF options
parser.add_argument('--rff_dim_min', type=int, default=808)
parser.add_argument('--rff_dim_max', type=int, default=1200)
parser.add_argument('--rff_dim_step', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=16)


args = parser.parse_args()

if not os.path.exists(args.dir_out):
    os.makedirs(os.path.join(args.dir_out, 'output/'))

# allocate space
n_obs_list = util.arrange_full(args.n_obs_min, args.n_obs_max, args.n_obs_step)
dim_in_list = util.arrange_full(args.dim_in_min, args.dim_in_max, args.dim_in_step)
rff_dim_list = util.arrange_full(args.rff_dim_min, args.rff_dim_max, args.rff_dim_step)

print('running experiments for the following setups')
print('number of observations: ', n_obs_list)
print('number of input dimensions: ', dim_in_list)

## allocate space for results: obs x dim_in x rff_dim x rep x input
res_shape = (len(n_obs_list), len(dim_in_list), len(rff_dim_list), args.n_rep, np.max(dim_in_list))
res = {
    'psi_mean': np.full(res_shape, np.nan),
    'psi_var': np.full(res_shape, np.nan),
    'n_obs_list': n_obs_list,
    'dim_in_list': dim_in_list,
    'rff_dim_list': dim_in_list
    }

seed = 0
for i, n_obs in enumerate(n_obs_list):
    for j, dim_in in enumerate(dim_in_list):
        for l, rff_dim in enumerate(rff_dim_list):

            print('n_obs [%d/%d], dim_in [%d/%d], rff_dim [%d/%d]' % (i,len(n_obs_list),j,len(dim_in_list),l,len(rff_dim_list)))
        
            for k in range(args.n_rep):
                seed += 1

                Z, X, Y, sig2 = util.load_data(args.dataset_name, n_obs=n_obs, dim_in=dim_in, seed=seed)

                if sig2 is None:
                    sig2 = np.var(Y) # initial guess for sig2, could be improved?

                if args.subtract_covariates:
                    if X is None:
                        print('error: no covariates to subtract')
                    else:
                        Y = util.resid_linear_model(X,Y)

                if args.model=='GP':
                    m = models.GPyVarImportance(Z, Y, sig2=sig2, \
                        opt_kernel_hyperparam=args.opt_kernel_hyperparam, \
                        opt_sig2=args.opt_likelihood_variance,\
                        lengthscale=args.kernel_lengthscale, variance=args.kernel_variance)

                    m.train()
                
                elif args.model=='RFF':
                    m = models.RffVarImportance(Z)
                    m.train(Z, Y, sig2, rff_dim=rff_dim, batch_size=args.batch_size, epochs=args.epochs)

                psi_est = m.estimate_psi(Z)
                res['psi_mean'][i,j,l,k,:dim_in] = psi_est[0]
                res['psi_var'][i,j,l,k,:dim_in] = psi_est[1]


np.save(os.path.join(args.dir_out, 'results.npy'), res)



# visualization
# res has dimensions: obs x dim_in x rff_dim x rep x input
res = np.load(os.path.join(args.dir_out, 'results.npy'), allow_pickle=True).item()

# average over reps
psi_mean_mean = np.mean(res['psi_mean'], 3) # mean over psi samples, mean over reps
psi_mean_med = np.median(res['psi_mean'], 3) # mean over psi samples, median over reps

psi_var_mean = np.mean(res['psi_var'], 3) # variance over psi samples, mean over reps
psi_var_med = np.median(res['psi_var'], 3) # variance over psi samples, median over reps

## grid plots
idx_rff_dim = 0 # slice of rff_dim to plot
fig, ax = util.plot_results_grid(psi_mean_mean[:,:,idx_rff_dim,:], res['dim_in_list'], res['n_obs_list'], 'num. inputs', 'num. inputs')
fig.savefig(os.path.join(args.dir_out, 'estimated_psi_fixed_rff_dim.png'))

idx_dim_in = -1 # slice of dim_in to plot
fig, ax = util.plot_results_grid(psi_mean_mean[:,idx_dim_in,:,:], res['rff_dim_list'], res['n_obs_list'], 'num. rff', 'num. inputs')
fig.savefig(os.path.join(args.dir_out, 'estimated_psi_fixed_dim_in.png'))

idx_n_obs = 0 # slice of n_obs to plot
fig, ax = util.plot_results_grid(psi_mean_mean[idx_n_obs,:,:,:], res['n_obs_list'], res['rff_dim_list'], 'num. rff', 'num. inputs')
fig.savefig(os.path.join(args.dir_out, 'estimated_psi_fixed_n_obs.png'))


## violin plots
for idx_dim_in, dim_in in enumerate(res['dim_in_list']):

    fig, ax = util.plot_results_dist(np.squeeze(res['psi_mean'][:,idx_dim_in,:,:]), dim_in, res['n_obs_list'], data_true=None)
    ax[0].set_title('distribution of variable importance $\psi$ \n(rbf_opt input, %d variables)' % dim_in)
    fig.savefig(os.path.join(args.dir_out, 'psi_dist_dim_in=%d.png' % dim_in))

