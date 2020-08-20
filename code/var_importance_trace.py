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

parser.add_argument('--n_obs', type=int, default=10)
parser.add_argument('--dim_in', type=int, default=2)
parser.add_argument('--n_rep', type=int, default=2)

parser.add_argument('--model', type=str, default='RFFHS', help='select "GP" or "RFF"')

parser.add_argument('--reg_param_list', type=str)

parser.add_argument('--sig2', type=float, default=.01)

# GP options
parser.add_argument('--opt_likelihood_variance', action='store_true')
parser.add_argument('--opt_kernel_hyperparam', action='store_true')
parser.add_argument('--kernel_lengthscale', type=float, default=1.0)
parser.add_argument('--kernel_variance', type=float, default=1.0)

# RFF options
parser.add_argument('--rff_dim', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=16)

args = parser.parse_args()

if not os.path.exists(args.dir_out):
    os.makedirs(os.path.join(args.dir_out, 'output/'))

with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
    f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))

# allocate space
#reg_param_list = util.arrange_full(args.reg_param_min, args.reg_param_max, args.reg_param_step)
reg_param_list = eval(args.reg_param_list)

print('running experiments for the following setups')
print('regularization parameters: ', reg_param_list)

## allocate space for results: reg_param x rep x input
res_shape = (len(reg_param_list), args.n_rep, args.dim_in)

res = {
    'psi_mean': np.full(res_shape, np.nan),
    'psi_var': np.full(res_shape, np.nan),
    'reg_param_list': reg_param_list
    }

for i, reg_param in enumerate(reg_param_list):

    print('reg_param [%d/%d]' % (i, len(reg_param_list)))

    for k in range(args.n_rep):
        seed = k

        Z, X, Y, sig2 = util.load_data(args.dataset_name, n_obs=args.n_obs, dim_in=args.dim_in, sig2=args.sig2, seed=seed)

        Z, Y = util.standardize_data(Z, Y)

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
            #print('GP lengthscale: ', m.model.kern.lengthscale)
            #print('GP variance: ', m.model.kern.variance)
            print('model: ', m.model)
    
        elif args.model=='RFF':
            m = models.RffVarImportance(Z)
            m.train(Z, Y, sig2, rff_dim=args.rff_dim, batch_size=args.batch_size, epochs=args.epochs)

        elif args.model=='RFF-PYTORCH':
            m = models.RffVarImportancePytorch(Z, Y, dim_hidden=args.rff_dim, sig2_inv=1/sig2)
            m.train()

        elif args.model=='RFFHS':
            m = models.RffHsVarImportance(Z, Y, dim_hidden=args.rff_dim, sig2_inv=1/sig2, infer_nu=False, nu=reg_param)
            m.train(n_epochs=args.epochs)

        elif args.model=='BKMR':
            m = models.BKMRVarImportance(Z, Y, sig2)
            m.train()

        psi_est = m.estimate_psi(Z)
        res['psi_mean'][i,k,:] = psi_est[0]
        res['psi_var'][i,k,:] = psi_est[1]

        ## slices
        if hasattr(m, 'sample_f_post'):
            fig, ax = util.plot_slices(m.sample_f_post, Z, Y, quantile=.5, n_samp=500, figsize=(4*args.dim_in,4))
            fig.savefig(os.path.join(args.dir_out,'reg_param=%f.png' % reg_param))
            plt.close('all')
                        

np.save(os.path.join(args.dir_out, 'results.npy'), res)

# visualization
# res has dimensions: reg_param x rep x input
res = np.load(os.path.join(args.dir_out, 'results.npy'), allow_pickle=True).item()

# average over reps
psi_mean_mean = np.mean(res['psi_mean'], 1) # mean over psi samples, mean over reps
psi_mean_med = np.median(res['psi_mean'], 1) # mean over psi samples, median over reps

psi_var_mean = np.mean(res['psi_var'], 1) # variance over psi samples, mean over reps
psi_var_med = np.median(res['psi_var'], 1) # variance over psi samples, median over reps

## grid plots
fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('Accent')
for i in range(args.dim_in):
    lb = psi_mean_mean[:,i] - 1.96*psi_var_mean[:,i]
    ub = psi_mean_mean[:,i] + 1.96*psi_var_mean[:,i]
    ax.plot(reg_param_list.reshape(-1), psi_mean_mean[:,i], '-o', color=cmap(i))
    ax.fill_between(reg_param_list.reshape(-1), lb, ub, alpha=.5, color=cmap(i))

fig.savefig(os.path.join(args.dir_out, 'trace.png'))


