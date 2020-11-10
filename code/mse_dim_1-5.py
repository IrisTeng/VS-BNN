import gpflow
import numpy as np
import os
import argparse
import exp.util as util
import exp.models as models
import exp.toy as toy
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='rbf')
parser.add_argument('--dir_out', type=str, default='output/')
parser.add_argument('--subtract_covariates', action='store_true',
                    help='use Y - beta*X as outcome')

parser.add_argument('--n_obs_min', type=int, default=250)
parser.add_argument('--n_obs_max', type=int, default=2000)
parser.add_argument('--n_obs_step', type=int, default=250)

parser.add_argument('--n_obs_true', type=int, default=200)

parser.add_argument('--dim_in_min', type=int, default=1)
parser.add_argument('--dim_in_max', type=int, default=5)
parser.add_argument('--dim_in_step', type=int, default=1)

parser.add_argument('--n_rep', type=int, default=10)

parser.add_argument('--model', type=str, default='RFF',
                    help='select "GP" or "RFF"')

# GP options
parser.add_argument('--opt_likelihood_variance', action='store_true')
parser.add_argument('--opt_kernel_hyperparam', action='store_true')
parser.add_argument('--kernel_lengthscale', type=float, default=1.0)
parser.add_argument('--kernel_variance', type=float, default=1.0)

# RFF options
parser.add_argument('--rff_dim_min', type=int, default=1200)
parser.add_argument('--rff_dim_max', type=int, default=1200)
parser.add_argument('--rff_dim_step', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=16)

# observational noise variance (sig2) options
parser.add_argument('--sig2_min', type=float, default=np.sqrt(.001))
parser.add_argument('--sig2_max', type=float, default=np.sqrt(.001))
parser.add_argument('--sig2_n_step', type=float, default=1)


args = parser.parse_args()

if not os.path.exists(args.dir_out):
    os.makedirs(os.path.join(args.dir_out, 'output/'))

csvFile = open("mse_rff.csv", "a")
fileHeader = ["dim", "n_obs", "std_MSE", "std_MSE_gp"]
writer = csv.writer(csvFile)



# allocate space
n_obs_list = util.arrange_full(args.n_obs_min, args.n_obs_max, args.n_obs_step)
n_obs_list = np.append(100, n_obs_list)
dim_in_list = util.arrange_full(args.dim_in_min, args.dim_in_max, args.dim_in_step)
rff_dim_list = util.arrange_full(args.rff_dim_min, args.rff_dim_max, args.rff_dim_step)
sig2_list = np.linspace(args.sig2_min, args.sig2_max, args.sig2_n_step)

print('running experiments for the following setups')
print('number of observations: ', n_obs_list)
print('number of input dimensions: ', dim_in_list)
print('observational noises: ', sig2_list)

fixed_standarization=True
opt_kernel_hyperparam=False
opt_likelihood_variance=False
batch_size = 16
epochs = 1
n_test=100
seed = 0

# generate data from GP
#############################################################################
for i, n_obs in enumerate(n_obs_list):
    for j, dim_in in enumerate(dim_in_list):
        for l, rff_dim in enumerate(rff_dim_list):
            for s, sig2 in enumerate(sig2_list):

                print('n_obs [%d/%d], dim_in [%d/%d], rff_dim [%d/%d], sig2 [%d/%d]' % \
                    (i,len(n_obs_list),j,len(dim_in_list),l,
                     len(rff_dim_list),s,len(sig2_list)))

                std_MSE = 0
                std_MSE_gp = 0
                for k in range(args.n_rep):
                    seed += 1

                    dataset = toy.rbf_data(dim_in=dim_in)
                    kern = gpflow.kernels.RBF(input_dim=dim_in, lengthscales=0.4, variance=1.0)
                    dataset.sample_f(n_train_max=2000, n_test=n_test, seed=seed)

                    if fixed_standarization:
                        x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
                        mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
                        mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)

                    original_x_train, original_y_train = \
                        dataset.train_samples(n_data=n_obs, seed=seed)
                    original_x_test, original_y_test = \
                        dataset.test_samples(n_data=n_test, seed=seed + 1)

                    # standarize data
                    if fixed_standarization:
                        train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                                 original_x_test, original_y_test,
                                                                                 mean_x_standard, std_x_standard,
                                                                                 mean_y_standard, std_y_standard)
                        noise_std = dataset.y_std / std_y_standard

                    else:
                        train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                                 original_x_test, original_y_test)
                        noise_std = dataset.y_std / np.std(original_y_train)

                    ## GP model
                    m = gpflow.models.GPR(train_x, train_y, kern=kern)

                    # likelihood variance
                    if not opt_likelihood_variance:
                        m.likelihood.variance.trainable = False
                        m.likelihood.variance = noise_std ** 2  # fixed observation variance

                    # optimize hyperparameters
                    if opt_kernel_hyperparam:
                        # opt = gpflow.train.ScipyOptimizer() # Replace with AdamOptimizer?
                        # opt.minimize(m)
                        opt = gpflow.train.AdamOptimizer(.05)  # Replace with AdamOptimizer?
                        opt.minimize(m)

                    mean_gp, var = m.predict_y(test_x)

                    ## RFF model
                    m2 = models.RffVarImportance(train_x)
                    m2.train(train_x, train_y, sig2=noise_std, rff_scale=0.4, rff_dim=rff_dim,
                             batch_size=batch_size, epochs=epochs)
                    pred_tst, pred_cov_tst = m2.predict(test_x)

                    RR = np.mean((test_y - np.mean(test_y)) ** 2)
                    std_MSE_tmp = np.mean((pred_tst - test_y) ** 2) / RR
                    std_MSE_tmp2 = np.mean((mean_gp - test_y) ** 2) / RR
                    std_MSE += std_MSE_tmp
                    std_MSE_gp += std_MSE_tmp2

                std_MSE = std_MSE / args.n_rep
                std_MSE_gp = std_MSE_gp / args.n_rep
                add_info = [dim_in, n_obs, std_MSE, std_MSE_gp]
                writer.writerow(add_info)
                print(add_info)

csvFile.close()




# generate data from RFF
#############################################################################
for j, dim_in in enumerate(dim_in_list):
    for l, rff_dim in enumerate(rff_dim_list):
        for s, sig2 in enumerate(sig2_list):

            print('dim_in [%d/%d], rff_dim [%d/%d], sig2 [%d/%d]' % \
                  (j,len(dim_in_list),l, len(rff_dim_list),s,len(sig2_list)))

            # generate dataset0 to extract RFF parameters
            dataset0 = toy.rbf_data(dim_in=dim_in)
            dataset0.sample_f(n_train_max=2000, n_test=n_test, seed=seed)

            if fixed_standarization:
                x_standard, y_standard = dataset0.train_samples(n_data=1000, seed=0)
                mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
                mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)

            original_x_train, original_y_train = \
                dataset0.train_samples(n_data=500, seed=seed)
            original_x_test, original_y_test = \
                dataset0.test_samples(n_data=n_test, seed=seed + 1)

            # standarize data
            if fixed_standarization:
                train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                         original_x_test, original_y_test,
                                                                         mean_x_standard, std_x_standard,
                                                                         mean_y_standard, std_y_standard)
                noise_std = dataset0.y_std / std_y_standard

            else:
                train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                         original_x_test, original_y_test)
                noise_std = dataset0.y_std / np.std(original_y_train)

            m0 = models.RffVarImportance(train_x)
            m0.train(train_x, train_y, sig2=noise_std, rff_scale=0.4, rff_dim=rff_dim,
                     batch_size=batch_size, epochs=epochs)
            W0, b, beta_rff, Sigma_beta = m0.return_value()

            for i, n_obs in enumerate(n_obs_list):

                std_MSE = 0
                std_MSE_gp = 0
                for k in range(args.n_rep):
                    seed += 1

                    dataset = toy.rff_data(dim_in=dim_in)
                    kern = gpflow.kernels.RBF(input_dim=dim_in, lengthscales=0.4, variance=1.0)
                    dataset.sample_f(W0, b, beta_rff, n_train_max=2000, n_test=n_test, seed=seed)

                    if fixed_standarization:
                        x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
                        mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
                        mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)

                    original_x_train, original_y_train = \
                        dataset.train_samples(n_data=n_obs, seed=seed)
                    original_x_test, original_y_test = \
                        dataset.test_samples(n_data=n_test, seed=seed + 1)

                    # standarize data
                    if fixed_standarization:
                        train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train,
                                                                                 original_x_test, original_y_test,
                                                                                 mean_x_standard, std_x_standard,
                                                                                 mean_y_standard, std_y_standard)
                        noise_std = dataset.y_std / std_y_standard

                    else:
                        train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train,
                                                                                 original_x_test, original_y_test)
                        noise_std = dataset.y_std / np.std(original_y_train)

                    ## GP model
                    m = gpflow.models.GPR(train_x, train_y, kern=kern)

                    # likelihood variance
                    if not opt_likelihood_variance:
                        m.likelihood.variance.trainable = False
                        m.likelihood.variance = noise_std ** 2  # fixed observation variance

                    # optimize hyperparameters
                    if opt_kernel_hyperparam:
                        # opt = gpflow.train.ScipyOptimizer() # Replace with AdamOptimizer?
                        # opt.minimize(m)
                        opt = gpflow.train.AdamOptimizer(.05)  # Replace with AdamOptimizer?
                        opt.minimize(m)

                    mean_gp, var = m.predict_y(test_x)

                    ## RFF model
                    test_x_val = np.sqrt(2. / W0.shape[1]) * np.cos(np.matmul(test_x, W0) + b)
                    pred_tst = np.matmul(test_x_val, beta_rff)

                    RR = np.mean((test_y - np.mean(test_y)) ** 2)
                    std_MSE_tmp = np.mean((pred_tst - test_y) ** 2) / RR
                    std_MSE_tmp2 = np.mean((mean_gp - test_y) ** 2) / RR
                    std_MSE += std_MSE_tmp
                    std_MSE_gp += std_MSE_tmp2

                std_MSE = std_MSE / args.n_rep
                std_MSE_gp = std_MSE_gp / args.n_rep
                add_info = [dim_in, n_obs, std_MSE, std_MSE_gp]
                writer.writerow(add_info)
                print(add_info)
csvFile.close()
