import gpflow
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import csv

from exp.toy import rbf_toy
import exp.util as util
import exp.kernelized as kernel_layers

dataset = "rbf"
fixed_standarization=True
opt_kernel_hyperparam=False
opt_likelihood_variance=False
## data
dataset = dict(rbf=rbf_toy)[dataset]()
seed = 0
n_test = 50
EPOCHS = 16
batch_size = 16
data_dim = 1
rff_dim = 12
learning_rate = .01

n_train_all = np.arange(200, 900, 200)
n_rep = 2

## allocate space
n_all = n_train_all.size

# summary statistics for GP
rmse = np.zeros((n_all, n_rep))
picp = np.zeros((n_all, n_rep))
mpiw = np.zeros((n_all, n_rep))
test_log_likelihood = np.zeros((n_all, n_rep))

# summary statistics for woodbury-version covariance, rff
rmse2 = np.zeros((n_all, n_rep))
picp2 = np.zeros((n_all, n_rep))
mpiw2 = np.zeros((n_all, n_rep))
test_log_likelihood2 = np.zeros((n_all, n_rep))

# summary statistics for woodbury-version covariance, orf
rmse20 = np.zeros((n_all, n_rep))
picp20 = np.zeros((n_all, n_rep))
mpiw20 = np.zeros((n_all, n_rep))
test_log_likelihood20 = np.zeros((n_all, n_rep))

# summary statistics for population-version covariance, rff
rmse3 = np.zeros((n_all, n_rep))
picp3 = np.zeros((n_all, n_rep))
mpiw3 = np.zeros((n_all, n_rep))
test_log_likelihood3 = np.zeros((n_all, n_rep))

# summary statistics for woodbury-version covariance, rff
rmse29 = np.zeros((n_all, n_rep))
picp29 = np.zeros((n_all, n_rep))
mpiw29 = np.zeros((n_all, n_rep))
test_log_likelihood29 = np.zeros((n_all, n_rep))

# summary statistics for woodbury-version covariance, orf
rmse209 = np.zeros((n_all, n_rep))
picp209 = np.zeros((n_all, n_rep))
mpiw209 = np.zeros((n_all, n_rep))
test_log_likelihood209 = np.zeros((n_all, n_rep))

# summary statistics for population-version covariance, rff
rmse39 = np.zeros((n_all, n_rep))
picp39 = np.zeros((n_all, n_rep))
mpiw39 = np.zeros((n_all, n_rep))
test_log_likelihood39 = np.zeros((n_all, n_rep))


test_log_likelihood[:, 0] = -7.5
test_log_likelihood2[:, 0] = -7.5
test_log_likelihood3[:, 0] = -7.5
test_log_likelihood20[:, 0] = -7.5
test_log_likelihood29[:, 0] = -7.5
test_log_likelihood39[:, 0] = -7.5
test_log_likelihood209[:, 0] = -7.5

output_dir = 'output/rff_ss40/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## kernel
if dataset.name == 'rbf':
    kern = gpflow.kernels.RBF(input_dim=1, lengthscales=0.4, variance=1.0)
    dataset.sample_f(n_train_max=np.maximum(1000, np.max(n_train_all)), n_test=n_test, seed=seed)
    # initialize new function

elif dataset.name == 'x3':
    kern = gpflow.kernels.Polynomial(input_dim=1, degree=3, variance=1.0, offset=0.0)

elif dataset.name == 'sin':
    #kern = gpflow.kernels.Periodic(input_dim=1) + gpflow.kernels.RBF(input_dim=1)
    kern = gpflow.kernels.Periodic(input_dim=1, lengthscales=0.002, variance=2.2, period=273.4) + \
           gpflow.kernels.RBF(input_dim=1, lengthscales=31.6, variance=1.0)

elif dataset.name == 'comp':
    #kern = gpflow.kernels.Polynomial(degree=1, input_dim=1) * gpflow.kernels.Periodic(input_dim=1)
    kern = gpflow.kernels.Polynomial(degree=1, input_dim=1, offset=1.11, variance=1.58) * \
           gpflow.kernels.Periodic(input_dim=1, lengthscales=1.20, variance=1.33, period=1.31)


if fixed_standarization:
    x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
    mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
    mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)

csvFile = open("rff10000.csv", "a")
fileHeader = ["n_train", "n_rep",
              "lll_gp", "rmse_gp", "picp_gp", "mpiw_gp",
              "lll_rff", "rmse_rff", "picp_rff", "mpiw_rff",
              "lll_orf", "rmse_orf", "picp_orf", "mpiw_orf",
              "lll_rff_pop", "rmse_rff_pop", "picp_rff_pop", "mpiw_rff_pop",
              "lll_rff_d2", "rmse_rff_d2", "picp_rff_d2", "mpiw_rff_d2",
              "lll_orf_d2", "rmse_orf_d2", "picp_orf_d2", "mpiw_orf_d2",
              "lll_rff_pop_d2", "rmse_rff_pop_d2", "picp_rff_pop_d2", "mpiw_rff_pop_d2"]
writer = csv.writer(csvFile)


for i, n_train in enumerate(n_train_all):

    for j in range(n_rep):

        seed += 2

        original_x_train, original_y_train = \
            dataset.train_samples(n_data=n_train, seed=seed)
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

        plt.plot(test_x, test_y)
        plt.savefig('temp.png')

        ## GP model
        m = gpflow.models.GPR(train_x, train_y, kern=kern)

        # likelihood variance
        if not opt_likelihood_variance:
            m.likelihood.variance.trainable = False
            m.likelihood.variance = noise_std**2 # fixed observation variance

        # optimize hyperparameters
        if opt_kernel_hyperparam:
            #opt = gpflow.train.ScipyOptimizer() # Replace with AdamOptimizer?
            #opt.minimize(m)
            opt = gpflow.train.AdamOptimizer(.05) # Replace with AdamOptimizer?
            opt.minimize(m)

        noise_std_opt = m.likelihood.variance.read_value().item()
        print('noise: ', noise_std_opt)

        print(m.kern.as_pandas_table())

        ## posterior

        ## plot and compute summary statistics using GP
        xx = np.linspace(-2, 2, n_test).reshape(n_test, 1)
        mean, var = m.predict_y(xx)
        samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        #plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, mean, 'C0', lw=2)
        plt.fill_between(xx[:,0],
                         mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                         mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3,3)
        plt.savefig(output_dir + '/gp_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions
        f_pred, f_pred_cov = m.predict_f_full_cov(test_x)
        f_pred_cov = f_pred_cov[0]

        # predictive (i.e. including noise variance)
        y_pred_cov = f_pred_cov + noise_std_opt * np.eye(f_pred.shape[0])
        y_pred, y_pred_var = m.predict_y(test_x)  # y_pred = f_pred

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)
        # re_y_pred_cov = y_pred_cov

        ## evaluate posterior
        try:
            test_log_likelihood[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        except:
            test_log_likelihood[i, j] = test_log_likelihood[i, j-1]

        rmse[i, j] = util.rmse(test_y, y_pred)
        picp[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw[i, j] = util.mpiw(y_pred_lb, y_pred_ub)


        # extract rff features
        model_graph = tf.Graph()
        model_sess = tf.Session(graph=model_graph)

        with model_graph.as_default():
            X = tf.placeholder(dtype=tf.float64, shape=[None, data_dim])
            Y_true = tf.placeholder(dtype=tf.float64, shape=[None, 1])
            H_inv = tf.placeholder(dtype=tf.float64, shape=[rff_dim, rff_dim])
            Phi_y = tf.placeholder(dtype=tf.float64, shape=[rff_dim, 1])

            global_step = tf.Variable(0, trainable=False)

            rff_layer = kernel_layers.RandomFourierFeatures(output_dim=rff_dim,
                                                            kernel_initializer='gaussian',
                                                            scale=0.4)
            dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                kernel_regularizer=keras.regularizers.l2(l=noise_std ** 2))

            ## define model
            rff_output = tf.cast(rff_layer(X) * np.sqrt(2. / rff_dim), dtype=tf.float64)
            Y_pred = dense_layer(rff_output)

            ## define loss
            mean_loss = tf.losses.mean_squared_error(Y_true, Y_pred)

            ## define training
            train_mean = tf.train.AdamOptimizer(learning_rate
                                                ).minimize(mean_loss,
                                                           global_step=global_step)

            tf.summary.scalar("loss", mean_loss)
            tf.summary.histogram("histogram loss", mean_loss)
            summary_op = tf.summary.merge_all()

            weight_cov = util.minibatch_woodbury_update(rff_output, H_inv)

            covl_xy = util.minibatch_interaction_update(Phi_y, rff_output, Y_true)

            pred_cov = tf.matmul(rff_output, tf.matmul(weight_cov,
                                                       rff_output, transpose_b=True))

        ### Training and Evaluation ###
        X_batches = util.split_into_batches(train_x, batch_size) * EPOCHS
        Y_batches = util.split_into_batches(train_y, batch_size) * EPOCHS

        num_steps = X_batches.__len__()
        num_batch = int(num_steps / EPOCHS)

        with model_sess as sess:
            sess.run(tf.global_variables_initializer())

            writer2 = tf.summary.FileWriter('./graphs', model_graph)

            rff_1 = sess.run(rff_output, feed_dict={X: X_batches[0]})
            weight_cov_val = util.compute_inverse(rff_1, sig_sq=noise_std ** 2)
            covl_xy_val = np.matmul(rff_1.T, Y_batches[0])

            for batch_id in range(1, num_batch):
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                try:
                    weight_cov_val, covl_xy_val = sess.run([weight_cov, covl_xy],
                                                           feed_dict={X: X_batch,
                                                                      Y_true: Y_batch,
                                                                      H_inv: weight_cov_val,
                                                                      Phi_y: covl_xy_val})
                except:
                    print("\n================================\n"
                          "Problem occurred at Step {}\n"
                          "================================".format(batch_id))

            weight_tmp = weight_cov_val
            covl_tmp = covl_xy_val

            beta = np.matmul(weight_cov_val, covl_xy_val)

            for batch_id in range(0, num_steps):
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                try:
                    _, summary = sess.run([train_mean, summary_op],
                                          feed_dict={X: X_batch,
                                                     Y_true: Y_batch})

                except:
                    print("\n================================\n"
                          "Problem occurred at Step {}\n"
                          "================================".format(batch_id))

            # beta2 = np.matmul(weight_cov_val, covl_xy_val)

            weight_cov_val = weight_tmp * noise_std ** 2
            # weight_cov_val2 = weight_cov_val * noise_std ** 2

            ## prediction using woodbury-version covariance
            pred_mean_xx, pre_cov_xx = \
                sess.run([Y_pred, pred_cov],
                         feed_dict={X: xx, Y_true: test_y,
                                    H_inv: weight_cov_val})

            pred_mean_tst, pre_cov_tst = \
                sess.run([Y_pred, pred_cov],
                         feed_dict={X: test_x,
                                    Y_true: test_y,
                                    H_inv: weight_cov_val})

            # compute rff, to compute population-version covariance later
            rff_output_val = sess.run(rff_output, feed_dict={X: xx})
            train_x_val = sess.run(rff_output, feed_dict={X: train_x})
            tst_xx_val = sess.run(rff_output, feed_dict={X: test_x})


        ## weight covariance matrix for all data, should equal to weight_tmp
        weight_cov_all = util.compute_inverse(train_x_val, sig_sq=noise_std ** 2)
        print("weight difference (wo comment): ", np.max(np.abs(weight_tmp - weight_cov_all)))

        covl_xy_all = np.matmul(train_x_val.T, train_y)
        print("covl difference (wo comment): ", np.max(np.abs(covl_tmp - covl_xy_all)))

        beta_all = np.matmul(weight_cov_all, covl_xy_all)
        print("beta difference (wo comment): ", np.max(np.abs(beta - beta_all)))

        ## plot and compute summary statistics using woodbury-version covariance
        yy_pred_cov = pre_cov_xx + noise_std ** 2 * np.eye(n_test)
        var_xx = np.diagonal(yy_pred_cov).reshape(-1, 1)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/rff_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions

        f_pred = pred_mean_tst

        # predictive (i.e. including noise variance)
        y_pred_cov = pre_cov_tst + noise_std ** 2 * np.eye(f_pred.shape[0])
        y_pred = f_pred  # y_pred = f_pred
        y_pred_var = np.diagonal(y_pred_cov).reshape(-1, 1)

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)

        ## evaluate posterior
        try:
            test_log_likelihood2[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        except:
            test_log_likelihood2[i, j] = test_log_likelihood2[i, j-1]

        rmse2[i, j] = util.rmse(test_y, y_pred)
        picp2[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw2[i, j] = util.mpiw(y_pred_lb, y_pred_ub)


        ## plot and compute summary statistics using bayesian-lnr mean prediction
        pred_mean_xx = np.matmul(rff_output_val, beta)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/lnr_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions

        f_pred = np.matmul(tst_xx_val, beta)

        # predictive (i.e. including noise variance)
        y_pred = f_pred  # y_pred = f_pred

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)

        ## evaluate posterior
        try:
            test_log_likelihood20[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        except:
            test_log_likelihood20[i, j] = test_log_likelihood20[i, j - 1]

        rmse20[i, j] = util.rmse(test_y, y_pred)
        picp20[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw20[i, j] = util.mpiw(y_pred_lb, y_pred_ub)


        ## plot and compute summary statistics using population-version covariance
        cov_xx_marg = np.matmul(rff_output_val, rff_output_val.T)
        covl = np.matmul(rff_output_val, train_x_val.T)
        train_x_inv = np.linalg.inv(np.matmul(train_x_val, train_x_val.T)
                                    + noise_std ** 2 * np.identity(n_train))

        beta_pop = np.matmul(train_x_val.T, np.matmul(train_x_inv, train_y))
        print("beta_pop difference (wo comment): ", np.max(np.abs(beta - beta_pop)))

        pred_mean_xx = np.matmul(covl, np.matmul(train_x_inv, train_y))

        pre_cov_xx = cov_xx_marg - np.matmul(covl, np.matmul(train_x_inv, covl.T))
        yy_pred_cov = pre_cov_xx + noise_std ** 2 * np.eye(n_test)
        var_xx = np.diagonal(yy_pred_cov).reshape(-1, 1)

        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/rff-pop_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions
        cov_xx_marg3 = np.matmul(tst_xx_val, tst_xx_val.T)
        covl3 = np.matmul(tst_xx_val, train_x_val.T)
        train_x_inv = np.linalg.inv(np.matmul(train_x_val, train_x_val.T)
                                    + noise_std ** 2 * np.identity(n_train))

        pred_mean_tst = np.matmul(covl3, np.matmul(train_x_inv, train_y))

        pre_cov_tst = cov_xx_marg3 - np.matmul(covl3, np.matmul(train_x_inv, covl3.T))

        f_pred = pred_mean_tst

        # predictive (i.e. including noise variance)
        y_pred_cov = pre_cov_tst + noise_std ** 2 * np.eye(f_pred.shape[0])
        y_pred = f_pred  # y_pred = f_pred
        y_pred_var = np.diagonal(y_pred_cov).reshape(-1, 1)

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)

        ## evaluate posterior
        try:
            test_log_likelihood3[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        except:
            test_log_likelihood3[i, j] = test_log_likelihood3[i, j - 1]

        rmse3[i, j] = util.rmse(test_y, y_pred)
        picp3[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw3[i, j] = util.mpiw(y_pred_lb, y_pred_ub)

        # extract rff features
        model_graph = tf.Graph()
        model_sess = tf.Session(graph=model_graph)

        with model_graph.as_default():
            X = tf.placeholder(dtype=tf.float64, shape=[None, data_dim])
            Y_true = tf.placeholder(dtype=tf.float64, shape=[None, 1])
            H_inv = tf.placeholder(dtype=tf.float64, shape=[rff_dim, rff_dim])
            Phi_y = tf.placeholder(dtype=tf.float64, shape=[rff_dim, 1])

            global_step = tf.Variable(0, trainable=False)

            rff_layer = kernel_layers.RandomFourierFeatures(output_dim=rff_dim,
                                                            kernel_initializer='gaussian',
                                                            scale=0.4)
            dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                kernel_regularizer=keras.regularizers.l2(l=noise_std ** 2))

            ## define model
            rff_output = tf.cast(rff_layer(X) * np.sqrt(2. / rff_dim), dtype=tf.float64)
            Y_pred = dense_layer(rff_output)

            ## define loss
            mean_loss = tf.losses.mean_squared_error(Y_true, Y_pred)

            ## define training
            train_mean = tf.train.AdamOptimizer(learning_rate
                                                ).minimize(mean_loss,
                                                           global_step=global_step)

            tf.summary.scalar("loss", mean_loss)
            tf.summary.histogram("histogram loss", mean_loss)
            summary_op = tf.summary.merge_all()

            weight_cov = util.minibatch_woodbury_update(rff_output, H_inv)

            covl_xy = util.minibatch_interaction_update(Phi_y, rff_output, Y_true)

            pred_cov = tf.matmul(rff_output, tf.matmul(weight_cov,
                                                       rff_output, transpose_b=True))

        ### Training and Evaluation ###

        with model_sess as sess:
            sess.run(tf.global_variables_initializer())

            writer2 = tf.summary.FileWriter('./graphs', model_graph)

            rff_1 = sess.run(rff_output, feed_dict={X: X_batches[0]})
            weight_cov_val = util.compute_inverse(rff_1, sig_sq=noise_std ** 2)
            covl_xy_val = np.matmul(rff_1.T, Y_batches[0])

            for batch_id in range(1, num_batch):
                X_batch = X_batches[batch_id]
                Y_batch = Y_batches[batch_id]

                ## update posterior mean/covariance
                try:
                    weight_cov_val, covl_xy_val = sess.run([weight_cov, covl_xy],
                                                           feed_dict={X: X_batch,
                                                                      Y_true: Y_batch,
                                                                      H_inv: weight_cov_val,
                                                                      Phi_y: covl_xy_val})
                except:
                    print("\n================================\n"
                          "Problem occurred at Step {}\n"
                          "================================".format(batch_id))

            weight_tmp = weight_cov_val
            covl_tmp = covl_xy_val

            beta = np.matmul(weight_cov_val, covl_xy_val)

            # for batch_id in range(0, num_steps):
            #     X_batch = X_batches[batch_id]
            #     Y_batch = Y_batches[batch_id]
            #
            #     ## update posterior mean/covariance
            #     try:
            #         _, summary = sess.run([train_mean, summary_op],
            #                               feed_dict={X: X_batch,
            #                                          Y_true: Y_batch})
            #
            #     except:
            #         print("\n================================\n"
            #               "Problem occurred at Step {}\n"
            #               "================================".format(batch_id))

            # beta2 = np.matmul(weight_cov_val, covl_xy_val)

            weight_cov_val = weight_tmp * noise_std ** 2
            # weight_cov_val2 = weight_cov_val * noise_std ** 2

            ## prediction using woodbury-version covariance
            pred_mean_xx, pre_cov_xx = \
                sess.run([Y_pred, pred_cov],
                         feed_dict={X: xx, Y_true: test_y,
                                    H_inv: weight_cov_val})

            pred_mean_tst, pre_cov_tst = \
                sess.run([Y_pred, pred_cov],
                         feed_dict={X: test_x,
                                    Y_true: test_y,
                                    H_inv: weight_cov_val})

            # compute rff, to compute population-version covariance later
            rff_output_val = sess.run(rff_output, feed_dict={X: xx})
            train_x_val = sess.run(rff_output, feed_dict={X: train_x})
            tst_xx_val = sess.run(rff_output, feed_dict={X: test_x})

        ## weight covariance matrix for all data, should equal to weight_tmp
        weight_cov_all = util.compute_inverse(train_x_val, sig_sq=noise_std ** 2)
        print("weight difference (w comment): ", np.max(np.abs(weight_tmp - weight_cov_all)))

        covl_xy_all = np.matmul(train_x_val.T, train_y)
        print("covl difference (w comment): ", np.max(np.abs(covl_tmp - covl_xy_all)))

        beta_all = np.matmul(weight_cov_all, covl_xy_all)
        print("beta difference (w comment): ", np.max(np.abs(beta - beta_all)))

        ## plot and compute summary statistics using woodbury-version covariance
        yy_pred_cov = pre_cov_xx + noise_std ** 2 * np.eye(n_test)
        var_xx = np.diagonal(yy_pred_cov).reshape(-1, 1)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/rff_d2_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions

        f_pred = pred_mean_tst

        # predictive (i.e. including noise variance)
        y_pred_cov = pre_cov_tst + noise_std ** 2 * np.eye(f_pred.shape[0])
        y_pred = f_pred  # y_pred = f_pred
        y_pred_var = np.diagonal(y_pred_cov).reshape(-1, 1)

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)

        ## evaluate posterior
        try:
            test_log_likelihood29[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        except:
            test_log_likelihood29[i, j] = test_log_likelihood29[i, j - 1]

        rmse29[i, j] = util.rmse(test_y, y_pred)
        picp29[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw29[i, j] = util.mpiw(y_pred_lb, y_pred_ub)

        ## plot and compute summary statistics using bayesian-lnr mean prediction
        pred_mean_xx = np.matmul(rff_output_val, beta)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/lnr_d2_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions

        f_pred = np.matmul(tst_xx_val, beta)

        # predictive (i.e. including noise variance)
        y_pred = f_pred  # y_pred = f_pred

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)

        ## evaluate posterior
        try:
            test_log_likelihood209[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        except:
            test_log_likelihood209[i, j] = test_log_likelihood209[i, j - 1]

        rmse209[i, j] = util.rmse(test_y, y_pred)
        picp209[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw209[i, j] = util.mpiw(y_pred_lb, y_pred_ub)

        ## plot and compute summary statistics using population-version covariance
        cov_xx_marg = np.matmul(rff_output_val, rff_output_val.T)
        covl = np.matmul(rff_output_val, train_x_val.T)
        train_x_inv = np.linalg.inv(np.matmul(train_x_val, train_x_val.T)
                                    + noise_std ** 2 * np.identity(n_train))

        beta_pop = np.matmul(train_x_val.T, np.matmul(train_x_inv, train_y))
        print("beta_pop difference (w comment): ", np.max(np.abs(beta - beta_pop)))

        pred_mean_xx = np.matmul(covl, np.matmul(train_x_inv, train_y))

        pre_cov_xx = cov_xx_marg - np.matmul(covl, np.matmul(train_x_inv, covl.T))
        yy_pred_cov = pre_cov_xx + noise_std ** 2 * np.eye(n_test)
        var_xx = np.diagonal(yy_pred_cov).reshape(-1, 1)

        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        # plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, pred_mean_xx, 'C0', lw=2)
        plt.fill_between(xx[:, 0],
                         pred_mean_xx[:, 0] - 1.96 * np.sqrt(var_xx[:, 0]),
                         pred_mean_xx[:, 0] + 1.96 * np.sqrt(var_xx[:, 0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3, 3)
        plt.savefig(output_dir + '/rff-pop_d2_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions
        cov_xx_marg3 = np.matmul(tst_xx_val, tst_xx_val.T)
        covl3 = np.matmul(tst_xx_val, train_x_val.T)
        train_x_inv = np.linalg.inv(np.matmul(train_x_val, train_x_val.T)
                                    + noise_std ** 2 * np.identity(n_train))

        pred_mean_tst = np.matmul(covl3, np.matmul(train_x_inv, train_y))

        pre_cov_tst = cov_xx_marg3 - np.matmul(covl3, np.matmul(train_x_inv, covl3.T))

        f_pred = pred_mean_tst

        # predictive (i.e. including noise variance)
        y_pred_cov = pre_cov_tst + noise_std ** 2 * np.eye(f_pred.shape[0])
        y_pred = f_pred  # y_pred = f_pred
        y_pred_var = np.diagonal(y_pred_cov).reshape(-1, 1)

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)

        ## evaluate posterior
        try:
            test_log_likelihood39[i, j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        except:
            test_log_likelihood39[i, j] = test_log_likelihood39[i, j - 1]

        rmse39[i, j] = util.rmse(test_y, y_pred)
        picp39[i, j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw39[i, j] = util.mpiw(y_pred_lb, y_pred_ub)

        add_info = [n_train, j,
                    test_log_likelihood[i, j], rmse[i, j], picp[i, j], mpiw[i, j],
                    test_log_likelihood2[i, j], rmse2[i, j], picp2[i, j], mpiw2[i, j],
                    test_log_likelihood20[i, j], rmse20[i, j], picp20[i, j], mpiw20[i, j],
                    test_log_likelihood3[i, j], rmse3[i, j], picp3[i, j], mpiw3[i, j],
                    test_log_likelihood29[i, j], rmse29[i, j], picp29[i, j], mpiw29[i, j],
                    test_log_likelihood209[i, j], rmse209[i, j], picp209[i, j], mpiw209[i, j],
                    test_log_likelihood39[i, j], rmse39[i, j], picp39[i, j], mpiw39[i, j]]

        writer.writerow(add_info)
        print(add_info)

csvFile.close()
