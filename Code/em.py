import math

import numpy as np
import pandas as pd
from skmisc.loess import loess
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt


def normal_pdf(x):
    # Normal probability density function

    return 1 / np.sqrt(2 * math.pi) * np.exp(-x ** 2 / 2)


def EM_initial_guess(data, times, nulls):
    # Initialize the EM algorithm as indicated in the report.
    # Returns the estimated sigma, the fitted loess and the uniform probabilities.
    n_genes, n_times = data.shape

    sigmas = np.sqrt(np.var(data, axis=1))

    fit_loess = np.zeros(data.shape)
    # For every gene
    for ix, row in data.iterrows():
        # Fit a lowess.
        model = loess(x=times, y=row)
        model.fit()
        fit_loess[ix] = model.predict(newdata=times).values

    # Set the probabilities p_j uniformly.
    n_0 = np.sum(np.sum(nulls))
    prob = n_0 / (2 * n_0 * n_genes)
    # p is 0.5 * probability to be 0
    p = [prob for _ in range(n_genes)]

    return sigmas, fit_loess, p


def EM_E_step(data, p, sigmas, fit_loess, nulls):
    # Implementation of E-step, as indicated in the report.
    q = np.zeros(data.shape)
    mu = np.zeros(data.shape)
    # For every point
    for ix, row in data.iterrows():
        for i in range(len(row)):
            y = row[i]
            s = sigmas[ix]

            # We find the mean of the truncated distribution mu.
            phi = lambda x: normal_pdf((y - x) / s)
            f = lambda x: fit_loess[ix][i] - x + s * (phi(x)) / (1 - phi(x))
            mu[ix][i] = fsolve(f, 0)

            # We update q, q being the probability that z_j(t) is 1, ie that y_j(t) is a TRUE 0.
            q[ix][i] = (p[ix] * normal_pdf((y - mu[ix][i]) / s)) / (1 - p[ix] + p[ix] * normal_pdf((y - mu[ix][i]) / s))

    # The probability of NOT being a dropped point for a
    q[data > 0] = 1
    return q


def EM_M_step(data, q, times, nulls):
    fit_loess = np.zeros(data.shape)
    p = np.zeros(data.shape[0])

    labels = np.zeros(data.shape)
    # labels[q>0.5] = 1

    w = np.copy(q)
    w[np.isnan(q)] = 1

    for ix, row in data.iterrows():
        # Update the function f_j for every gene by fitting a weighted loess.
        model = loess(x=times, y=row, weights=w[ix])
        model.fit()
        fit_loess[ix] = model.predict(times).values

        # Update the probabilities p_j.
        p[ix] = np.sum(q[ix]) / len(row)

    return fit_loess, p, labels


def EM_log_likelihood_calc(num_clusters, num_samples, data, mu, sigma, alpha):
    L = np.zeros((num_samples, num_clusters))
    for k in range(num_clusters):
        L[:, k] = alpha[k] * gaussian_pdf(data, mu[k], sigma[k])
    return np.sum(np.log(np.sum(L, axis=1)))


def EM(data, times, thresh=0.0000000001, max_iter=100):
    update = True
    nulls = data == 0
    sigmas, fit_loess, p = EM_initial_guess(data, times, nulls)
    n_iter = 0
    q = np.zeros(data.shape)

    while update > thresh and n_iter < max_iter:
        n_iter += 1
        old_q = np.copy(q)
        q = EM_E_step(data, p, sigmas, fit_loess, nulls)
        fit_loess, p, labels = EM_M_step(data, q, times, nulls)
        update = np.sum(np.sum(np.abs(q - old_q)))

        print("Iteration {}, update {}".format(n_iter, update))

    return q, fit_loess


def show_data(data, loess, times, labels):
    for ix, row in data.iterrows():
        plt.figure()
        plt.plot(times, loess[ix])
        plt.scatter(times, row, marker='.', c=labels[ix], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
        plt.xlabel('Time')
        plt.ylabel('Expression')
        plt.show()


if __name__ == '__main__':
    n_genes = 100
    print('Loading data...')
    data = pd.read_csv('../cache/synthetic_data.csv', index_col=0, nrows=n_genes)
    print('Loading times...')
    times = np.array(pd.read_csv('../cache/times.csv', index_col=0))
    times = times.reshape((500,))
    q, fit_loess = EM(data, times, thresh=0.00000000000000000000000001 / n_genes)
    labels = np.zeros(q.shape)
    labels[q > 0.5] = 1
    show_data(data, fit_loess, times, labels)
