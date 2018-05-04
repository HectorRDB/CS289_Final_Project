import math

import numpy as np
import pandas as pd
from skmisc.loess import loess
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm


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
    prob = n_0 / (2 * n_times * n_genes)
    # p is 0.5 * probability to be 0
    p = [0.9 for _ in range(n_genes)]

    return sigmas, fit_loess, p


def EM_E_step(data, p, sigmas, fit_loess):
    # Implementation of E-step, as indicated in the report.
    q = np.zeros(data.shape)

    # For every point
    for ix, row in data.iterrows():
        s = sigmas[ix]

        for i in range(len(row)):

            # If the observed value is 0:
            if row[i] == 0:
                f = fit_loess[ix][i]

                Phi = norm.cdf((- f / s))

                # Proba to be a dropped 0:
                p1 = 1 - p[ix]

                # Proba to be a real 0:
                p2 = p[ix] * Phi

                # We update q, q being the probability that the point is a TRUE 0.
                q[ix][i] = p2 / (p1 + p2)

            # If the observed value if not 0, the point is not dropped.
            else:
                q[ix][i] = 1

    return q


def EM_M_step(data, q, times):
    fit_loess = np.zeros(data.shape)
    p = np.zeros(data.shape[0])

    for ix, row in data.iterrows():
        # Update the function f_j for every gene by fitting a weighted loess.

        model = loess(x=times, y=row, weights=q[ix])
        model.fit()
        fit_loess[ix] = model.predict(times).values

        # Update the probabilities p_j
        p[ix] = np.mean(q[ix])

    return fit_loess, p


def EM_log_likelihood_calc(num_clusters, num_samples, data, mu, sigma, alpha):
    L = np.zeros((num_samples, num_clusters))
    for k in range(num_clusters):
        L[:, k] = alpha[k] * gaussian_pdf(data, mu[k], sigma[k])
    return np.sum(np.log(np.sum(L, axis=1)))


def EM(data, times, thresh=0.001, max_iter=100):
    update = 10
    nulls = data == 0
    sigmas, fit_loess, p = EM_initial_guess(data, times, nulls)
    n_iter = 0
    q = np.zeros(data.shape)

    while update > thresh and n_iter < max_iter:
        n_iter += 1
        old_q = np.copy(q)
        q = EM_E_step(data, p, sigmas, fit_loess)
        fit_loess, p = EM_M_step(data, q, times)
        update = np.sum(np.sum(np.abs(q - old_q)))

        print("Iteration {}, update {}".format(n_iter, update))

    return q, fit_loess


def show_data(data, loess, times, labels):
    real_classes = pd.read_csv('../cache/dropped_points.csv', index_col=0)
    real_functions = pd.read_csv('../cache/generative_functions.csv', index_col=0)
    for ix, row in data.iterrows():
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(times, loess[ix])
        plt.scatter(times, row, marker='.', c=labels[ix], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
        plt.xlabel('Time')
        plt.ylabel('Expression')
        plt.title('Predictions for function %s' % ix)
        plt.subplot(122)
        f = lambda x: np.polyval(real_functions.loc[ix, :], x)
        real_values = [max(f(t), 0) for t in times]
        plt.plot(times, real_values)
        plt.scatter(times, row, marker='.', c=real_classes.loc[ix, :],
                    cmap=matplotlib.colors.ListedColormap(['red', 'green']))
        plt.title('Real classes and function %s' % ix)
        plt.xlabel('Time')
        plt.ylabel('Expression')
    plt.show()


if __name__ == '__main__':
    n_genes = 5
    print('Loading data...')
    data = pd.read_csv('../cache/synthetic_data.csv', index_col=0, nrows=n_genes)
    print('Loading times...')
    times = np.array(pd.read_csv('../cache/times.csv', index_col=0))
    times = times.reshape((500,))
    print('Starting EM...')
    q, fit_loess = EM(data, times, thresh=0.1 * n_genes, max_iter=10)

    labels = np.zeros(q.shape)
    labels[q > 0.05] = 1

    for ix, row in enumerate(q):
        print(row[range(5)])
        print()

    # Fit loess without considering the dropped data
    # pred_values = pd.DataFrame()
    # for ix, row in data.iterrows():
    #     y = row[labels[ix]==1]
    #     x = times[labels[ix]==1]
    #     model = loess(x=x, y=y)
    #     model.fit()
    #     pred = model.predict(times).values
    #     pred_values = pred_values.append(pd.Series(pred), ignore_index=True)

    show_data(data, fit_loess, times, labels)
