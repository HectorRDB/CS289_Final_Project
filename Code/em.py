import math

import numpy as np
import pandas as pd
from skmisc.loess import loess
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.stats import norm

import Test_Clustering as test

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
    p = [prob for _ in range(n_genes)]

    return sigmas, fit_loess, p


def EM_E_step(data, p, sigmas, fit_loess):
    # Implementation of E-step, as indicated in the report.
    q = np.zeros(data.shape)

    # For every point
    for ix, row in data.iterrows():
        s = sigmas[ix]
        if s == 0:
            pass

        else:

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

        if np.var(row) == 0:
            pass

        else:

            # Update the function f_j for every gene by fitting a weighted loess.
            model = loess(x=times, y=row, weights=q[ix])
            model.fit()
            fit_loess[ix] = model.predict(times).values

            # Update the probabilities p_j
            p[ix] = np.mean(q[ix])

    # We know that our function cannot be negative.
    fit_loess[fit_loess < 0] = 0

    return fit_loess, p


def EM(data, times, thresh=0.001, max_iter=100):
    update = 100000
    nulls = data == 0
    loess_functions = []

    sigmas, fit_loess, p = EM_initial_guess(data, times, nulls)
    loess_functions.append(fit_loess)
    n_iter = 0
    q = np.zeros(data.shape)
    while update > thresh and n_iter < max_iter:
        n_iter += 1
        print('Iteration %s' % n_iter)
        old_q = np.copy(q)
        q = EM_E_step(data, p, sigmas, fit_loess)
        fit_loess, p = EM_M_step(data, q, times)
        loess_functions.append(fit_loess)
        update = np.sum(np.sum(np.abs(q - old_q)))

        print("Iteration {}, update {}".format(n_iter, update))

    return q, loess_functions


def show_data(data, loess, times, labels):
    real_classes = pd.read_csv('../cache/dropped_points.csv', index_col=0)
    real_functions = pd.read_csv('../cache/generative_functions.csv', index_col=0)
    loess = loess[-1]

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


def evaluate_classification(data, labels, real_labels):
    em_labels, real_labels = test.preprocess_em(data, labels, real_labels)
    tpr, tnr, acc, rand_acc = test.test(np.array([em_labels]), real_labels)


    print('Accuracy: %s' % acc)
    print('Random accuracy: %s' % rand_acc)
    print('True Positive Rate: %s' % tpr)
    print('True Negative Rate: %s' % tnr)


def show_loess(data, loess, real_functions, labels):
    for ix, row in data.iterrows():
        plt.figure()
        plt.scatter(times, row, marker='.',
                    c=labels.loc[ix, :], cmap=matplotlib.colors.ListedColormap(['red', 'green']))

        # Plot every iteration of loess:
        for i, l in enumerate(loess):
            if i < 5:
                plt.plot(times, l[ix], label='iteration %s' % i, linewidth=2)
                plt.xlabel('Time')
                plt.ylabel('Expression')
                plt.title('Evolution of the loess fit for the function %s' % ix)

        f = lambda x: np.polyval(real_functions.loc[ix, :], x)
        real_values = [max(f(t), 0) for t in times]
        plt.plot(times, real_values, c='r', linewidth=4, label='generative function')
        plt.legend()
        plt.savefig('../figure/loess_fit_%s.png' % ix)

    plt.show()


def main(n_genes=None, evaluate=False, show=False, save=False):
    print('Loading data...')
    data = pd.read_csv('../cache/synthetic_data.csv', index_col=0, nrows=n_genes)
    times = np.array(pd.read_csv('../cache/times.csv', index_col=0, nrows=n_genes))
    real_labels = pd.read_csv('../cache/dropped_points.csv', index_col=0, nrows=n_genes)
    real_functions = pd.read_csv('../cache/generative_functions.csv', index_col=0, nrows=n_genes)
    times = times.reshape((500,))

    print('Starting EM...')
    q, functions = EM(data, times, thresh=0.1 * len(data), max_iter=100)

    labels = np.zeros(q.shape)
    labels[q > 0.5] = 1

    if save:
        pd.DataFrame(q).to_csv('../cache/q.csv')
        pd.DataFrame(labels).to_csv('../cache/em_labels.csv')
        pd.DataFrame(functions).to_csv(('../cache/loess_predictions.csv'))

    if evaluate:
        evaluate_classification(labels, real_labels)

    if show:
        show_data(data, functions, times, labels)
        show_loess(data, functions, real_functions, real_labels)


if __name__ == '__main__':
    main()
