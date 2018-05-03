import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_function(d=3):
    # Generates a function of degree d.

    coeff = np.random.uniform(-10, 10, size=d)
    return lambda x: np.polyval(coeff, x), coeff


def get_functions(n, d):
    # Generates n functions of degree d with a least one positive value on [0,1]

    functions = []
    coefficients = pd.DataFrame()
    while len(functions) < n:
        f, coeff = get_function(d)
        if np.max(f(np.arange(0, 1, 0.1))) > 0:
            functions.append(f)
            coefficients = coefficients.append(pd.Series(coeff), ignore_index=True)
    return functions, coefficients


def get_times(n_times=500):
    # Generates n_times timestamps between [0,1] to evaluate the function

    return np.sort(np.random.uniform(0, 1, size=n_times))


def add_noise(values, var=2, type='gaussian'):
    # Adds a noise of type type to the values with a variance of var.
    # This function also operates the cut-of to 0 (we need only positive values for biological significance).

    if type == 'gaussian':
        noise = np.random.normal(0, var, size=values.shape)
        noisy_values = values + noise
        # idx = noisy_values < 0
        # noisy_values[idx] = 0

    else:
        raise ValueError('This type of noise (%s) is not supported yet.' % type)

    return noisy_values


def evaluate_functions(functions, times):
    # Evaluates the functions at each timestamp.
    # Returns a DataFrame containing the values of the functions.

    values = pd.DataFrame()
    for f in functions:
        v = f(times)
        values = values.append(pd.Series(v), ignore_index=True)
    return values


def drop_points(values, range=(0.2, 0.4)):
    # Drops a random number of points (in the specified range)
    # Returns both the values (after the drop) and the classes of each point (0: not dropped, 1: dropped).

    classes = pd.DataFrame()
    for ix, row in values.iterrows():
        # For every function we decide on a random number of points to drop.
        n_dropped = np.random.randint(low=len(row) * range[0], high=len(row) * range[1])
        # We set random indices to drop
        idx = np.random.randint(low=0, high=len(row), size=n_dropped)
        row[idx] = 0
        c = pd.Series(np.zeros(len(row)))
        c[idx] = 1
        classes = classes.append(pd.Series(c), ignore_index=True)
    return values, classes


def plot_functions(functions, values, times):
    # Plots the noisy values and the functions.

    plt.figure()
    for ix, row in values.iterrows():
        real_values = [max(functions[ix](t), 0) for t in times]
        plt.plot(times, real_values, linewidth=2.0)
        plt.scatter(times, row, marker='.')
    plt.xlabel('Time')
    plt.ylabel('Expression')
    plt.show()


def get_dataset(n_genes, n_times, n_dropped, var=2, noise='gaussian', save=False):
    # Generates the dataset. If save=True, saves the coefficients of the functions, the noisy values and the classes.

    functions, coefficients = get_functions(n=n_genes, d=degree)
    times = get_times(n_times)
    values = evaluate_functions(functions, times)
    noisy_values = add_noise(values, var, noise)
    noisy_values[noisy_values < 0] = 0
    dropped_values, classes = drop_points(noisy_values, n_dropped)
    if save:
        coefficients.to_csv(('../cache/generative_functions.csv'))
        noisy_values.to_csv('../cache/synthetic_data.csv')
        classes.to_csv(('../cache/dropped_points.csv'))
    else:
        plot_functions(functions, dropped_values, times)
    return values, classes


n_genes = 1000
degree = 12
n_times = 500
var = 2
noise = 'gaussian'
np.random.seed(2)

if __name__ == '__main__':
    # functions, coefficients = get_functions(n=5, d=degree)
    # times = get_times(n_times)
    # values = evaluate_functions(functions, times)
    # values = add_noise(values, var, noise)
    # values, classes = drop_points(values)
    # plot_functions(functions, values, times)

    get_dataset(n_genes, n_times, n_dropped=(0.2, 0.4), save=True)
