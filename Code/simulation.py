import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_genes = 1000
degree = 12
n_times = 500
var = 2
noise = 'gaussian'


def get_function(degree=3):
    coeff = np.random.uniform(-10, 10, size=degree)
    return lambda x: np.polyval(coeff, x), coeff


def get_functions(n, d):
    functions = []
    coefficients = pd.DataFrame()
    while len(functions) < n:
        f, coeff = get_function(d)
        if np.max(f(np.arange(0, 1, 0.1))) > 0:
            functions.append(f)
            coefficients = coefficients.append(pd.Series(coeff), ignore_index=True)
    return functions, coefficients


def get_times(n_times=500):
    return np.sort(np.random.uniform(0, 1, size=n_times))


def add_noise(values, var=2, type='gaussian'):
    if type == 'gaussian':
        noise = np.random.normal(0, var, size=values.shape)
        noisy_values = values + noise
        noisy_values[noisy_values < 0] = 0
    return noisy_values


def evaluate_functions(functions, times):
    values = pd.DataFrame()
    for f in functions:
        v = f(times)
        values = values.append(pd.Series(v), ignore_index=True)
    return values


def drop_points(values, range=(0.2, 0.4)):
    classes = pd.DataFrame()
    for ix, row in values.iterrows():
        n_dropped = np.random.randint(low=len(row) * range[0], high=len(row) * range[1])
        idx = np.random.randint(low=0, high=len(row), size=n_dropped)
        values[idx] = 0
        c = pd.Series(np.zeros(len(row)))
        c[idx] = 1
        classes = classes.append(pd.Series(c), ignore_index=True)
    return values, classes


def plot_functions(functions, values, times):
    plt.figure()
    for ix, row in values.iterrows():
        real_values = [max(functions[ix](t), 0) for t in times]
        plt.plot(times, real_values, linewidth=2.0)
        plt.scatter(times, row, marker='.')
    plt.xlabel('Time')
    plt.ylabel('Expression')
    plt.show()


def get_dataset(n_genes, n_times, n_dropped, var=2, noise='gaussian', save=False):
    functions, coefficients = get_functions(n=n_genes, d=degree)
    times = get_times(n_times)
    values = evaluate_functions(functions, times)
    values = add_noise(values, var, noise)
    values, classes = drop_points(values, n_dropped)
    if save:
        coefficients.to_csv(('../cache/generative_functions.csv'))
        values.to_csv('../cache/synthetic_data.csv')
        classes.to_csv(('../cache/dropped_points.csv'))
    return values, classes


if __name__ == '__main__':
    # functions, coefficients = get_functions(n=n_genes,d=degree)
    # times = get_times(n_times)
    # values = evaluate_functions(functions,  times)
    # values = add_noise(values, var, noise)
    # values, classes = drop_points(values)
    # plot_functions(functions, values, times)

    get_dataset(n_genes, n_times, n_dropped=(0.2, 0.4), save=True)
