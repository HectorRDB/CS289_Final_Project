Here will go all the various scripts


-------------------------------------------

simulation.py

Script generating a random function to test our methods.

Functions:

get_dataset(n_genes, n_times, p_dropped, var=2, noise='gaussian', save=False)
Inputs:
-n_genes: int
The number of function generated

-n_times: int
The number of timestamps

-p_dropped: tuple of floats in [0,1], default=(0.15,0.3)
The range for the proportion of dropped points. For each function, the actually number of dropped points will be selected uniformly in this range.

-var: float, default=2
Variance of the noise added to the data

-noise: str, default='gaussian'
Type of noise added to the data. For now, only gaussian noise is supported.

-save: boolean, default=False
Save or visualise the dataset.
If true the following files are saved in the cache: 'synthetic_data.csv', 'times.csv', 'dropped_points.csv', 'generative_functions.csv'

Outputs:
-values: pandas.DataFrame
A DataFrame of size n_genes x n_times containing the noisy data

-times: numpy.array
A vector of size n_times containing the sorted timestamps corresponding to values

-classes: pandas.DataFrame
A DataFrame of size n_genes x n_times containing the class of the data

--------

plot_functions(functions, values, times, classes)
Plot on graph per function with on the same graph the real function, the noisy values and colors the 2 classes.

Inputs:
-functions: list
A list of functions to graph. Ideally they should be constructed from the coefficients in generative_functions.csv, using numpy.polyval to generate the function from the coefficients.

- values: pandas.DataFrame
A DataFrame containing the noisy values corresponding the the functions (each row of the DataFrame contains the values of a function).

- times: pandas.DataFrame
A DataFrame containing the timestamps of the points

- classes. pandas.DataFrame
A DataFrame containing the classes of the points. Used to color the graph. Red points are dropped/false zeros and green points are real values.

Output:
None


-------------------------------------------

