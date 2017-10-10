import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_samples(t, Zs, wrap, size=(1.5, 1.5)):
    """ Plot a number of processes in a grid.

    Args:
        t: index array of size n
        Zs: mxn array of m process samples
        wrap: start a new line after this number of plots
        size: (width, height) of each plot
    """
    n_samples = Zs.shape[0]
    df = pd.DataFrame(dict(
        sample=np.repeat(range(n_samples), len(t)),
        t=np.tile(t, n_samples),
        Z=Zs.ravel()))
    w, h = size
    grid = sns.FacetGrid(df, col='sample', hue='sample', col_wrap=wrap,
                         size=h, aspect=w/h)
    grid.map(plt.plot, 't', 'Z', marker="o", ms=4)
    plt.show()


def plot_gp(mu, k, t, dimension=(2, 4), size=(1.5, 1.5)):
    """Plot samples of a Gaussian process.

    Args:
        mu: mean function.
        k: covariance function.
        t: array of inputs.
        dimension: tuple (h, w); make a grid of h x w samples/plots.
        size: (width, height) of each plot
    Returns
        Mean and covariance matrix of the GP-distribution.
    """
    mu_ = np.array([mu(i) for i in t])
    C = np.array([[k(i, j) for j in t] for i in t])
    h, w = dimension
    samples = np.random.multivariate_normal(mu_, C, h * w)
    plot_samples(t, samples, w, size)
    return mu_, C
