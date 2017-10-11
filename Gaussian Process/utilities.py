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
    sns.set_style('ticks')
    grid = sns.FacetGrid(df, col='sample', hue='sample', col_wrap=wrap,
                         size=h, aspect=w/h)
    grid.map(plt.plot, 't', 'Z', marker="o", ms=4)
    plt.show()


def mean(mu, t):
    """Evaluate mean function for index-set t.

    Args:
        mu: mean function
        t: index set

    Returns: mean output for t.
    """
    return np.array([mu(i) for i in t])


def covariance(k, a, b):
    """Evaluate covariance function for index-sets a and b.

    Args:
        k: covariance function
        a, b: two index-sets (arrays)

    Returns: a x b covariance matrix.
    """
    return np.array([[k(i, j) for j in b] for i in a])


def posterior(mu, k, ta, tb, Zb):
    """Posterior GP given some observations.

    Args:
        mu: mean function.
        k: covariance function.
        ta: posterior process indices.
        tb: observation indices.
        Zb: observation outputs.

    Returns:
        m: mean of posterior over ta.
        D: covariance of posterior over ta x ta.
    """
    mua = mean(mu, ta)
    Kaa = covariance(k, ta, ta)

    mub = mean(mu, tb)
    Kbb = covariance(k, tb, tb)

    Kab = covariance(k, ta, tb)

    Kbb_inv = np.linalg.inv(Kbb)
    m = mua + Kab.dot(Kbb_inv).dot(Zb - mub)
    D = Kaa - Kab.dot(Kbb_inv).dot(Kab.T)

    return m, D


def noisy_posterior(mu, k, sigma, ta, tb, Zb):
    """Noisy posterior GP given some observations.

    Args:
        mu: mean function.
        k: covariance function.
        sigma: output observation variance.
        ta: posterior process indices.
        tb: observation indices.
        Zb: observation outputs.

    Returns:
        m: mean of posterior over ta.
        D: covariance of posterior over ta x ta.
    """
    mua = mean(mu, ta)
    Kaa = covariance(k, ta, ta)

    mub = mean(mu, tb)
    Kbb = covariance(k, tb, tb)

    Kab = covariance(k, ta, tb)

    Kbb_inv = np.linalg.inv(Kbb + sigma**2 * np.eye(len(tb)))
    m = mua + Kab.dot(Kbb_inv).dot(Zb - mub)
    D = (Kaa + sigma**2 * np.eye(len(ta))) - Kab.dot(Kbb_inv).dot(Kab.T)

    return m, D


def plot_gp(mu, k, t, dimension=(2, 6), size=(1.5, 1.5)):
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
    mu_ = mean(mu, t)
    C = covariance(k, t, t)
    h, w = dimension
    samples = np.random.multivariate_normal(mu_, C, h * w)
    plot_samples(t, samples, w, size)
    return mu_, C
