"""
Goodness-of-fit metrics for testing whether normalized residuals are
consistent with a reference chi-squared distribution.

This module provides utilities to:

- compute chi-squared values from residual vectors and covariance matrices
- evaluate goodness of fit to a chi-squared reference distribution using
  a Cramér-von Mises statistic
- examine binned Pearson residuals using equiprobable chi-squared bins

References
----------
Horwood, J. T., Aristoff, J. M., Singh, N., Poore, A. B., & Hejduk, M. D.
"Beyond covariance realism: a new metric for uncertainty realism."
Proceedings of SPIE, Signal and Data Processing of Small Targets, Vol. 9092,
2014.
"""

import numpy as np
import scipy.stats


def chi_squared_statistic(residuals, covariance):
    """
    Compute chi-squared statistics from residual vectors and covariance matrices.

    For each residual vector ``dx`` and covariance matrix ``C``, this computes

        chi2 = dx^T C^-1 dx

    Parameters
    ----------
    residuals : array-like of shape (n, m) or (m,)
        Residual vectors of dimension ``m``. May be a single residual vector
        or an array of ``n`` residual vectors.
    covariance : array-like of shape (n, m, m) or (m, m)
        Covariance matrix or matrices corresponding to ``residuals``.

    Returns
    -------
    ndarray or float
        Array of chi-squared values, or a scalar if scalar inputs are provided.

    Raises
    ------
    ValueError
        If the input shapes are incompatible.
    numpy.linalg.LinAlgError
        If any covariance matrix is singular.
    """
    residuals = np.asarray(residuals, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if covariance.ndim == 2 and residuals.ndim == 1:
        scalar = True
        covariance = covariance[None, ...]
        residuals = residuals[None, ...]
    elif covariance.ndim == 3 and residuals.ndim == 2:
        scalar = False
    else:
        raise ValueError(
            "Expected residuals with shape (m,) and covariance with shape (m, m), "
            "or residuals with shape (n, m) and covariance with shape (n, m, m)."
        )

    if residuals.shape[0] != covariance.shape[0]:
        raise ValueError("Number of residual vectors and covariance matrices must match.")
    if residuals.shape[1] != covariance.shape[1] or covariance.shape[1] != covariance.shape[2]:
        raise ValueError("Incompatible residual and covariance dimensions.")

    covariance_inv = np.array([np.linalg.inv(c) for c in covariance])
    values = np.einsum("ij,ijk,ik->i", residuals, covariance_inv, residuals)

    return values[0] if scalar else values


def cvm_chi_squared_test(values, ndof, return_pvalue=False):
    """
    Compute the Cramér-von Mises statistic for fit to a chi-squared distribution.

    Parameters
    ----------
    values : array-like
        Sample of chi-squared values to test.
    ndof : int
        Degrees of freedom of the reference chi-squared distribution.
    return_pvalue : bool, optional
        If True, return an approximate p-value derived from the test statistic.

    Returns
    -------
    float
        Cramér-von Mises statistic, or an approximate p-value if
        ``return_pvalue=True``.

    Notes
    -----
    Larger Cramér-von Mises statistics indicate worse agreement with the
    reference distribution.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if len(values) == 0:
        raise ValueError("values must not be empty.")
    if ndof <= 0:
        raise ValueError("ndof must be positive.")

    values = np.sort(values)
    cdf = scipy.stats.chi2.cdf(values, ndof)

    n = len(values)
    ranks = np.arange(1, n + 1)
    statistic = 1.0 / (12 * n) + np.sum(((2 * ranks - 1) / (2 * n) - cdf) ** 2)

    if return_pvalue:
        return cvm_statistic_to_pvalue(statistic, n)
    return statistic


def cvm_statistic_to_pvalue(statistic, n):
    """
    Convert a Cramér-von Mises statistic to an approximate p-value.

    Parameters
    ----------
    statistic : float
        Cramér-von Mises test statistic.
    n : int
        Number of samples used to compute the statistic. Must be at least 5.

    Returns
    -------
    float
        Approximate p-value.

    Notes
    -----
    This uses tabulated values from Stephens (1986) with a finite-sample
    correction. The result is an approximation and should be interpreted with
    caution, especially in the tails.
    """
    if n < 5:
        raise ValueError("n must be >= 5")

    statistic = (statistic - 0.4 / n + 0.6 / n**2) / (1 + 1 / n)

    pvalues = np.array(
        [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
         0.25, 0.15, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
    )
    critical_values = np.array(
        [0.025, 0.046, 0.062, 0.078, 0.097, 0.118, 0.146, 0.184,
         0.209, 0.284, 0.347, 0.461, 0.581, 0.743, 0.869, 1.167]
    )

    return np.interp(statistic, critical_values, pvalues)


def pearson_bin_residuals(values, ndof, nbin):
    """
    Compute binned Pearson residuals for goodness-of-fit to a chi-squared distribution.

    The sample is binned using equiprobable bins under the reference
    chi-squared distribution. For each bin, this returns

        (observed - expected) / sqrt(expected)

    Parameters
    ----------
    values : array-like
        Sample of chi-squared values to test.
    ndof : int
        Degrees of freedom of the reference chi-squared distribution.
    nbin : int
        Number of equiprobable bins.

    Returns
    -------
    ndarray
        Pearson residuals for each bin.

    Notes
    -----
    These residuals are useful for diagnosing where the sample differs from
    the reference distribution, for example in the tails or near the center.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if len(values) == 0:
        raise ValueError("values must not be empty.")
    if ndof <= 0:
        raise ValueError("ndof must be positive.")
    if nbin <= 0:
        raise ValueError("nbin must be positive.")

    quantiles = np.linspace(0.0, 1.0, nbin + 1)
    cuts = np.concatenate(
        [[-np.inf], scipy.stats.chi2.ppf(quantiles[1:-1], ndof), [np.inf]]
    )
    bin_index = np.searchsorted(cuts, values) - 1

    counts = np.bincount(bin_index, minlength=nbin)
    expected = len(values) / nbin

    return (counts - expected) / np.sqrt(expected)


chi2 = chi_squared_statistic
cvm_chi2_test = cvm_chi_squared_test
pearsons_chi = pearson_bin_residuals