import numpy as np
import pytest
from scipy.stats import chi2, cramervonmises

from simsit.uncertainty_metrics import (
    chi_squared_statistic,
    cvm_chi_squared_test,
    cvm_statistic_to_pvalue,
    pearson_bin_residuals,
)


def test_chi_squared_statistic_matches_quadratic_form():
    residuals = np.array([[1.0, 2.0], [-1.0, 3.0]])
    covariance = np.array([[[2.0, 0.5], [0.5, 1.0]]] * 2)

    result = chi_squared_statistic(residuals, covariance)
    expected = np.array(
        [r @ np.linalg.inv(c) @ r for r, c in zip(residuals, covariance)]
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-np.inf], [-1.0]])
def test_cvm_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        cvm_chi_squared_test(bad, 2)


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-np.inf], [-1.0]])
def test_pearson_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        pearson_bin_residuals(bad, 2, 4)


def test_pearson_returns_requested_number_of_bins():
    values = chi2.ppf(np.arange(1, 4) / 4, 2)

    result = pearson_bin_residuals(values, 2, 4)

    assert result.shape == (4,)
    assert np.isclose(np.sum(result), 0.0)


def test_cvm_statistic_matches_known_cdf_reference():
    values = np.array([0.2, 0.8, 1.4, 2.0, 3.5, 5.0])
    ours = cvm_chi_squared_test(values, 2)
    reference = cramervonmises(values, chi2(df=2).cdf).statistic

    np.testing.assert_allclose(ours, reference)


def test_cvm_pvalue_is_finite_and_monotonic():
    statistics = np.array([0.025, 0.1, 0.5, 1.2])
    pvalues = np.array(
        [cvm_statistic_to_pvalue(statistic, 20) for statistic in statistics]
    )

    assert np.all(np.isfinite(pvalues))
    assert np.all((pvalues >= 0.0) & (pvalues <= 1.0))
    assert np.all(np.diff(pvalues) < 0.0)
