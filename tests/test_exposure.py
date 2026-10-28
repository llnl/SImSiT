import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time
from scipy.integrate import quad

from simsit.exposure import sample_exposure


EPOCH = Time("2025-01-01", scale="utc")


def run(position, rate, **kwargs):
    return sample_exposure(position, rate, EPOCH, 8.0, **kwargs)


def test_constant_straight_motion_and_rate_are_closed_form():
    result = run(lambda t: [2 + 0.5 * (t.gps - EPOCH.gps), -1], lambda t: 3.0)
    np.testing.assert_allclose(result.offsets_s, np.arange(9.0))
    np.testing.assert_allclose(result.positions_px[[0, -1]], [[2, -1], [6, -1]])
    np.testing.assert_allclose(result.segment_photons, 3.0)
    assert result.total_photons == pytest.approx(24.0)
    assert result.estimated_flux_error == 0


def test_linear_flux_has_exact_total_and_converges_centroid_gradient_shift():
    result = run(
        lambda t: [t.gps - EPOCH.gps, 0],
        lambda t: 2 + (t.gps - EPOCH.gps),
        pixel_tolerance=1e-4,
        flux_relative_tolerance=1e-10,
    )
    # Integral of 2 + t on [0, 8].
    assert result.total_photons == pytest.approx(48.0)
    assert np.max(np.diff(result.offsets_s)) < 1.0
    midpoint = (result.positions_px[:-1, 0] + result.positions_px[1:, 0]) / 2
    centroid = np.average(midpoint, weights=result.segment_photons)
    expected = quad(lambda x: x * (2 + x), 0, 8)[0] / 48
    assert centroid == pytest.approx(expected, abs=2e-4)


def test_quadratic_curved_path_converges_against_analytic_integral():
    result = run(
        lambda t: [(t.gps - EPOCH.gps) ** 2, 0.1 * (t.gps - EPOCH.gps) ** 2],
        lambda t: 1.0,
        pixel_tolerance=1e-5,
        flux_relative_tolerance=1e-10,
        max_samples=20000,
    )
    assert result.total_photons == pytest.approx(8.0)
    np.testing.assert_allclose(result.positions_px[:, 0], result.offsets_s**2, atol=1e-12)
    assert np.max(np.diff(result.offsets_s)) < 0.2


def test_gaussian_flare_matches_scipy_quad():
    sigma = 0.31
    center = 3.7
    rate = lambda t: np.exp(-0.5 * ((t.gps - EPOCH.gps - center) / sigma) ** 2)
    result = run(
        lambda t: [0, 0],
        rate,
        max_step_s=0.2,
        pixel_tolerance=1.0,
        flux_relative_tolerance=2e-5,
    )
    expected = quad(lambda x: np.exp(-0.5 * ((x - center) / sigma) ** 2), 0, 8)[0]
    assert result.total_photons == pytest.approx(expected, rel=2e-5)


def test_zero_rate_is_finite_and_has_zero_weights():
    result = run(lambda t: [t.gps - EPOCH.gps, 4], lambda t: 0.0)
    assert result.total_photons == 0
    np.testing.assert_array_equal(result.segment_photons, 0)
    assert result.estimated_flux_error == 0


def test_explicit_step_resolves_aliasing_sensitive_sinusoid():
    rate = lambda t: 2 + np.sin(2 * np.pi * 3 * (t.gps - EPOCH.gps) / 8)
    result = run(
        lambda t: [0, 0],
        rate,
        max_step_s=0.05,
        pixel_tolerance=1.0,
        flux_relative_tolerance=1e-5,
    )
    assert result.total_photons == pytest.approx(16.0, abs=1e-7)


@pytest.mark.parametrize(
    "rate",
    [lambda t: -1.0, lambda t: np.nan, lambda t: np.inf],
)
def test_bad_rates_rejected(rate):
    with pytest.raises(ValueError, match="rate"):
        run(lambda t: [0, 0], rate)


def test_bad_position_and_t0_rejected():
    with pytest.raises(ValueError, match="position"):
        run(lambda t: [0, np.nan], lambda t: 1)
    with pytest.raises(ValueError, match="scalar"):
        sample_exposure(lambda t: [0, 0], lambda t: 1, [1e9, 1e9 + 1], 1)


def test_arrays_are_immutable_and_limits_fail_loudly():
    result = run(lambda t: [(t.gps - EPOCH.gps) ** 2, 0], lambda t: 1, max_step_s=8)
    with pytest.raises(ValueError):
        result.offsets_s[0] = 3
    with pytest.raises(ValueError):
        result.positions_px.setflags(write=True)
    with pytest.raises(ValueError):
        result.segment_photons.setflags(write=True)
    with pytest.raises(RuntimeError, match="max_depth"):
        run(
            lambda t: [(t.gps - EPOCH.gps) ** 2, 0],
            lambda t: 1,
            pixel_tolerance=1e-12,
            max_depth=0,
        )


def test_duration_units_and_seed_budget_are_explicit():
    result = sample_exposure(lambda t: [0, 0] * u.pix, lambda t: 10 / u.s, EPOCH, 1 * u.min)
    assert result.offsets_s[-1] == 60
    assert result.total_photons == pytest.approx(600)
    with pytest.raises(RuntimeError, match="seed"):
        sample_exposure(
            lambda t: [0, 0], lambda t: 10, EPOCH, 60,
            max_step_s=1e-300, max_samples=33,
        )
    with pytest.raises(ValueError, match="masked"):
        run(lambda t: np.ma.array([0, 0], mask=[True, False]), lambda t: 1)


def test_symmetric_flare_refines_streak_width_not_just_total_or_centroid():
    rate = lambda t: 1 + 100 * np.exp(-0.5 * ((t.gps - EPOCH.gps - 4) / 0.5)**2)
    result = run(
        lambda t: [t.gps - EPOCH.gps, 0], rate,
        pixel_tolerance=0.002, flux_relative_tolerance=1e-4, max_samples=20000,
    )
    a = result.positions_px[:-1, 0]
    b = result.positions_px[1:, 0]
    # Second moment of each uniform line segment is (a²+ab+b²)/3.
    rendered_second = np.average((a*a + a*b + b*b) / 3, weights=result.segment_photons)
    integral = quad(lambda x: 1 + 100 * np.exp(-0.5 * ((x - 4) / 0.5)**2), 0, 8)[0]
    reference = quad(lambda x: x*x * (1 + 100 * np.exp(-0.5 * ((x - 4) / 0.5)**2)), 0, 8)[0] / integral
    assert rendered_second == pytest.approx(reference, abs=0.003)
