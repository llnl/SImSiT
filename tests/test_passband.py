import numpy as np
import pytest
import astropy.units as u
from scipy.integrate import quad

from simsit.passband import (
    Passband,
    ab_magnitude_to_photon_rate,
    photon_rate,
    spectrum_to_magnitude,
)


@pytest.fixture
def top_hat():
    return Passband("test", [400, 500], [1, 1], "synthetic test curve")


def test_passband_is_validated_serialized_and_immutable():
    band = Passband.from_config(
        {"name": "x", "wavelength_nm": [400, 500], "throughput": [0, 0.5], "reference": "supplied"}
    )
    assert band.to_config() == {
        "name": "x",
        "wavelength_nm": [400.0, 500.0],
        "throughput": [0.0, 0.5],
        "reference": "supplied",
    }
    assert band.provenance == "supplied"
    with pytest.raises(ValueError):
        band.wavelength_nm[0] = 401
    with pytest.raises(ValueError):
        band.wavelength_nm.setflags(write=True)
    with pytest.raises(ValueError):
        band.throughput.setflags(write=True)
    with pytest.raises(ValueError):
        Passband("bad", [400, 399], [1, 1], "source")
    with pytest.raises(ValueError):
        Passband("bad", [400, 500], [0, 0], "source")


def test_flat_fnu_top_hat_has_closed_form_photon_rate(top_hat):
    magnitude = 0.0
    actual = ab_magnitude_to_photon_rate(magnitude, top_hat, 1.0).to_value(1 / u.s)
    fnu = 10 ** (-0.4 * 48.60) * 1e-3  # AB definition, erg/s/cm2/Hz -> W/m2/Hz.
    expected = fnu / 6.62607015e-34 * np.log(500 / 400)
    assert actual == pytest.approx(expected, rel=2e-13)


def test_ab_photon_rate_supports_magnitude_arrays(top_hat):
    rates = ab_magnitude_to_photon_rate(np.array([0.0, 1.0]), top_hat, 2.0)
    assert rates.shape == (2,)
    assert rates.unit.is_equivalent(1 / u.s)
    assert rates[0].value / rates[1].value == pytest.approx(10**0.4, rel=1e-13)


def test_narrow_linear_segments_match_quad_oracle():
    left = 1.0
    width = 1e-12
    band = Passband(
        "narrow",
        [left, left + width, left + 2 * width],
        [0.0, 0.25, 1.0],
        "synthetic narrow test curve",
    )
    actual = sum(
        quad(
            lambda wavelength, lo, hi, t_lo, t_hi: (
                t_lo
                + (t_hi - t_lo) * (wavelength - lo) / (hi - lo)
            )
            / wavelength,
            lo,
            hi,
            args=(lo, hi, t_lo, t_hi),
            epsabs=1e-30,
            epsrel=1e-12,
        )[0]
        for lo, hi, t_lo, t_hi in zip(
            band.wavelength_nm[:-1],
            band.wavelength_nm[1:],
            band.throughput[:-1],
            band.throughput[1:],
        )
    )
    from simsit.passband import _integral_linear_response_over_wavelength

    assert _integral_linear_response_over_wavelength(band) == pytest.approx(
        actual, rel=1e-12
    )


def test_sampled_photon_rate_matches_quad_oracle(top_hat):
    wavelength = np.linspace(400, 500, 11) * u.nm
    flux = (2e-12 * (wavelength.to_value(u.nm) / 450) ** 1) * u.W / u.m**2 / u.nm
    actual = photon_rate(wavelength, flux, top_hat, 3.0).to_value(1 / u.s)
    h = 6.62607015e-34
    c = 299792458.0
    oracle = 3 * quad(
        lambda x: 2e-12 * (x / (450e-9)) * 1e9 * x / h / c,
        400e-9,
        500e-9,
    )[0]
    assert actual == pytest.approx(oracle, rel=2e-11)


def test_band_mean_magnitude_recovers_nonflat_spectrum_photon_rate():
    band = Passband(
        "sloped",
        [400, 450, 500],
        [0.2, 0.8, 0.4],
        "synthetic test curve",
    )
    wavelength = np.array([400, 425, 475, 500]) * u.nm
    flux = (
        np.array([1.0, 2.0, 1.5, 3.0]) * 1e-12 * u.W / u.m**2 / u.nm
    )
    area = 2.5

    magnitude = spectrum_to_magnitude(wavelength, flux, band)
    direct = photon_rate(wavelength, flux, band, area)
    recovered = ab_magnitude_to_photon_rate(magnitude, band, area)

    wavelength_m = wavelength.to_value(u.m)
    flux_si = flux.to_value(u.W / u.m**2 / u.m)
    response_wavelength = band.wavelength_nm * 1e-9
    knots = np.unique(np.concatenate((wavelength_m, response_wavelength)))
    oracle = area * sum(
        quad(
            lambda x: (
                np.interp(x, wavelength_m, flux_si)
                * np.interp(x, response_wavelength, band.throughput)
                * x
                / 6.62607015e-34
                / 299792458.0
            ),
            left,
            right,
            epsabs=0.0,
            epsrel=2e-12,
        )[0]
        for left, right in zip(knots[:-1], knots[1:])
    )

    assert np.isfinite(magnitude)
    assert direct.to_value(1 / u.s) == pytest.approx(oracle, rel=2e-12)
    assert recovered.to_value(1 / u.s) == pytest.approx(oracle, rel=2e-12)


def test_band_magnitude_matches_flat_fnu_and_unit_invariance(top_hat):
    wavelength_nm = np.linspace(400, 500, 11)
    fnu = (7.5 * u.ABmag).to(u.Jy)
    flux_nm = (fnu * const_c / (wavelength_nm * u.nm) ** 2).to(
        u.W / u.m**2 / u.nm
    )
    mag_nm = spectrum_to_magnitude(wavelength_nm * u.nm, flux_nm, top_hat)
    mag_um = spectrum_to_magnitude(
        wavelength_nm / 1000 * u.um, flux_nm.to(u.W / u.m**2 / u.um), top_hat
    )
    assert mag_nm == pytest.approx(7.5, abs=4e-4)
    assert mag_um == pytest.approx(mag_nm, abs=2e-11)


def test_zero_flux_is_infinite_but_negative_and_uncovered_are_rejected(top_hat):
    wavelength = np.array([400, 450, 500]) * u.nm
    zero = np.zeros(3) * u.W / u.m**2 / u.nm
    assert np.isinf(spectrum_to_magnitude(wavelength, zero, top_hat))
    with pytest.raises(ValueError):
        spectrum_to_magnitude(wavelength, -zero + [-1, 0, 0] * u.W / u.m**2 / u.nm, top_hat)
    with pytest.raises(ValueError):
        spectrum_to_magnitude(
            np.array([450, 500]) * u.nm,
            np.ones(2) * u.W / u.m**2 / u.nm,
            top_hat,
        )


# Kept local to the test so the expected unit conversion is explicit.
const_c = 299792458 * u.m / u.s
