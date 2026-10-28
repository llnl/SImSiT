import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from simsit import GaiaStarCatalog, Instrument, Passband
from simsit.photometry import spectrum_to_magnitude, synthetic_photometry


def response():
    return Passband("test_optical", [400, 500], [1, 1], "Analytic test top-hat")


def detector(**kwargs):
    config = {
        "image_shape": [32, 32], "gain": 2, "read_noise": 3, "pixel_scale": 1,
        "aperture": 1.2, "obscuration": 0.2,
        "distortion": {"th": [0, 1], "dthdr": [1, 1]}, "passband": response(),
    }
    config.update(kwargs)
    return Instrument(**config)


def spectra():
    return Table({
        "coord_ra": [0.0] * u.rad, "coord_dec": [0.0] * u.rad,
        "pm_ra": [0.0] * u.mas / u.yr, "pm_dec": [0.0] * u.mas / u.yr,
        "parallax": [0.0] * u.arcsec, "phot_g_mean_mag": [15.0],
        "wavelength": [[400.0, 450.0, 500.0]] * u.nm,
        "flux": [[2e-12, 2e-12, 2e-12]] * u.W / u.m**2 / u.nm,
    })


def test_spectrum_and_band_magnitude_match_closed_photon_integral():
    instrument = detector()
    wave = np.array([400, 450, 500]) * u.nm
    flux = np.full(3, 2e-12) * u.W / u.m**2 / u.nm
    mag = spectrum_to_magnitude(wave, flux, instrument.passband)
    expected = (
        instrument.collecting_area * 2e-12 * 1e9
        * (500e-9**2 - 400e-9**2) / 2 / (6.62607015e-34 * 299792458)
    )
    np.testing.assert_allclose(instrument.spectral_photon_rate(wave, flux).value, expected, rtol=1e-12)
    np.testing.assert_allclose(instrument.photon_rate(mag).value, expected, rtol=1e-12)
    np.testing.assert_allclose(instrument.photon_rate(instrument.photon_zeropoint).value, 1, rtol=1e-12)


def test_spectral_catalog_roundtrip_and_response_mismatch(tmp_path):
    raw = spectra()
    band = response()
    calibrated = synthetic_photometry(
        raw, band, wavelength_column="wavelength", flux_column="flux",
    )
    assert "mag" not in raw.colnames
    assert raw["phot_g_mean_mag"][0] == 15
    calibrated.write(tmp_path / "1.fits")
    result = GaiaStarCatalog(
        str(tmp_path), passband=band, magnitude_column="mag",
    )._get_trixel_stars(1, Time("J2000"))
    np.testing.assert_allclose(result["mag"], calibrated["mag"], atol=1e-12, rtol=0)
    different = Passband("test_optical", [400, 550], [1, 1], "Different test response")
    with pytest.raises(ValueError, match="match"):
        GaiaStarCatalog(
            str(tmp_path), passband=different, magnitude_column="mag",
        )._get_trixel_stars(1, Time("J2000"))


def test_raw_gaia_g_cannot_enter_calibrated_pipeline(tmp_path):
    raw = spectra()
    raw.write(tmp_path / "1.fits")
    with pytest.raises(ValueError, match="match"):
        GaiaStarCatalog(
            str(tmp_path), passband=response(), magnitude_column="phot_g_mean_mag",
        )._get_trixel_stars(1, Time("J2000"))


def test_empirical_calibration_requires_reference_and_physical_throughput():
    with pytest.raises(ValueError, match="passband"):
        detector(passband=None, zeropoint=20, zeropoint_reference="Synthetic test calibration")
    with pytest.raises(ValueError, match="reference"):
        detector(zeropoint=20)
    instrument = detector()
    with pytest.raises(ValueError, match="throughput"):
        detector(
            zeropoint=instrument.photon_zeropoint + 1,
            zeropoint_reference="Impossible analytic test calibration",
        )
    with pytest.raises(ValueError, match="throughput"):
        instrument.photon_scale(1)


def test_zero_flux_and_credentials_in_response_reference():
    assert detector().photon_rate(np.inf).value == 0
    for reference in (
        "https://example.invalid/data?token=fake-test-value",
        "https://example.invalid/data#password=fake-test-value",
        "https://example.invalid/data#secret=fake-test-value",
    ):
        with pytest.raises(ValueError, match="credentials"):
            Passband("test", [400, 500], [1, 1], reference)
