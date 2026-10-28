import numpy as np
import requests
import pytest
from astropy.table import Table

from simsit.photometry import (
    convert_gaia_magnitude,
    get_gaia_magnitude_histogram,
    get_gaia_sources,
    vega_to_ab_offset,
)


def test_convert_gaia_magnitude():
    gaia_g = 15.0
    gaia_bp = 16.0
    gaia_rp = 14.0
    color = gaia_bp - gaia_rp

    result = convert_gaia_magnitude(gaia_g, gaia_bp, gaia_rp, "2MASS_Ks")
    expected = gaia_g + 0.1885 + (-2.092 * color) + (0.1345 * color**2)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    result = convert_gaia_magnitude(gaia_g, gaia_bp, gaia_rp, "SDSS12_i")
    expected = gaia_g + 0.29676 + (-0.64728 * color) + (0.10141 * color**2)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_vega_to_ab_offset():
    offsets = {
        "2MASS_Ks": 1.85,
        "2MASS_H": 1.39,
        "2MASS_J": 0.91,
        "G": 0.105,
        "BP": 0.0292,
        "RP": 0.3542,
    }

    for filter_name, expected_offset in offsets.items():
        result = vega_to_ab_offset(filter_name)
        assert np.isclose(result, expected_offset), (
            f"Expected {expected_offset} for {filter_name}, got {result}"
        )


def test_sdss_i_relation_and_published_color_domain():
    # Evans et al. (2018) Table A.2 gives G-i, not i-G.
    color = 2.0
    expected = 15.0 - (-0.29676 + 0.64728 * color - 0.10141 * color**2)
    np.testing.assert_allclose(
        convert_gaia_magnitude(15, 16, 14, "SDSS12_i"), expected, atol=1e-12, rtol=0,
    )
    for color in (0.0, 4.5, np.nan):
        with pytest.raises(ValueError, match="BP-RP"):
            convert_gaia_magnitude(15, 14 + color, 14, "SDSS12_i")


def test_gaia_magnitude_histogram_has_edges_and_finite_filter(monkeypatch):
    catalog = Table(
        {
            "phot_g_mean_mag": [1.0, 2.0, 3.0, np.nan, np.inf],
            "phot_bp_mean_mag": [1.0, 2.0, 3.0, np.nan, np.inf],
            "phot_rp_mean_mag": [1.0, 2.0, 3.0, np.nan, np.inf],
        }
    )
    monkeypatch.setattr(
        "simsit.photometry.get_gaia_sources",
        lambda *args, **kwargs: catalog,
    )

    result = get_gaia_magnitude_histogram(0, 0, 1, nbins=2)

    for values in result.values():
        assert len(values["bins"]) == 3
        assert len(values["values"]) == 2
        assert np.all(np.isfinite(values["bins"]))
        assert np.sum(values["values"]) == 3


def test_gaia_magnitude_histogram_constant_values():
    catalog = Table(
        {
            "phot_g_mean_mag": [5.0, 5.0],
            "phot_bp_mean_mag": [5.0, 5.0],
            "phot_rp_mean_mag": [5.0, 5.0],
        }
    )

    import simsit.photometry as photometry
    original = photometry.get_gaia_sources
    photometry.get_gaia_sources = lambda *args, **kwargs: catalog
    try:
        result = get_gaia_magnitude_histogram(0, 0, 1, nbins=1)
    finally:
        photometry.get_gaia_sources = original

    assert len(result["G"]["bins"]) == 2
    assert result["G"]["values"].tolist() == [2]


@pytest.mark.integration
def test_get_gaia_sources():
    ra_deg = 10.0
    dec_deg = 10.0
    radius_deg = 1.0
    star_lim = 10

    results = get_gaia_sources(
        ra_deg,
        dec_deg,
        radius_deg,
        gaia_data_release=3,
        star_lim=star_lim,
    )
    assert isinstance(results, Table), "Expected results to be an Astropy Table"
    assert len(results) <= star_lim, f"Expected at most {star_lim} results, got {len(results)}"


@pytest.mark.integration
def test_get_gaia_magnitude_histogram():
    nbins = 50
    try:
        results = get_gaia_magnitude_histogram(
            ra_deg=10.0,
            dec_deg=10.0,
            radius_deg=1.0,
            gaia_data_release=3,
            star_lim=100,
            nbins=nbins,
        )
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Skipping Gaia integration test due to remote service failure: {exc}")

    assert isinstance(results, dict), "Expected results to be a dictionary"
    for filt in ["G", "BP", "RP"]:
        assert filt in results, f"Expected filter {filt} in results"
        assert "bins" in results[filt], f"Expected 'bins' key in {filt} results"
        assert "values" in results[filt], f"Expected 'values' key in {filt} results"
        assert len(results[filt]["bins"]) in (0, nbins + 1), (
            f"Expected {nbins + 1} edges or empty result for {filt}, "
            f"got {len(results[filt]['bins'])}"
        )
