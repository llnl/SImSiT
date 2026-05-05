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
        assert len(results[filt]["bins"]) in (0, nbins), (
            f"Expected {nbins} bins or empty result for {filt}, got {len(results[filt]['bins'])}"
        )