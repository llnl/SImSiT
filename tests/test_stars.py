import galsim
import numpy as np
import pytest
import simsit
from astropy.table import MaskedColumn, Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time


def test_mock_star_catalog():
    catalog = simsit.MockStarCatalog()
    rng = np.random.default_rng(57721)

    for _ in range(10):
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        ph = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(ph)
        y = r * np.sin(ph)

        center = galsim.CelestialCoord.from_xyz(x, y, z)
        time = None

        stars1 = catalog.get_stars(center, center, 0.1 * galsim.degrees, time)

        center2 = center.greatCirclePoint(
            galsim.CelestialCoord.from_xyz(0, 0, 1),
            0.09 * galsim.degrees,
        )
        stars2 = catalog.get_stars(center2, center2, 0.3 * galsim.degrees, time)

        # Every star in the smaller query should appear in the larger query.
        stars2_coords = set(zip(np.asarray(stars2["ra_rad"]), np.asarray(stars2["dec_rad"])))
        stars1_coords = set(zip(np.asarray(stars1["ra_rad"]), np.asarray(stars1["dec_rad"])))

        assert stars1_coords.issubset(stars2_coords)


def test_mock_catalog_samples_variable_width_histogram():
    catalog = simsit.MockStarCatalog(
        level=1,
        seed=123,
        magnitude_hists={
            "G": {
                "bins": np.array([0.0, 1.0, 4.0]),
                "values": np.array([1, 1]),
            }
        },
    )
    samples = []
    for idx in [62, 58, 61]:
        samples.append(catalog._get_trixel_stars(idx, None)["g_mag"])
    samples = np.concatenate(samples)

    assert np.all((samples >= 0.0) & (samples <= 4.0 + 0.3542))
    assert np.any(samples > 1.0 + 0.3542)


def test_gaia_catalog_supports_lsst_and_standard_units(monkeypatch):
    captured = {}

    def apparent(ra, dec, time, **kwargs):
        captured["ra"] = np.asarray(ra)
        captured["dec"] = np.asarray(dec)
        captured.update(kwargs)
        return np.asarray(ra), np.asarray(dec)

    monkeypatch.setattr("simsit.catalog.catalog_to_apparent", apparent)

    lsst = Table(
        {
            "coord_ra": [1.0] * u.rad,
            "coord_dec": [0.5] * u.rad,
            "pm_ra": [2.0] * u.mas / u.yr,
            "pm_dec": [-3.0] * u.mas / u.yr,
            "parallax": [0.1] * u.arcsec,
            "phot_g_mean_mag": [15.0],
        }
    )
    standard = Table(
        {
            "ra": [10.0],
            "dec": [20.0],
            "pmra": [2.0],
            "pmdec": [-3.0],
            "parallax": [100.0],
            "phot_g_mean_mag": [15.0],
            "ref_epoch": [2016.0],
        }
    )

    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: lsst)
    lsst_result = simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(
        1, Time("J2000")
    )
    assert np.isclose(lsst_result["i_mag"][0], 15.105)
    assert np.isclose(captured["parallax"][0], 0.1)

    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: standard)
    standard_result = simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(
        1, Time("J2000")
    )
    assert np.isclose(standard_result["ra"][0], 10.0)
    assert np.isclose(captured["parallax"][0], 0.1)
    expected_j2000 = SkyCoord(
        ra=[10.0] * u.deg,
        dec=[20.0] * u.deg,
        pm_ra_cosdec=[2.0] * u.mas / u.yr,
        pm_dec=[-3.0] * u.mas / u.yr,
        obstime=Time([2016.0], format="jyear"),
    ).apply_space_motion(new_obstime=Time("J2000"))
    np.testing.assert_allclose(
        captured["ra"][0], expected_j2000.ra.rad[0], rtol=0.0, atol=1e-15
    )
    assert abs(captured["ra"][0] - np.deg2rad(10.0)) > 1e-8
    np.testing.assert_allclose(
        captured["pmra"],
        expected_j2000.pm_ra_cosdec.to_value(u.mas / u.yr),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        captured["pmdec"],
        expected_j2000.pm_dec.to_value(u.mas / u.yr),
        rtol=0.0,
        atol=1e-12,
    )


def test_standard_gaia_catalog_requires_reference_epoch(monkeypatch):
    standard = Table(
        {
            "ra": [10.0],
            "dec": [20.0],
            "pmra": [2.0],
            "pmdec": [-3.0],
            "parallax": [100.0],
            "phot_g_mean_mag": [15.0],
        }
    )
    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: standard)

    with pytest.raises(ValueError, match="ref_epoch"):
        simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(1, Time("J2000"))


def test_gaia_catalog_accepts_physical_flux_density(monkeypatch):
    flux_only = Table(
        {
            "coord_ra": [0.0] * u.rad,
            "coord_dec": [0.0] * u.rad,
            "pm_ra": [0.0] * u.mas / u.yr,
            "pm_dec": [0.0] * u.mas / u.yr,
            "parallax": [0.0] * u.arcsec,
            "phot_g_mean_flux": [1.0] * u.nJy,
        }
    )
    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: flux_only)

    result = simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(
        1, Time("J2000")
    )

    assert np.isclose(result["i_mag"][0], 31.4)


def test_gaia_catalog_preserves_flux_density_mask(monkeypatch):
    from astropy.table import MaskedColumn

    data = Table({
        "coord_ra": [0.0, 1.0] * u.rad,
        "coord_dec": [0.0, 0.0] * u.rad,
        "pm_ra": [0.0, 0.0] * u.mas / u.yr,
        "pm_dec": [0.0, 0.0] * u.mas / u.yr,
        "parallax": [0.0, 0.0] * u.arcsec,
    })
    data["phot_g_mean_flux"] = MaskedColumn([1.0, 2.0], mask=[False, True], unit=u.nJy)
    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: data)
    result = simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(1, Time("J2000"))
    assert len(result) == 1
    np.testing.assert_allclose(result["i_mag"], [31.4], atol=1e-10, rtol=0)


def test_gaia_catalog_rejects_electron_rate_flux(monkeypatch):
    flux_only = Table(
        {
            "coord_ra": [0.0] * u.rad,
            "coord_dec": [0.0] * u.rad,
            "pm_ra": [0.0] * u.mas / u.yr,
            "pm_dec": [0.0] * u.mas / u.yr,
            "parallax": [0.0] * u.arcsec,
            "phot_g_mean_flux": [1000.0] * u.electron / u.s,
        }
    )
    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: flux_only)

    with pytest.raises(ValueError, match="physical flux density"):
        simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(1, Time("J2000"))


def test_gaia_catalog_masks_invalid_rows_and_preserves_empty_shape(monkeypatch):
    data = Table(
        {
            "coord_ra": MaskedColumn(
                [1.0, 2.0], mask=[False, True], unit=u.rad
            ),
            "coord_dec": [0.5, 0.6] * u.rad,
            "pm_ra": [2.0, 3.0] * u.mas / u.yr,
            "pm_dec": [-3.0, -4.0] * u.mas / u.yr,
            "parallax": [0.1, 0.2] * u.arcsec,
            "phot_g_mean_mag": [15.0, 16.0],
        }
    )
    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: data)
    result = simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(
        1, Time("J2000")
    )

    assert len(result) == 1

    empty = Table(
        {
            "coord_ra": [np.nan] * u.rad,
            "coord_dec": [np.nan] * u.rad,
            "pm_ra": [np.nan] * u.mas / u.yr,
            "pm_dec": [np.nan] * u.mas / u.yr,
            "parallax": [np.nan] * u.arcsec,
            "phot_g_mean_mag": [np.nan],
        }
    )
    monkeypatch.setattr("simsit.catalog.Table.read", lambda _: empty)
    result = simsit.GaiaStarCatalog("/tmp")._get_trixel_stars(
        1, Time("J2000")
    )

    assert len(result) == 0
    assert result.colnames == ["ra_rad", "dec_rad", "i_mag", "ra", "dec"]
