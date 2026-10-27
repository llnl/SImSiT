import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import galsim
import numpy as np
import pytest
import yaml
from astropy.time import Time

from simsit.tracker import InertialTracker, transform_wcs
from simsit.wcs import radial_wcs


DTEXT = """
distortion:
    th: [0.        , 0.03448276, 0.06896552, 0.10344828, 0.13793103,
         0.17241379, 0.20689655, 0.24137931, 0.27586207, 0.31034483,
         0.34482759, 0.37931034, 0.4137931 , 0.44827586, 0.48275862,
         0.51724138, 0.55172414, 0.5862069 , 0.62068966, 0.65517241,
         0.68965517, 0.72413793, 0.75862069, 0.79310345, 0.82758621,
         0.86206897, 0.89655172, 0.93103448, 0.96551724, 1.]
    dthdr: [0.06124069, 0.06126784, 0.06124352, 0.06120296, 0.06114596,
            0.06107225, 0.06098148, 0.06087321, 0.06074759, 0.06060343,
            0.0604463 , 0.06029759, 0.06011246, 0.05992273, 0.05973642,
            0.05947572, 0.05919784, 0.05893438, 0.05865396, 0.05834981,
            0.05815874, 0.05775472, 0.05736799, 0.05702492, 0.0565666 ,
            0.0563616 , 0.05585923, 0.05531151, 0.05477258, 0.05446725]
"""


def test_transform_wcs():
    t0 = Time("J2000")
    boresight0 = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
    rot_sky_pos0 = 0 * galsim.degrees
    rot_axis = galsim.CelestialCoord(0 * galsim.degrees, 90 * galsim.degrees)
    rot_rate = 60 * galsim.arcsec
    mount = "EQ"

    tracker = InertialTracker(
        t0=t0,
        boresight0=boresight0,
        rot_sky_pos0=rot_sky_pos0,
        rot_axis=rot_axis,
        rot_rate=rot_rate,
        mount=mount,
    )

    distortion = yaml.safe_load(DTEXT)["distortion"]
    pixel_scale = 25.0  # micron / pixel

    dthdr_arcsec_per_pix = np.asarray(distortion["dthdr"]) * pixel_scale

    wcs0 = radial_wcs(
        distortion["th"],
        dthdr_arcsec_per_pix,
        world_origin=boresight0,
        rot_sky_pos=rot_sky_pos0,
    )

    t1 = t0 + 100 * u.s
    boresight1 = tracker.get_boresight(t1)
    rot_sky_pos1 = tracker.get_rot_sky_pos(t1)

    wcs1 = radial_wcs(
        distortion["th"],
        dthdr_arcsec_per_pix,
        world_origin=boresight1,
        rot_sky_pos=rot_sky_pos1,
    )

    wcs1b = transform_wcs(wcs0, boresight1, rot_sky_pos1)

    np.testing.assert_allclose(
        wcs0.center.distanceTo(wcs1.center).deg * 3600,
        100 * 60,
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        0,
        wcs1.center.distanceTo(wcs1b.center).deg * 3600,
        rtol=0,
        atol=1e-8,
    )

    rng = np.random.default_rng(57721)
    ra, dec = rng.uniform(-0.3, 0.3, size=(2, 1000))
    x1, y1 = wcs1.radecToxy(ra, dec, units="degrees")
    x1b, y1b = wcs1b.radecToxy(ra, dec, units="degrees")

    np.testing.assert_allclose(x1, x1b, rtol=0, atol=1e-3)
    np.testing.assert_allclose(y1, y1b, rtol=0, atol=1e-3)


def test_radial_wcs_two_samples_and_validation():
    center = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
    wcs = radial_wcs([0.0, 1.0], [25.0, 24.0], center)
    x, y = wcs.radecToxy(
        np.array([0.0, 0.25]), np.array([0.0, 0.25]), units="degrees"
    )
    assert np.all(np.isfinite([*x, *y]))

    with pytest.raises(ValueError, match="strictly increasing"):
        radial_wcs([0.1, 1.0], [25.0, 24.0], center)
    with pytest.raises(ValueError, match="strictly positive"):
        radial_wcs([0.0, 1.0], [25.0, 0.0], center)


def test_transform_wcs_rotates_distortion_before_projection():
    th = np.linspace(0, 1, 10)
    scale = 25.0 * (1.0 + 0.15 * th**2)
    center0 = galsim.CelestialCoord(10 * galsim.degrees, 20 * galsim.degrees)
    center1 = galsim.CelestialCoord(10.2 * galsim.degrees, 20.1 * galsim.degrees)
    wcs0 = radial_wcs(th, scale, center0, rot_sky_pos=17 * galsim.degrees)
    expected = radial_wcs(th, scale, center1, rot_sky_pos=41 * galsim.degrees)
    transformed = transform_wcs(wcs0, center1, 41 * galsim.degrees)

    rng = np.random.default_rng(57721)
    ra = rng.uniform(9.9, 10.5, 1000)
    dec = rng.uniform(19.8, 20.4, 1000)
    x_expected, y_expected = expected.radecToxy(ra, dec, units="degrees")
    x_actual, y_actual = transformed.radecToxy(ra, dec, units="degrees")
    np.testing.assert_allclose(x_actual, x_expected, rtol=0, atol=2e-3)
    np.testing.assert_allclose(y_actual, y_expected, rtol=0, atol=2e-3)


def test_transform_wcs_serialized_sip_matches_astropy_and_fresh_wcs():
    th = np.linspace(0, 1, 10)
    scale = 25.0 * (1.0 + 0.15 * th**2)
    center0 = galsim.CelestialCoord(10 * galsim.degrees, 20 * galsim.degrees)
    center1 = galsim.CelestialCoord(10.2 * galsim.degrees, 20.1 * galsim.degrees)
    wcs0 = radial_wcs(th, scale, center0, rot_sky_pos=17 * galsim.degrees)
    expected = radial_wcs(th, scale, center1, rot_sky_pos=41 * galsim.degrees)
    transformed = transform_wcs(wcs0, center1, 41 * galsim.degrees)

    header = fits.Header()
    transformed.writeToFitsHeader(header, galsim.BoundsI(1, 100, 1, 100))
    astropy_wcs = WCS(header)
    assert header["A_ORDER"] == header["B_ORDER"]
    assert header.get("A_1_0", 0.0) == 0.0
    assert header.get("B_0_1", 0.0) == 0.0

    rng = np.random.default_rng(90210)
    ra = rng.uniform(9.9, 10.5, 1000)
    dec = rng.uniform(19.8, 20.4, 1000)
    x_galsim, y_galsim = transformed.radecToxy(ra, dec, units="degrees")
    x_astropy, y_astropy = astropy_wcs.all_world2pix(ra, dec, 1)
    np.testing.assert_allclose(x_astropy, x_galsim, rtol=0, atol=2e-3)
    np.testing.assert_allclose(y_astropy, y_galsim, rtol=0, atol=2e-3)

    ra_astropy, dec_astropy = astropy_wcs.all_pix2world(x_galsim, y_galsim, 1)
    np.testing.assert_allclose(ra_astropy, ra, rtol=0, atol=2e-8)
    np.testing.assert_allclose(dec_astropy, dec, rtol=0, atol=2e-8)

    x_fresh, y_fresh = expected.radecToxy(ra, dec, units="degrees")
    np.testing.assert_allclose(x_galsim, x_fresh, rtol=0, atol=2e-3)
    np.testing.assert_allclose(y_galsim, y_fresh, rtol=0, atol=2e-3)


def test_transform_wcs_factors_noncanonical_linear_roundoff():
    center = galsim.CelestialCoord(10 * galsim.degrees, 20 * galsim.degrees)
    wcs = radial_wcs([0, 1], [25, 25], center)
    wcs.ab[0, 1, 0] = np.nextafter(1.0, 2.0)
    wcs.ab[1, 0, 1] = np.nextafter(1.0, 2.0)
    transformed = transform_wcs(wcs, center, 37 * galsim.degrees)
    np.testing.assert_array_equal(transformed.ab[:, 1, 0], [1.0, 0.0])
    np.testing.assert_array_equal(transformed.ab[:, 0, 1], [0.0, 1.0])
    header = fits.Header()
    transformed.writeToFitsHeader(header, galsim.BoundsI(1, 100, 1, 100))
    for key in ("A_0_0", "A_1_0", "A_0_1", "B_0_0", "B_1_0", "B_0_1"):
        assert header.get(key, 0.0) == 0.0
    # Constant plate scale independently fixes radius; rotation must preserve it.
    x, y = transformed.radecToxy(np.array([10.01]), np.array([20.0]), units="degrees")
    expected_radius = 0.01 * np.cos(np.deg2rad(20.0)) * 3600 / 25
    np.testing.assert_allclose(np.hypot(x, y), expected_radius, atol=1e-4, rtol=0)
