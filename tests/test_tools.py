import galsim
import numpy as np
import pytest
import ssatk
from astropy.table import Table
from astropy.time import Time

from simsit import Instrument, SiderealTracker
from simsit.tools import draw_stars, generate_orbit, get_line_gsobject


class RadialObserver:
    def __init__(self, coord):
        self.position = ssatk.constants.WGS72_EARTH_RADIUS * np.array(coord.get_xyz())

    def getRV(self, time):
        return self.position, np.zeros(3)


@pytest.mark.parametrize("dec", [-80, -30, 0, 30, 80])
@pytest.mark.parametrize("heading", [0, np.pi / 2, np.pi])
def test_orbit_heading_against_spherical_tangent_basis(dec, heading):
    ra = 0.4
    delta = np.deg2rad(dec)
    coord = galsim.CelestialCoord(ra * galsim.radians, delta * galsim.radians)
    orbit = generate_orbit(
        500e3, coord, heading, 7800, 10, RadialObserver(coord), Time("2020-01-01"),
    )
    radial = np.array(coord.get_xyz())
    north = np.array([-np.sin(delta) * np.cos(ra), -np.sin(delta) * np.sin(ra), np.cos(delta)])
    east = np.array([-np.sin(ra), np.cos(ra), 0])
    expected = 7800 * (north * np.cos(heading) + east * np.sin(heading)) + 10 * radial
    np.testing.assert_allclose(orbit.v, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(orbit.r), ssatk.constants.WGS72_EARTH_RADIUS + 500e3)


def test_deterministic_expected_star_flux():
    detector = Instrument(
        image_shape=(41, 41), gain=1, read_noise=0, pixel_scale=1, aperture=1,
        distortion={"th": [0, 0.1, 0.2, 0.3], "dthdr": [1, 1, 1, 1]},
    )
    center = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
    wcs = detector.get_wcs(center, 0 * galsim.degrees)
    stars = Table({"ra": [0.0], "dec": [0.0], "nphot": [1000.0]})
    images = []
    for _ in range(2):
        image = detector.init_image(sky_phot=0, exptime=1)
        draw_stars(
            stars,
            t0=Time("2020-01-01"), exptime=1, wcs0=wcs,
            tracker=SiderealTracker(center, 0 * galsim.degrees),
            psf=galsim.Gaussian(sigma=1), image=image,
        )
        images.append(image.array.copy())
    np.testing.assert_array_equal(images[0], images[1])
    np.testing.assert_allclose(images[0].sum(), 1000, rtol=1e-4)
    assert stars["integration_samples"][0] >= 5
    assert stars["estimated_flux_error"][0] == 0

    empty = Table({"ra": [], "dec": [], "nphot": []})
    draw_stars(
        empty,
        t0=Time("2020-01-01"), exptime=1, wcs0=wcs,
        tracker=SiderealTracker(center, 0 * galsim.degrees),
        psf=galsim.Gaussian(sigma=1), image=detector.init_image(sky_phot=0, exptime=1),
    )
    assert len(empty["integration_samples"]) == 0
    assert len(empty["estimated_flux_error"]) == 0


def test_line_orientation_against_endpoint_geometry():
    start = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
    end = start.deproject(-6 * galsim.arcsec, 8 * galsim.arcsec)
    obj = galsim.Convolve(get_line_gsobject(start, end), galsim.Gaussian(sigma=1))
    image = obj.drawImage(nx=81, ny=81, scale=0.3, method="auto")
    yy, xx = np.indices(image.array.shape)
    xx = (xx - 40) * 0.3
    yy = (yy - 40) * 0.3
    weights = image.array / image.array.sum()
    covariance = np.array([
        [np.sum(weights * xx**2), np.sum(weights * xx * yy)],
        [np.sum(weights * xx * yy), np.sum(weights * yy**2)],
    ])
    # Uniform segment covariance is displacement outer product / 12.
    expected = np.eye(2) * (1 + 0.3**2 / 12) + np.outer([-6, 8], [-6, 8]) / 12
    np.testing.assert_allclose(covariance, expected, atol=0.003, rtol=0)
