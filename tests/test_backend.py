import astropy.units as u
import numpy as np
import pytest
import ssatk
from astropy.coordinates import EarthLocation
from astropy.time import Time

from simsit import backend


def test_state_units_and_circular_conservation():
    radius = 7e6
    speed = np.sqrt(ssatk.constants.EARTH_MU / radius)
    epoch = Time("2020-01-01")
    orbit = backend.orbit_from_state(
        [7000, 0, 0] * u.km, [0, speed / 1000, 0] * u.km / u.s, epoch,
    )
    times = epoch + np.arange(0, 601, 100) * u.s
    r, v = backend.states(orbit, times, propagator=backend.keplerian_propagator())
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), radius, atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), speed, atol=1e-8, rtol=0)
    with pytest.raises(ValueError, match="GCRF"):
        backend.states(orbit, epoch, frame="ITRF")


def test_explicit_epoch_interpretation():
    np.testing.assert_allclose(backend.as_time([1e9, 1e9 + 1]).gps, [1e9, 1e9 + 1])
    with pytest.raises(TypeError, match="GPS"):
        backend.as_time("2020-01-01")
    with pytest.raises(ValueError, match="finite"):
        backend.as_time(np.nan)
    with pytest.raises(ValueError, match="masked"):
        backend.as_time(np.ma.array([1e9], mask=[True]))
    with pytest.raises(ValueError, match="masked"):
        backend.orbit_from_state(
            np.ma.array([7e6, 0, 0], mask=[True, False, False]),
            [0, 7500, 0], Time("2020-01-01"),
        )


def test_public_terrestrial_coordinates_recover_site():
    location = EarthLocation.from_geodetic(-106.6 * u.deg, 35.1 * u.deg, 1600 * u.m)
    observer = backend.observer_from_location(location)
    recovered = backend.observer_location(observer)
    np.testing.assert_allclose(
        [recovered.x.value, recovered.y.value, recovered.z.value],
        [location.x.value, location.y.value, location.z.value], atol=1e-8, rtol=0,
    )


class CenterObserver:
    def getRV(self, time):
        shape = np.shape(time) + (3,)
        return np.zeros(shape), np.zeros(shape)


def test_light_time_against_retarded_circular_orbit():
    radius = 7e6
    speed = np.sqrt(ssatk.constants.EARTH_MU / radius)
    rate = speed / radius
    epoch = Time("2020-01-01")
    orbit = backend.orbit_from_state([radius, 0, 0], [0, speed, 0], epoch)
    time = epoch + 100 * u.s
    geometric = backend.sky_coordinates(orbit, time, observer=CenterObserver())
    apparent = backend.sky_coordinates(orbit, time, observer=CenterObserver(), mode="apparent")
    np.testing.assert_allclose(geometric[0], rate * 100, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        apparent[0], rate * (100 - radius / 299792458.0), atol=1e-10, rtol=0,
    )
    np.testing.assert_allclose(apparent[1], 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(apparent[2], radius, atol=1e-6, rtol=0)


def test_catalog_units_against_tangent_motion():
    ra, dec = backend.catalog_to_apparent(
        np.array([0.0]) * u.deg, np.array([0.0]) * u.deg, Time("J2001"),
        pmra=np.array([1000.0]) * u.mas / u.yr,
        pmdec=np.array([0.0]) * u.mas / u.yr,
        parallax=np.array([0.0]) * u.mas, skipAberration=True,
    )
    expected = np.arctan(np.deg2rad(1 / 3600))
    np.testing.assert_allclose(ra, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(dec, 0, atol=1e-12, rtol=0)
