import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time
from scipy.integrate import quad
import ssatk

from simsit import Instrument, Passband
from simsit.brightness import PhysicalBrightness
from ssatk import Facet


@pytest.fixture
def instrument():
    return Instrument(
        image_shape=(8, 8), gain=2.0, read_noise=1.0, pixel_scale=1.0,
        aperture=1.0, quantum_efficiency=0.4,
        passband=Passband("top-hat", [500, 600], [1, 1], "test"),
    )


POSITION = np.array([7.0e6, 0, 0]) * u.m
OBSERVER = np.array([7.0e6, 7.0e6, 0]) * u.m
EPOCH = Time("2025-01-01")


def test_incident_rate_is_separate_from_qe_and_matches_flat_spectrum(instrument):
    brightness = PhysicalBrightness(
        instrument,
        params={
            "radius_m": 1.0, "albedo": 0.0, "include_sun": False,
            "include_earthshine": False,
            "include_moonshine": False, "include_thermal": True,
            "temperature_K": 5800.0, "emissivity": 1.0,
        },
        spectral_relative_tolerance=1e-3,
    )
    actual = brightness.photon_rate(
        POSITION, EPOCH, OBSERVER,
    ).to_value(1 / u.s)
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    temperature = 5800.0
    radius = 1.0
    distance = np.linalg.norm((POSITION - OBSERVER).to_value(u.m))
    expected = instrument.collecting_area * quad(
        lambda wavelength: (
            np.pi * (2 * h * c**2 / wavelength**5)
            / np.expm1(h * c / (wavelength * k * temperature))
            * (radius / distance) ** 2
            * wavelength / (h * c)
        ),
        500e-9, 600e-9, epsabs=0.0, epsrel=1e-11,
    )[0]
    assert actual == pytest.approx(expected, rel=3e-3)
    assert brightness.instrument.quantum_efficiency == 0.4


def test_lambertian_reflection_has_expected_range_and_albedo_scaling(instrument):
    one = PhysicalBrightness(
        instrument,
        params={
            "radius_m": 1.0, "albedo": 0.2, "k_extinction": 0.0,
            "include_earthshine": False, "include_moonshine": False,
        },
        spectral_relative_tolerance=2e-3,
    )
    two = PhysicalBrightness(
        instrument,
        params={
            "radius_m": 1.0, "albedo": 0.4, "k_extinction": 0.0,
            "include_earthshine": False, "include_moonshine": False,
        },
        spectral_relative_tolerance=2e-3,
    )
    first = one.photon_rate(POSITION, EPOCH, OBSERVER).value
    second = two.photon_rate(POSITION, EPOCH, OBSERVER).value
    assert second / first == pytest.approx(2.0, rel=4e-3)
    farther = np.array([7.0e6, 14.0e6, 0]) * u.m
    distant = one.photon_rate(POSITION, EPOCH, farther).value
    assert first / distant == pytest.approx(4.0, rel=4e-3)

    sun_direction = ssatk.ssapy.utils.sunPos(EPOCH) - POSITION.to_value(u.m)
    observer_direction = OBSERVER.to_value(u.m) - POSITION.to_value(u.m)
    normal = sun_direction / np.linalg.norm(sun_direction)
    normal += observer_direction / np.linalg.norm(observer_direction)
    normal /= np.linalg.norm(normal)
    facet = Facet(area=2.0, normal_body=normal, diffuse_reflectivity=0.5)
    faceted = PhysicalBrightness(
        instrument,
        model="faceted",
        params={
            "facets": [facet], "quaternion": [1.0, 0.0, 0.0, 0.0],
            "include_earthshine": False, "include_moonshine": False,
        },
        spectral_relative_tolerance=2e-3,
    )
    assert faceted.photon_rate(POSITION, EPOCH, OBSERVER).value > 0


def test_blackbody_oracle_shape_and_bounded_nonconvergence(instrument):
    brightness = PhysicalBrightness(
        instrument,
        params={
            "radius_m": 1.0, "albedo": 0.0, "include_sun": False,
            "include_earthshine": False,
            "include_moonshine": False, "include_thermal": True,
            "temperature_K": 5800.0, "emissivity": 1.0,
        },
        max_spectral_samples=11,
        spectral_relative_tolerance=1e-12,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        brightness.photon_rate(
            POSITION, EPOCH, OBSERVER,
        )


def test_dark_and_faceted_models_return_zero_without_illumination(instrument):
    dark = PhysicalBrightness(
        instrument,
        params={
            "radius_m": 1.0, "albedo": 0.3, "include_sun": False,
            "include_earthshine": False,
            "include_moonshine": False,
        },
    )
    assert dark.photon_rate(POSITION, EPOCH, OBSERVER).value == 0.0

    facet = Facet(
        area=2.0, normal_body=[1.0, 0.0, 0.0], diffuse_reflectivity=0.5,
    )
    faceted = PhysicalBrightness(
        instrument,
        model="faceted",
        params={
            "facets": [facet], "quaternion": [1.0, 0.0, 0.0, 0.0],
            "include_sun": False, "include_earthshine": False,
            "include_moonshine": False,
        },
    )
    assert faceted.photon_rate(POSITION, EPOCH, OBSERVER).value == 0.0
