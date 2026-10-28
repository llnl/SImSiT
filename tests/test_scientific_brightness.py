"""Independent references for physical brightness and rendered photon flux."""

from __future__ import annotations

from scipy.integrate import quad

import astropy.units as u
from astropy.time import Time
import galsim
import numpy as np
import pytest

from simsit import Instrument, Passband, SiderealTracker
from simsit.brightness import PhysicalBrightness
from simsit.tools import draw_sat
import simsit.backend as backend


H = 6.62607015e-34
C = 299792458.0
K_B = 1.380649e-23
POSITION = np.array([7.0e6, 0.0, 0.0]) * u.m
OBSERVER = np.array([7.0e6, 7.0e6, 0.0]) * u.m
EPOCH = Time("2025-01-01")
TEMPERATURE = 5800.0
RADIUS = 1.0


def _instrument():
    return Instrument(
        image_shape=(41, 41),
        gain=2.0,
        read_noise=0.0,
        quantum_efficiency=1.0,
        pixel_scale=1.0,
        aperture=1.0,
        passband=Passband(
            "scientific-brightness",
            [500.0, 550.0, 600.0],
            [0.2, 0.8, 0.3],
            "Synthetic piecewise-linear validation response",
        ),
    )


def _brightness(instrument):
    return PhysicalBrightness(
        instrument,
        params={
            "radius_m": RADIUS,
            "albedo": 0.0,
            "include_sun": False,
            "include_earthshine": False,
            "include_moonshine": False,
            "include_thermal": True,
            "temperature_K": TEMPERATURE,
            "emissivity": 1.0,
        },
        spectral_relative_tolerance=2e-3,
    )


def _independent_planck_passband_rate(instrument):
    """Integrate Planck radiance and the tabulated response independently."""
    distance = np.linalg.norm((POSITION - OBSERVER).to_value(u.m))
    passband = instrument.passband

    def integrand(wavelength):
        response = np.interp(
            wavelength,
            passband.wavelength_nm * 1e-9,
            passband.throughput,
        )
        radiance = (
            np.pi * 2.0 * H * C**2 / wavelength**5
            / np.expm1(H * C / (wavelength * K_B * TEMPERATURE))
        )
        irradiance = radiance * (RADIUS / distance) ** 2
        return irradiance * response * wavelength / (H * C)

    integral = quad(
        integrand,
        passband.wavelength_nm[0] * 1e-9,
        passband.wavelength_nm[-1] * 1e-9,
        epsabs=0.0,
        epsrel=1e-11,
        points=passband.wavelength_nm[1:-1] * 1e-9,
    )[0]
    return instrument.collecting_area * integral


def test_physical_brightness_matches_independent_planck_passband_integral():
    instrument = _instrument()
    actual = _brightness(instrument).photon_rate(
        POSITION, EPOCH, OBSERVER,
    ).to_value(1 / u.s)
    expected = _independent_planck_passband_rate(instrument)

    assert actual == pytest.approx(expected, rel=4e-3)


def test_physical_brightness_rate_reaches_rendered_integrated_flux(monkeypatch):
    instrument = _instrument()
    brightness = _brightness(instrument)
    wcs = galsim.TanWCS(
        galsim.AffineTransform(
            1.0, 0.0, 0.0, 1.0, origin=galsim.PositionD(0.0, 0.0),
        ),
        galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees),
        units=galsim.arcsec,
    )

    def fixed_sky_coordinates(_orbit, times, **_kwargs):
        count = len(np.atleast_1d(times))
        coord = wcs.posToWorld(galsim.PositionD(0.23, -0.31))
        return (
            np.full(count, coord.ra.rad),
            np.full(count, coord.dec.rad),
            np.full(count, np.linalg.norm(POSITION.to_value(u.m))),
        )

    monkeypatch.setattr(backend, "sky_coordinates", fixed_sky_coordinates)
    image = galsim.Image(galsim.BoundsI(-20, 20, -20, 20))

    def physical_rate(time):
        return brightness.photon_rate(POSITION, time, OBSERVER)

    class SyntheticOrbit:
        pass

    table, _ = draw_sat(
        [SyntheticOrbit()],
        t0=EPOCH,
        exptime=2.0,
        wcs0=wcs,
        tracker=SiderealTracker(wcs.center, 0 * galsim.degrees),
        psf=galsim.Gaussian(
            sigma=1.0,
            gsparams=galsim.GSParams(
                folding_threshold=1e-6,
                maxk_threshold=1e-6,
            ),
        ),
        image=image,
        observer=None,
        nsplit=2,
        rate_functions=[physical_rate],
    )

    expected_rate = _independent_planck_passband_rate(instrument)
    expected_photons = 2.0 * expected_rate
    assert table["nphot"][0] == pytest.approx(expected_photons, rel=5e-3)
    assert image.array.sum() == pytest.approx(expected_photons, rel=5e-3)
