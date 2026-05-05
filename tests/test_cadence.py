import astropy.units as u
import numpy as np
import pytest
import simsit
import ssapy
from astropy.time import Time


def test_simple_cadence():
    exptime = 1.3 * u.s
    delay = 11.1 * u.s
    cadence = simsit.SimpleCadence(exptime, delay)

    rng = np.random.default_rng(1234)
    t0 = Time("2010-01-01T00:00:00")
    orbit = ssapy.Orbit.fromKeplerianElements(
        ssapy.constants.WGS72_EARTH_RADIUS + rng.uniform(400e3, 1000e3),
        rng.uniform(0.001, 0.002),
        np.deg2rad(60.0 + rng.uniform(-5.0, 5.0)),
        rng.uniform(0.0, 2 * np.pi),
        rng.uniform(0.0, 2 * np.pi),
        rng.uniform(0.0, 2 * np.pi),
        t0,
        propkw={
            "area": rng.uniform(1.0, 2.0),
            "mass": rng.uniform(100.0, 500.0),
            "CD": rng.uniform(1.8, 2.2),
            "CR": rng.uniform(1.1, 1.5),
        },
    )

    observers = []
    for _ in range(4):
        observers.append(
            ssapy.EarthObserver(
                lon=rng.uniform(0.0, 360.0),
                lat=rng.uniform(30.0, 70.0),
                elevation=rng.uniform(100.0, 3000.0),
                fast=True,
            )
        )

    prop = ssapy.propagator.KeplerianPropagator()

    try:
        passes = ssapy.compute.find_passes(
            orbit,
            observers,
            t0,
            1 * u.d,
            5 * u.min,
            propagator=prop,
            horizon=np.deg2rad(10.0),
        )
    except Exception as exc:
        pytest.skip(f"Skipping cadence integration test due to external ssapy/IERS issue: {exc}")

    for observer, times in passes.items():
        for time in times[:2]:
            refinement = ssapy.compute.refine_pass(
                orbit,
                observer,
                time,
                propagator=prop,
                horizon=np.deg2rad(10.0),
            )
            t_start, t_end = cadence.get_times(refinement)

            np.testing.assert_allclose(t_end.gps - t_start.gps, exptime.to(u.s).value)
            np.testing.assert_allclose(np.diff(t_end.gps), delay.to(u.s).value)
            np.testing.assert_allclose(np.diff(t_start.gps), delay.to(u.s).value)

            alt, az = ssapy.quickAltAz(
                refinement["orbit"],
                t_start,
                refinement["observer"],
                propagator=refinement["propagator"],
            )
            assert np.all(alt >= np.deg2rad(10.0 - 1 / 60))

            r, v = ssapy.rv(
                refinement["orbit"],
                t_start,
                propagator=refinement["propagator"],
            )
            r_par, r_perp = ssapy.compute.earthShadowCoords(r, t_start)
            shadow_height = r_perp - ssapy.constants.WGS72_EARTH_RADIUS

            assert np.all((r_par < 1) | (shadow_height > -1))