"""
Utility functions for observation simulation and image generation.
"""

import contextlib
import sys

import astropy.units as u
import galsim
import numpy as np
from astropy.coordinates import AltAz, GCRS
from astropy.table import Table
from astropy.time import Time

from .tracker import transform_wcs
from . import backend
from .exposure import sample_exposure

__all__ = [
    "draw_stars",
    "draw_sat",
    "random_dark_time",
    "random_boresight",
    "random_disk",
    "generate_orbit",
    "get_line_gsobject",
    "getLineGSObject",
    "DummyFile",
    "nostdout",
]


def draw_stars(stars, *, t0, exptime, wcs0, tracker, psf, image, nsplit=2, integration=None):
    """
    Draw stars into an image while accounting for tracker motion during exposure.

    Parameters
    ----------
    stars : astropy.table.Table
        Input star table with columns:

        - ``ra``: right ascension in degrees
        - ``dec``: declination in degrees
        - ``nphot``: number of photons to draw
    t0 : astropy.time.Time
        Exposure start time.
    exptime : float
        Exposure time in seconds.
    wcs0 : galsim.GSFitsWCS
        Reference WCS at the start of the exposure.
    tracker : Tracker
        Tracker object providing boresight and rotation as a function of time.
    psf : galsim.GSObject
        Point-spread function model.
    image : galsim.Image
        Output image to draw into.
    nsplit : int, optional
        Number of segments only when ``integration=False`` selects compatibility mode.
    integration : dict or False, optional
        Adaptive error/budget settings. Default uses bounded adaptive sampling;
        ``False`` explicitly selects the fixed-segment compatibility path.

    Returns
    -------
    astropy.table.Table
        Updated table containing the original star information plus:

        - ``x``, ``y``: image coordinates at exposure midpoint
        - ``x_FITS``, ``y_FITS``: FITS-style pixel coordinates
    """
    if integration is not False:
        return _draw_stars_adaptive(
            stars, t0=t0, exptime=exptime, wcs0=wcs0, tracker=tracker,
            psf=psf, image=image, settings=dict(integration or {}),
        )
    if not isinstance(nsplit, (int, np.integer)) or nsplit <= 0:
        raise ValueError("nsplit must be a positive integer.")
    dt = exptime / nsplit * u.s
    ras = []
    decs = []
    xs = []
    ys = []

    for isplit in range(nsplit + 1):
        t = t0 + dt * isplit
        boresight = tracker.get_boresight(t)
        rot_sky_pos = tracker.get_rot_sky_pos(t)
        wcs = transform_wcs(wcs0, boresight, rot_sky_pos)
        x, y = wcs.radecToxy(stars["ra"], stars["dec"], units="degrees")
        ra, dec = wcs0.xyToradec(x, y, units="radians")
        ras.append(ra)
        decs.append(dec)
        xs.append(x)
        ys.append(y)

    tmid = t0 + 0.5 * exptime * u.s
    boresight = tracker.get_boresight(tmid)
    rot_sky_pos = tracker.get_rot_sky_pos(tmid)
    wcs = transform_wcs(wcs0, boresight, rot_sky_pos)
    stars["x"], stars["y"] = wcs.radecToxy(stars["ra"], stars["dec"], units="degrees")

    for istar, star in enumerate(stars):
        for isplit in range(1, nsplit + 1):
            obj = galsim.Convolve(
                get_line_gsobject(
                    galsim.CelestialCoord(
                        ras[isplit - 1][istar] * galsim.radians,
                        decs[isplit - 1][istar] * galsim.radians,
                    ),
                    galsim.CelestialCoord(
                        ras[isplit][istar] * galsim.radians,
                        decs[isplit][istar] * galsim.radians,
                    ),
                ),
                psf,
            )
            xy = galsim.PositionD(
                0.5 * (xs[isplit - 1][istar] + xs[isplit][istar]),
                0.5 * (ys[isplit - 1][istar] + ys[isplit][istar]),
            )
            local_wcs = wcs0.local(xy)
            stamp = (obj * star["nphot"] / nsplit).drawImage(
                wcs=local_wcs,
                center=xy,
                method="auto",
            )
            bounds = stamp.bounds & image.bounds
            if bounds.area() > 0:
                image[bounds] += stamp[bounds]

    stars["x_FITS"] = stars["x"] - image.xmin + 1
    stars["y_FITS"] = stars["y"] - image.ymin + 1

    return stars


def render_expected_segment(start, end, photons, *, wcs, psf, image):
    """Render an expected photon integral on an image-plane line segment."""
    if not np.isfinite(photons) or photons < 0:
        raise ValueError("Expected photon weights must be finite and nonnegative.")
    if photons == 0:
        return
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all():
        raise ValueError("Segment endpoints must be finite pixel pairs.")
    delta = end - start
    length = np.linalg.norm(delta)
    center = galsim.PositionD(*((start + end) / 2))
    local = wcs.local(center)
    if length < 1e-12:
        line = galsim.DeltaFunction()
    else:
        angle = np.arctan2(-delta[0], delta[1]) * galsim.radians
        line = galsim.Box(1e-12, length).rotate(angle)
    obj = galsim.Convolve(local.toWorld(line), psf).withFlux(float(photons))
    stamp = obj.drawImage(wcs=local, center=center, method="auto")
    bounds = stamp.bounds & image.bounds
    if bounds.area() > 0:
        image[bounds] += stamp[bounds]


def _draw_stars_adaptive(stars, *, t0, exptime, wcs0, tracker, psf, image, settings):
    epoch = backend.as_time(t0)
    exptime = float(u.Quantity(exptime, u.s).to_value(u.s))
    if not epoch.isscalar or not np.isfinite(exptime) or exptime <= 0:
        raise ValueError("One epoch and a positive exposure duration are required.")
    cache = {}

    def at(time):
        key = (float(time.jd1), float(time.jd2))
        if key not in cache:
            cache[key] = transform_wcs(
                wcs0, tracker.get_boresight(time), tracker.get_rot_sky_pos(time),
            )
        return cache[key]

    midpoint = at(epoch + exptime / 2 * u.s)
    stars["x"], stars["y"] = midpoint.radecToxy(stars["ra"], stars["dec"], units="degrees")
    evaluations = []
    for star in stars:
        def position(time):
            x, y = at(time).radecToxy(float(star["ra"]), float(star["dec"]), units="degrees")
            return np.array([x, y])

        result = sample_exposure(
            position, lambda time: float(star["nphot"]) / exptime,
            epoch, exptime, **settings,
        )
        evaluations.append(result.evaluated_sample_count)
        if np.all(result.positions_px == result.positions_px[0]):
            render_expected_segment(
                result.positions_px[0], result.positions_px[-1], result.total_photons,
                wcs=wcs0, psf=psf, image=image,
            )
        else:
            for start, end, photons in zip(
                result.positions_px[:-1], result.positions_px[1:], result.segment_photons,
            ):
                render_expected_segment(start, end, photons, wcs=wcs0, psf=psf, image=image)
    stars["integration_samples"] = np.asarray(evaluations, dtype=int)
    stars["x_FITS"] = stars["x"] - image.xmin + 1
    stars["y_FITS"] = stars["y"] - image.ymin + 1
    return stars


def draw_sat(
    orbit,
    *,
    t0,
    exptime,
    wcs0,
    tracker,
    psf,
    image,
    observer,
    nphot=None,
    nsplit=10,
    propagator=None,
    observation_mode="geometric",
    rate_functions=None,
    ab_magnitude_functions=None,
):
    """
    Draw one or more tracked objects into an image.

    Parameters
    ----------
    orbit : sequence of ssatk.Orbit
        Orbits to draw.
    t0 : astropy.time.Time
        Exposure start time.
    exptime : float
        Exposure time in seconds.
    wcs0 : galsim.GSFitsWCS
        Reference WCS at the start of the exposure.
    tracker : Tracker
        Tracker object providing boresight and rotation as a function of time.
    psf : galsim.GSObject
        Point-spread function model.
    image : galsim.Image
        Output image to draw into.
    observer : ssatk.EarthObserver
        Observer location.
    nphot : sequence of float, optional
        Fixed photon counts for each orbit. Required unless ``rate_functions``
        supplies scalar photons/sec callbacks.
    nsplit : int, optional
        Number of time segments used to approximate motion during the exposure.
    propagator : object, optional
        Propagator used for orbit evaluation.
    rate_functions : sequence of callable, optional
        One callback per orbit. Each callback receives one scalar Astropy
        ``Time`` and returns incident photons/sec. Rates are cached per
        exposure and integrated with Simpson's rule on each fixed segment.
    ab_magnitude_functions : sequence of callable, optional
        Optional one-per-orbit callbacks converting an exposure-mean photon
        rate to an AB magnitude.

    Returns
    -------
    table : astropy.table.Table
        Table containing start/end sky and image coordinates and photon counts.
    wcs : galsim.GSFitsWCS or None
        Final WCS used during drawing. Returns None if no orbits were provided.
    """
    orbit = list(orbit)
    if rate_functions is not None and nphot is not None:
        raise ValueError("Choose fixed photon counts or time-varying rate functions, not both.")
    if rate_functions is not None:
        rate_functions = list(rate_functions)
        if len(rate_functions) != len(orbit) or any(
            not callable(fn) for fn in rate_functions
        ):
            raise ValueError("rate_functions must contain one callable per orbit.")
    if ab_magnitude_functions is not None:
        ab_magnitude_functions = list(ab_magnitude_functions)
        if len(ab_magnitude_functions) != len(orbit) or any(
            not callable(fn) for fn in ab_magnitude_functions
        ):
            raise ValueError("ab_magnitude_functions must contain one callable per orbit.")
    if nphot is None:
        if rate_functions is None:
            raise ValueError("nphot is required when rate_functions is not supplied.")
        nphot = np.zeros(len(orbit), dtype=float)
    else:
        nphot = np.array(nphot, dtype=float, copy=True)

    if len(orbit) != len(nphot):
        raise ValueError("orbit and nphot must have the same length.")
    if not isinstance(nsplit, (int, np.integer)) or nsplit <= 0:
        raise ValueError("nsplit must be a positive integer.")

    table = Table()
    ra0 = []
    dec0 = []
    ra1 = []
    dec1 = []
    xs0 = []
    ys0 = []
    xs1 = []
    ys1 = []

    dt = exptime / nsplit
    ts = t0 + np.linspace(0, exptime, nsplit + 1) * u.s
    wcst = None
    mean_ab_mag = np.full(len(orbit), np.nan)

    for idx, orb in enumerate(orbit):
        ra, dec, _ = backend.sky_coordinates(
            orb, ts, observer=observer, propagator=propagator, mode=observation_mode,
        )

        xy0 = wcs0.posToImage(
            galsim.CelestialCoord(ra[0] * galsim.radians, dec[0] * galsim.radians)
        )
        ra0.append(ra[0])
        dec0.append(dec[0])
        ra1.append(ra[-1])
        dec1.append(dec[-1])
        xs0.append(xy0.x)
        ys0.append(xy0.y)

        segment_nphot = np.full(nsplit, nphot[idx] / nsplit)
        if rate_functions is not None:
            cache = {}

            def rate(time):
                key = float(time.gps)
                if key not in cache:
                    result = rate_functions[idx](time)
                    if isinstance(result, u.Quantity):
                        result = result.to_value(1 / u.s)
                    result = float(result)
                    if not np.isfinite(result) or result < 0:
                        raise ValueError(
                            "Satellite rate callbacks must return finite "
                            "nonnegative photons/sec."
                        )
                    cache[key] = result
                return cache[key]

            for isplit in range(nsplit):
                left, right = ts[isplit], ts[isplit + 1]
                midpoint = left + 0.5 * (right - left)
                segment_nphot[isplit] = (right - left).to_value(u.s) / 6 * (
                    rate(left) + 4 * rate(midpoint) + rate(right)
                )
            nphot[idx] = np.sum(segment_nphot)
            if ab_magnitude_functions is not None:
                mean_ab_mag[idx] = ab_magnitude_functions[idx](nphot[idx] / exptime)

        for i in range(1, nsplit + 1):
            t = t0 + i * dt * u.s
            boresight = tracker.get_boresight(t)
            rot_sky_pos = tracker.get_rot_sky_pos(t)
            wcst = transform_wcs(wcs0, boresight, rot_sky_pos)
            xy1 = wcst.posToImage(
                galsim.CelestialCoord(ra[i] * galsim.radians, dec[i] * galsim.radians)
            )
            obj = galsim.Convolve(
                get_line_gsobject(wcs0.posToWorld(xy0), wcs0.posToWorld(xy1)),
                psf,
            )
            xy = (xy0 + xy1) / 2
            local_wcs = wcs0.local(xy)
            stamp = (obj * segment_nphot[i - 1]).drawImage(
                wcs=local_wcs,
                center=xy,
                method="auto",
            )
            bounds = stamp.bounds & image.bounds
            if bounds.area() > 0:
                image[bounds] += stamp[bounds]
            xy0 = xy1

        xs1.append(xy1.x)
        ys1.append(xy1.y)

    table["x0"] = np.asarray(xs0)
    table["y0"] = np.asarray(ys0)
    table["ra0"] = np.rad2deg(np.asarray(ra0))
    table["dec0"] = np.rad2deg(np.asarray(dec0))
    table["ra1"] = np.rad2deg(np.asarray(ra1))
    table["dec1"] = np.rad2deg(np.asarray(dec1))
    table["x1"] = np.asarray(xs1)
    table["y1"] = np.asarray(ys1)
    table["nphot"] = nphot
    table["mean_ab_mag"] = mean_ab_mag

    table["x0_FITS"] = table["x0"] - image.xmin + 1
    table["x1_FITS"] = table["x1"] - image.xmin + 1
    table["y0_FITS"] = table["y0"] - image.ymin + 1
    table["y1_FITS"] = table["y1"] - image.ymin + 1

    return table, wcst


def random_dark_time(*, t_day, site, rng):
    """
    Draw a random time between the next sunset and next sunrise.

    Parameters
    ----------
    t_day : astropy.time.Time
        Time before the night of interest.
    site : object
        Site object providing ``sun_set_time`` and ``sun_rise_time`` methods.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    astropy.time.Time
        Random time during the following night.
    """
    sunset = site.sun_set_time(t_day, which="next")
    sunrise = site.sun_rise_time(sunset, which="next")
    return Time(rng.uniform(sunset.gps, sunrise.gps), format="gps")


def random_boresight(*, observer, t0, horizon, rng):
    """
    Draw a random boresight uniformly from the spherical cap above a horizon cut.

    Parameters
    ----------
    observer : ssatk.EarthObserver
        Observer location.
    t0 : astropy.time.Time
        Observation time.
    horizon : galsim.Angle
        Minimum altitude.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    galsim.CelestialCoord
        Random boresight direction.
    """
    zmin = np.sin(horizon)
    z = rng.uniform(zmin, 1.0)
    alt = np.arctan(z / np.sqrt(1 - z**2))
    az = rng.uniform(0.0, 2 * np.pi)

    location = backend.observer_location(observer)
    sc = AltAz(az * u.rad, alt * u.rad, obstime=t0, location=location)
    radec = sc.transform_to(GCRS(obstime=t0))
    return galsim.CelestialCoord(
        radec.ra.rad * galsim.radians,
        radec.dec.rad * galsim.radians,
    )


def random_disk(center, radius, rng):
    """
    Draw an approximately uniform random point within a small spherical cap.

    Parameters
    ----------
    center : galsim.CelestialCoord
        Cap center.
    radius : galsim.Angle
        Cap radius.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    galsim.CelestialCoord
        Random point near ``center``.

    Notes
    -----
    This uses a small-angle approximation and is most accurate for small radii.
    """
    arclength = np.sqrt(rng.uniform(0, radius.rad**2)) * galsim.radians
    return center.greatCirclePoint(
        galsim.CelestialCoord.from_xyz(*rng.normal(size=3)),
        arclength,
    )


def generate_orbit(height, coord, heading, vperp, vpar, observer, t0):
    """
    Construct an orbit from line-of-sight geometry and assumed velocity components.

    Parameters
    ----------
    height : float
        Height above Earth's surface in meters.
    coord : galsim.CelestialCoord
        Line of sight from observer to object.
    heading : galsim.Angle
        Velocity position angle, measured from north through east.
    vperp : float
        Velocity component perpendicular to the radius vector, in m/s.
    vpar : float
        Velocity component parallel to the radius vector, in m/s.
    observer : ssatk.EarthObserver
        Observer location.
    t0 : astropy.time.Time
        Reference time.

    Returns
    -------
    ssatk.Orbit
        Constructed orbit.
    """
    r_obs, _ = observer.getRV(t0)
    los = np.array(coord.get_xyz())
    b = 2 * np.dot(los, r_obs)
    c = np.sum(r_obs**2) - (backend.EARTH_RADIUS_M + height) ** 2
    disc = b**2 - 4 * c
    if not np.isfinite(disc) or disc < 0:
        raise ValueError("Line of sight does not intersect the requested altitude shell.")
    rho = (-b + np.sqrt(disc)) / 2
    if rho <= 0:
        raise ValueError("Altitude shell must intersect the forward line of sight.")
    r_sat = r_obs + rho * los

    alpha = coord.ra
    delta = coord.dec
    north_los = np.array([
        -np.sin(delta) * np.cos(alpha),
        -np.sin(delta) * np.sin(alpha),
        np.cos(delta),
    ])
    east_los = np.array([-np.sin(alpha), np.cos(alpha), 0.0])
    n_head = north_los * np.cos(heading)
    n_head += east_los * np.sin(heading)

    n_sat = backend.normed(r_sat)
    vpar3 = n_sat * vpar
    vperp3 = n_head - n_sat * np.dot(n_head, n_sat)
    tangent_norm = backend.norm(vperp3)
    if tangent_norm < 1e-12:
        raise ValueError("Heading has no component tangent to the altitude shell.")
    vperp3 *= vperp / tangent_norm
    v_sat = vperp3 + vpar3

    return backend.orbit_from_state(r_sat, v_sat, t0)


def get_line_gsobject(p0, p1):
    """
    Construct a narrow line-shaped GalSim object between two sky positions.

    Parameters
    ----------
    p0, p1 : galsim.CelestialCoord
        Endpoints of the line.

    Returns
    -------
    galsim.GSObject
        Line-like object connecting the two positions.
    """
    ncp = galsim.CelestialCoord(0 * galsim.degrees, 90 * galsim.degrees)
    q = p0.angleBetween(ncp, p1)
    length = p0.distanceTo(p1) / galsim.arcsec
    return galsim.Box(1e-12, max(length, 1e-12)).rotate(q)


class DummyFile:
    """
    File-like wrapper that routes writes through ``tqdm.write``.
    """

    def __init__(self, file):
        self.file = file

    def write(self, x):
        from tqdm import tqdm

        tqdm.write(x, end="", file=self.file)

    def __eq__(self, other):
        return other is self.file


@contextlib.contextmanager
def nostdout():
    """
    Redirect stdout through a tqdm-compatible file wrapper.
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    try:
        yield
    finally:
        sys.stdout = save_stdout


getLineGSObject = get_line_gsobject
