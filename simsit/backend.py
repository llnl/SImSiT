"""SSATK observation boundary: GCRF states in SI and explicit GPS/Time epochs."""

import astropy.units as u
import numpy as np
import ssatk
from astropy.coordinates import EarthLocation
from astropy.time import Time, TimeDelta


EARTH_RADIUS_M = ssatk.constants.WGS72_EARTH_RADIUS
OBSERVATION_MODES = ("geometric", "apparent")
__all__ = [
    "as_time", "observation_mode", "orbit_from_state", "keplerian_propagator",
    "states", "sky_coordinates", "observer_from_location", "observer_location",
    "catalog_to_apparent", "find_passes", "refine_pass",
    "lambertian_reflection", "thermal_emission", "faceted_reflection", "facets_from_config",
]


def as_time(time):
    """Accept Astropy Time or explicitly interpreted numeric GPS seconds."""
    if isinstance(time, Time):
        result = time
    else:
        if np.ma.getmaskarray(time).any():
            raise ValueError("Epochs must not be masked.")
        values = np.asarray(time)
        if values.dtype.kind not in "iuf":
            raise TypeError("Epochs must be Astropy Time or numeric GPS seconds.")
        result = Time(values, format="gps")
    if np.any(result.mask):
        raise ValueError("Epochs must not be masked.")
    if result.ndim > 1 or not np.isfinite(result.gps).all():
        raise ValueError("Epochs must be finite scalars or one-dimensional arrays.")
    return result


def observation_mode(mode):
    """Validate the shared satellite pointing/rendering convention."""
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"Observation mode must be one of {OBSERVATION_MODES}.")
    return mode


def _gcrf(frame):
    if frame != "GCRF":
        raise ValueError("Orbit states must use GCRF; transform other frames explicitly.")


def _values(value, unit):
    if np.ma.getmaskarray(value).any():
        raise ValueError("Physical inputs must not be masked.")
    if isinstance(value, u.Quantity):
        value = value.to_value(unit)
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Physical inputs must be finite.")
    return result


def orbit_from_state(position_m, velocity_m_s, time, *, frame="GCRF"):
    """Construct an SSATK orbit from one finite GCRF SI state."""
    _gcrf(frame)
    position = _values(position_m, u.m)
    velocity = _values(velocity_m_s, u.m / u.s)
    epoch = as_time(time)
    if (position.shape != (3,) or velocity.shape != (3,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
            or not epoch.isscalar):
        raise ValueError("One finite position, velocity and scalar epoch are required.")
    return ssatk.Orbit(position, velocity, epoch)


def keplerian_propagator():
    """Return the two-body SSATK propagator used by synthetic orbit examples."""
    return ssatk.KeplerianPropagator()


def states(orbit, time, *, propagator=None, frame="GCRF"):
    """Return positions in meters and velocities in m/s, both in GCRF."""
    _gcrf(frame)
    kwargs = {} if propagator is None else {"propagator": propagator}
    return ssatk.rv(orbit, as_time(time), **kwargs)


def sky_coordinates(orbit, time, *, observer, propagator=None, mode="geometric"):
    """Return RA/Dec in radians and range in meters at photon arrival time.

    ``geometric`` uses simultaneous object/site positions. ``apparent`` asks
    SSATK for iterative light-time and observer-velocity aberration correction.
    Both use the same GCRF axes; neither adds atmospheric refraction.
    """
    observation_mode(mode)
    kwargs = {} if propagator is None else {"propagator": propagator}
    return ssatk.radec(
        orbit, as_time(time), observer=observer,
        obsAngleCorrection=None if mode == "geometric" else "exact", **kwargs,
    )


def observer_from_location(location):
    """Construct an SSATK Earth observer from a scalar WGS84 EarthLocation."""
    if not isinstance(location, EarthLocation) or not location.isscalar:
        raise TypeError("A scalar EarthLocation is required.")
    _values(location.itrs.cartesian.xyz, u.m)
    return ssatk.EarthObserver(
        lon=location.lon.to_value(u.deg), lat=location.lat.to_value(u.deg),
        elevation=location.height.to_value(u.m),
    )


def observer_location(observer):
    """Recover a terrestrial site through the observer's public SI ITRS state."""
    itrs = _values(observer.itrs, u.m)
    if itrs.shape != (3,) or not np.isfinite(itrs).all():
        raise ValueError("An Earth observer must expose one finite ITRS position in meters.")
    return EarthLocation.from_geocentric(*itrs, unit=u.m)


def catalog_to_apparent(ra, dec, time, **kwargs):
    """Convert J2000 stellar coordinates through SSATK's astrometry bridge.

    RA/Dec are radians, proper motions are mu_alpha*/mu_delta in mas/year,
    and parallax is arcseconds. Without an observer this is geocentric.
    """
    ra = _values(ra, u.rad)
    dec = _values(dec, u.rad)
    for name, unit in (("pmra", u.mas / u.yr), ("pmdec", u.mas / u.yr), ("parallax", u.arcsec)):
        if name in kwargs:
            kwargs[name] = _values(kwargs[name], unit)
    return ssatk.ssapy.utils.catalog_to_apparent(ra, dec, as_time(time), **kwargs)


def find_passes(orbit, observers, time, duration, step, **kwargs):
    """SSATK pass search; duration/step are explicit time quantities."""
    intervals = []
    for value in (duration, step):
        if isinstance(value, TimeDelta):
            seconds = value.sec
        elif isinstance(value, u.Quantity):
            seconds = value.to_value(u.s)
        else:
            raise TypeError("Pass intervals must be TimeDelta or time quantities.")
        if np.ndim(seconds) or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Pass intervals must be finite positive scalar durations.")
        intervals.append(seconds * u.s)
    return ssatk.ssapy.compute.find_passes(
        orbit, observers, as_time(time), *intervals, **kwargs,
    )


def refine_pass(orbit, observer, time, **kwargs):
    return ssatk.ssapy.compute.refine_pass(orbit, observer, as_time(time), **kwargs)


def _physical_position(position_gcrf_m):
    """Validate and normalize one explicit physical GCRF position."""
    position = _values(position_gcrf_m, u.m)
    if position.shape != (3,):
        raise ValueError("Physical model position must be one GCRF vector in meters.")
    return position


def _physical_observer(observer, time=None):
    """Validate an observer accepted by SSATK's physical models."""
    if isinstance(observer, EarthLocation):
        if not observer.isscalar:
            raise TypeError("A physical model observer must be scalar.")
        _values(observer.itrs.cartesian.xyz, u.m)
        return observer
    if isinstance(observer, ssatk.EarthObserver):
        return observer
    if hasattr(observer, "getRV"):
        if time is None:
            raise ValueError("Resolving an orbital observer requires an explicit epoch.")
        observer, _ = observer.getRV(as_time(time))
    position = _values(observer, u.m)
    if position.shape != (3,):
        raise ValueError("A physical model observer must be a GCRF vector in meters.")
    return position


def lambertian_reflection(position_gcrf_m, time, observer, **kwargs):
    """Call SSATK's released Lambertian reflection model at an explicit state."""
    from ssatk.compute.lambertian_magnitude import lambertian_reflection as model

    return model(
        _physical_position(position_gcrf_m),
        observer=_physical_observer(observer, time),
        time=as_time(time),
        **kwargs,
    )


def thermal_emission(position_gcrf_m, time, observer, **kwargs):
    """Call SSATK's released gray-body thermal model at an explicit state."""
    from ssatk.compute.lambertian_magnitude import thermal_emission as model

    return model(
        _physical_position(position_gcrf_m),
        observer=_physical_observer(observer, time),
        time=as_time(time),
        **kwargs,
    )


def faceted_reflection(position_gcrf_m, time, observer, *, quaternion, facets, **kwargs):
    """Call SSATK's released faceted reflection model at an explicit state."""
    from ssatk.compute.faceted_magnitude import faceted_reflection as model

    if np.asarray(quaternion).shape != (4,):
        raise ValueError("Physical model quaternion must be one [w, x, y, z] vector.")
    if not np.isfinite(np.asarray(quaternion, dtype=float)).all():
        raise ValueError("Physical model quaternion must be finite.")
    return model(
        _physical_position(position_gcrf_m),
        quaternion=np.asarray(quaternion, dtype=float),
        facets=tuple(facets),
        observer=_physical_observer(observer, time),
        time=as_time(time),
        **kwargs,
    )


def facets_from_config(facets):
    """Construct explicit SSATK facets; no mesh or optical properties are inferred."""
    return tuple(facet if isinstance(facet, ssatk.Facet) else ssatk.Facet(**facet) for facet in facets)


def normed(vector):
    return ssatk.normed(vector)


def norm(vector):
    return ssatk.norm(vector)
