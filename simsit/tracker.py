"""
Tracking models for observation simulation.

This module defines a small set of tracker classes that provide telescope
boresight and camera orientation as a function of time.
"""

from abc import ABC, abstractmethod

import galsim
import numpy as np
import ssapy

__all__ = [
    "Tracker",
    "OrbitTracker",
    "InertialTracker",
    "sidereal_tracker",
    "SiderealTracker",
    "transform_wcs",
]


class Tracker(ABC):
    """
    Abstract base class for telescope tracking models.

    Subclasses must provide the boresight direction and camera rotation angle
    at a requested time.
    """

    @abstractmethod
    def get_boresight(self, time):
        """
        Return the telescope boresight at the requested time.

        Parameters
        ----------
        time : astropy.time.Time
            Time at which to evaluate the boresight.

        Returns
        -------
        galsim.CelestialCoord
            Boresight pointing direction.
        """
        raise NotImplementedError

    @abstractmethod
    def get_rot_sky_pos(self, time):
        """
        Return the camera rotation angle at the requested time.

        Parameters
        ----------
        time : astropy.time.Time
            Time at which to evaluate the rotation angle.

        Returns
        -------
        galsim.Angle
            Camera rotation angle relative to the sky.
        """
        raise NotImplementedError


class OrbitTracker(Tracker):
    """
    Track a moving object by following its predicted sky position.

    Parameters
    ----------
    orbit : ssapy.Orbit
        Orbit to track.
    observer : ssapy.EarthObserver
        Observer location.
    t0 : astropy.time.Time
        Reference time for tracking.
    rot_sky_pos0 : galsim.Angle
        Camera rotation angle at the reference time.
    mount : {'EQ'}, optional
        Mount type. Only equatorial mounts are currently supported.
    propagator : ssapy.Propagator, optional
        Propagator used to predict object motion.
    """

    def __init__(self, *, orbit, observer, t0, rot_sky_pos0, mount="EQ", propagator=None):
        mount = mount.upper()
        if mount != "EQ":
            raise ValueError(f"Unsupported mount type: {mount}. Only 'EQ' is implemented.")

        self.orbit = orbit
        self.observer = observer
        self.t0 = t0
        self.rot_sky_pos0 = rot_sky_pos0
        self.mount = mount
        self.propagator = propagator

    def get_boresight(self, time):
        """
        Return the tracked object's apparent sky position.

        Parameters
        ----------
        time : astropy.time.Time
            Time at which to evaluate the pointing.

        Returns
        -------
        galsim.CelestialCoord
            Apparent right ascension and declination of the tracked object.
        """
        ra, dec, _ = ssapy.radec(
            self.orbit,
            time,
            observer=self.observer,
            propagator=self.propagator,
        )
        return galsim.CelestialCoord(ra * galsim.radians, dec * galsim.radians)

    def get_rot_sky_pos(self, time):
        """
        Return the camera rotation angle.

        For the currently implemented equatorial-mount behavior, the camera
        rotation angle is held fixed.

        Parameters
        ----------
        time : astropy.time.Time
            Evaluation time. Currently unused.

        Returns
        -------
        galsim.Angle
            Camera rotation angle.
        """
        return self.rot_sky_pos0


class InertialTracker(Tracker):
    """
    Track by rotating the boresight at a constant angular rate in inertial space.

    Parameters
    ----------
    t0 : astropy.time.Time
        Reference time for tracking.
    boresight0 : galsim.CelestialCoord
        Boresight at ``t0``.
    rot_sky_pos0 : galsim.Angle
        Camera rotation angle at ``t0``.
    rot_axis : galsim.CelestialCoord or None
        Axis of inertial rotation. Ignored if ``rot_rate`` is zero.
    rot_rate : galsim.Angle or float
        Rotation rate per second. If zero, the boresight is fixed.
    mount : {'EQ'}, optional
        Mount type. Only equatorial mounts are currently supported.

    Notes
    -----
    The boresight is propagated using the Rodrigues rotation formula.
    """

    def __init__(self, *, t0, boresight0, rot_sky_pos0, rot_axis, rot_rate, mount="EQ"):
        mount = mount.upper()
        if mount != "EQ":
            raise ValueError(f"Unsupported mount type: {mount}. Only 'EQ' is implemented.")

        self.t0 = t0
        self.boresight0 = boresight0
        self.rot_sky_pos0 = rot_sky_pos0
        self.rot_axis = rot_axis
        self.rot_rate = rot_rate
        self.mount = mount

        self._is_static = (rot_rate == 0 or rot_rate == 0.0)

        if not self._is_static:
            if t0 is None:
                raise ValueError("t0 must be provided when rot_rate is nonzero.")
            if rot_axis is None:
                raise ValueError("rot_axis must be provided when rot_rate is nonzero.")

            self._xyz0 = np.array(self.boresight0.get_xyz())
            axis_xyz = np.array(self.rot_axis.get_xyz())

            axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)

            self._cross = np.cross(axis_xyz, self._xyz0)
            self._dot = np.dot(axis_xyz, self._xyz0) * axis_xyz

    def get_boresight(self, time):
        """
        Return the boresight at the requested time.

        Parameters
        ----------
        time : astropy.time.Time
            Time at which to evaluate the boresight.

        Returns
        -------
        galsim.CelestialCoord
            Boresight direction rotated by ``(time - t0) * rot_rate`` about
            ``rot_axis``.
        """
        if self._is_static:
            return self.boresight0

        theta = (time - self.t0).sec * self.rot_rate

        xyz = (
            np.cos(theta) * self._xyz0
            + np.sin(theta) * self._cross
            + (1 - np.cos(theta)) * self._dot
        )
        return galsim.CelestialCoord.from_xyz(*xyz)

    def get_rot_sky_pos(self, time):
        """
        Return the camera rotation angle.

        For the currently implemented equatorial-mount behavior, the camera
        rotation angle is held fixed.

        Parameters
        ----------
        time : astropy.time.Time
            Evaluation time. Currently unused.

        Returns
        -------
        galsim.Angle
            Camera rotation angle.
        """
        return self.rot_sky_pos0


def sidereal_tracker(boresight0, rot_sky_pos0):
    """
    Create a tracker fixed with respect to the inertial sky.

    Parameters
    ----------
    boresight0 : galsim.CelestialCoord
        Fixed boresight direction.
    rot_sky_pos0 : galsim.Angle
        Fixed camera rotation angle.

    Returns
    -------
    InertialTracker
        Tracker with zero inertial rotation rate.
    """
    return InertialTracker(
        t0=None,
        boresight0=boresight0,
        rot_sky_pos0=rot_sky_pos0,
        rot_axis=None,
        rot_rate=0.0,
        mount="EQ",
    )


def transform_wcs(wcs0, boresight1, rot_sky_pos1):
    """
    Recenter and rotate a WCS.

    Parameters
    ----------
    wcs0 : galsim.GSFitsWCS
        Input WCS.
    boresight1 : galsim.CelestialCoord
        New field center.
    rot_sky_pos1 : galsim.Angle
        New camera rotation angle.

    Returns
    -------
    galsim.GSFitsWCS
        Transformed WCS.

    Notes
    -----
    This assumes the original camera rotation angle is encoded in the WCS CD
    matrix and that the WCS center corresponds to the original boresight.
    """
    rot_sky_pos0 = np.arctan2(-wcs0.cd[0, 1], -wcs0.cd[0, 0])
    delta_theta = rot_sky_pos1.rad - rot_sky_pos0

    sin_theta = np.sin(delta_theta)
    cos_theta = np.cos(delta_theta)

    wcs = wcs0.copy()
    rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    wcs.cd = rotation @ wcs.cd

    if hasattr(wcs, "cdinv"):
        del wcs.cdinv

    wcs.center = boresight1
    return wcs


SiderealTracker = sidereal_tracker