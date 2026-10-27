"""
Observation cadence models for simulation.
"""

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

__all__ = ["SimpleCadence"]


class SimpleCadence:
    """
    Uniformly spaced exposure cadence during the illuminated portion of a pass.

    Parameters
    ----------
    exptime : astropy.time.TimeDelta, astropy.units.Quantity, or float
        Exposure duration. Bare floats are interpreted as seconds.
    delay : astropy.time.TimeDelta, astropy.units.Quantity, or float
        Time between successive shutter openings. Bare floats are interpreted
        as seconds.
    """

    def __init__(self, exptime, delay):
        self.exptime = self._as_timedelta(exptime, "exptime")
        self.delay = self._as_timedelta(delay, "delay")

        if self.exptime.sec <= 0:
            raise ValueError("exptime must be positive.")
        if self.delay.sec <= 0:
            raise ValueError("delay must be positive.")

    @staticmethod
    def _as_timedelta(value, name):
        if isinstance(value, TimeDelta):
            delta = value
            if np.asarray(delta.sec).ndim != 0:
                raise ValueError(f"{name} must be a scalar duration.")
        elif isinstance(value, u.Quantity):
            if not value.isscalar:
                raise ValueError(f"{name} must be a scalar duration.")
            delta = TimeDelta(value.to_value(u.s), format="sec")
        else:
            value = np.asarray(value)
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar duration.")
            seconds = float(value)
            if not np.isfinite(seconds):
                raise ValueError(f"{name} must be finite.")
            delta = TimeDelta(seconds, format="sec")

        if not np.all(np.isfinite(delta.sec)):
            raise ValueError(f"{name} must be finite.")
        return delta

    def get_times(self, pass_):
        """
        Return exposure start and end times for a pass.

        Parameters
        ----------
        ``pass_`` : dict
            Pass description, typically returned by
            ``ssatk.ssapy.compute.refine_pass``.

        Returns
        -------
        t_start, t_end : astropy.time.Time
            Arrays of exposure start and end times.
        """
        if not pass_["illumAtStart"] and not pass_["illumAtEnd"]:
            return Time([], format="mjd"), Time([], format="mjd")

        if pass_["illumAtStart"]:
            t_start = pass_["tStart"]
        else:
            t_start = pass_["tTerminator"]

        if pass_["illumAtEnd"]:
            t_end = pass_["tEnd"]
        else:
            t_end = pass_["tTerminator"]

        available = (t_end - t_start).sec
        if available < self.exptime.sec:
            return Time([], format="mjd"), Time([], format="mjd")

        count = int(
            np.floor((available - self.exptime.sec) / self.delay.sec + 1e-12)
        ) + 1
        offsets = np.arange(count, dtype=float) * self.delay.sec
        starts = t_start + TimeDelta(offsets, format="sec")
        ends = starts + self.exptime
        return starts, ends
