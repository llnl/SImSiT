"""
Observation cadence models for simulation.
"""

import numpy as np
from astropy.time import Time, TimeDelta

__all__ = ["SimpleCadence"]


class SimpleCadence:
    """
    Uniformly spaced exposure cadence during the illuminated portion of a pass.

    Parameters
    ----------
    exptime : astropy.time.TimeDelta or float
        Exposure duration.
    delay : astropy.time.TimeDelta or float
        Time between successive shutter openings.
    """

    def __init__(self, exptime, delay):
        self.exptime = TimeDelta(exptime)
        self.delay = TimeDelta(delay)

        if self.exptime.sec <= 0:
            raise ValueError("exptime must be positive.")
        if self.delay.sec <= 0:
            raise ValueError("delay must be positive.")

    def get_times(self, pass_):
        """
        Return exposure start and end times for a pass.

        Parameters
        ----------
        pass_ : dict
            Pass description, typically returned by
            ``ssapy.compute.refine_pass``.

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

        starts = Time(
            np.arange(t_start.gps, t_end.gps, self.delay.sec),
            format="gps",
        )
        ends = starts + self.exptime
        return starts, ends