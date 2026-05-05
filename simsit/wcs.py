"""
World Coordinate System (WCS) generation with radial optical distortion.
"""

import galsim
import numpy as np


def radial_wcs(
    th,
    dthdr,
    world_origin,
    rot_sky_pos=0 * galsim.degrees,
    n_control=10,
    sip_order=3,
    verbose=False,
):
    """
    Make a WCS from a radial distortion model.

    Parameters
    ----------
    th : array-like
        Field angles in degrees.
    dthdr : array-like
        Radial plate scale in arcsec per pixel.
    world_origin : galsim.CelestialCoord
        Origin of RA and Dec.
    rot_sky_pos : galsim.Angle, optional
        Rotation angle on the sky.
    n_control : int, optional
        Number of control points to use.
    sip_order : int, optional
        Order of the SIP part of the fitted WCS.
    verbose : bool, optional
        If True, print residual statistics for the fitted WCS.

    Returns
    -------
    galsim.GSFitsWCS
        Fitted WCS object.
    """
    import warnings

    from scipy.integrate import IntegrationWarning, quad
    from scipy.interpolate import interp1d

    th = np.asarray(th, dtype=float)
    dthdr = np.asarray(dthdr, dtype=float)

    u = np.deg2rad(np.linspace(-th[-1], th[-1], n_control))
    u, v = np.meshgrid(u, u)
    rho = np.hypot(u, v)

    w = rho <= np.deg2rad(th[-1])
    u = u[w]
    v = v[w]
    rho = rho[w]

    interp = interp1d(th, dthdr, kind="cubic")  # deg -> arcsec/pix
    integrand = lambda arcsec: 1.0 / interp(arcsec / 3600.0)  # arcsec -> pix/arcsec

    x = np.empty_like(u)
    y = np.empty_like(u)

    for idx in np.ndindex(u.shape):
        if rho[idx] == 0:
            x[idx] = 0.0
            y[idx] = 0.0
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            r = quad(integrand, 0, np.rad2deg(rho[idx]) * 3600)[0]

        x[idx] = r * u[idx] / rho[idx]
        y[idx] = r * v[idx] / rho[idx]

    sth, cth = np.sin(rot_sky_pos), np.cos(rot_sky_pos)
    R = np.array([[cth, -sth], [sth, cth]])
    x, y = R @ np.array([x, y])

    ra, dec = world_origin._deproject(u, v, projection="postel")
    wcs = galsim.FittedSIPWCS(x, y, ra, dec, order=sip_order)

    if verbose:
        x1, y1 = wcs.radecToxy(ra, dec, units="radians")
        print(
            "WCS inversion x residuals:"
            f" mean = {np.mean(x - x1)}"
            f" std = {np.std(x - x1)}"
        )
        print(
            "WCS inversion y residuals:"
            f" mean = {np.mean(y - y1)}"
            f" std = {np.std(y - y1)}"
        )

    return wcs


# Backward-compatible alias
radialWCS = radial_wcs