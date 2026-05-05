"""
Star catalog interfaces for observation simulation.
"""

import os
from abc import ABC, abstractmethod

import astropy.units as u
import lsst.sphgeom as sphgeom
import numpy as np
from astropy.table import Table, vstack
from ssapy.utils import catalog_to_apparent

from .photometry import vega_to_ab_offset

__all__ = ["StarCatalog", "MockStarCatalog", "GaiaStarCatalog"]


def _coord_to_uv3d(coord):
    """
    Convert a celestial coordinate to an ``sphgeom.UnitVector3d``.

    Parameters
    ----------
    coord : galsim.CelestialCoord
        Coordinate to convert.

    Returns
    -------
    sphgeom.UnitVector3d
        Unit-vector representation of the coordinate.
    """
    return sphgeom.UnitVector3d(*coord.get_xyz())


def _circle_to_poly(circle, nvertex=100):
    """
    Approximate a spherical circle with a convex polygon.

    Parameters
    ----------
    circle : sphgeom.Circle
        Circle to approximate.
    nvertex : int, optional
        Number of polygon vertices.

    Returns
    -------
    sphgeom.ConvexPolygon
        Polygon approximation to the circle.
    """
    center = circle.getCenter()
    point = center.rotatedAround(
        sphgeom.UnitVector3d.northFrom(center),
        circle.getOpeningAngle(),
    )

    th = np.linspace(0, 2 * np.pi, nvertex)
    points = [
        point.rotatedAround(center, sphgeom.Angle.fromRadians(th_))
        for th_ in th
    ]
    return sphgeom.ConvexPolygon(points)


class StarCatalog(ABC):
    """
    Abstract base class for star catalogs using HTM pixelization.

    Parameters
    ----------
    level : int, optional
        HTM pixelization level.
    """

    def __init__(self, level=7):
        self.level = level
        self.htm = sphgeom.HtmPixelization(level)

    def get_stars(self, coord0, coord1, radius, time):
        """
        Get stars within a field of view, accounting for boresight motion.

        Parameters
        ----------
        coord0, coord1 : galsim.CelestialCoord
            Field center at the beginning and end of the exposure.
        radius : galsim.Angle
            Field radius.
        time : astropy.time.Time
            Observation time.

        Returns
        -------
        astropy.table.Table
            Table containing at least:

            - ``ra`` in degrees
            - ``dec`` in degrees
            - ``i_mag``
        """
        c0 = sphgeom.Circle(
            _coord_to_uv3d(coord0),
            sphgeom.Angle.fromDegrees(radius.deg),
        )
        c1 = sphgeom.Circle(
            _coord_to_uv3d(coord1),
            sphgeom.Angle.fromDegrees(radius.deg),
        )

        poly0 = _circle_to_poly(c0)
        poly1 = _circle_to_poly(c1)
        poly = sphgeom.ConvexPolygon(poly0.getVertices() + poly1.getVertices())

        ranges = self.htm.envelope(poly)
        table_list = []

        for start, end in ranges:
            for idx in range(start, end):
                table_list.append(self._get_trixel_stars(idx, time))

        if not table_list:
            return Table()

        out = vstack(table_list)
        w = poly.contains(out["ra_rad"], out["dec_rad"])
        return out[w]

    @abstractmethod
    def _get_trixel_stars(self, idx, time):
        """
        Return stars from a single HTM trixel.

        Parameters
        ----------
        idx : int
            HTM trixel index.
        time : astropy.time.Time
            Observation time.

        Returns
        -------
        astropy.table.Table
            Table of stars in the trixel.
        """
        raise NotImplementedError


class MockStarCatalog(StarCatalog):
    """
    Deterministic mock star catalog generated on the fly.

    For a fixed HTM level and seed, the same query always produces the same
    synthetic stars.

    Parameters
    ----------
    level : int, optional
        HTM level.
    seed : int, optional
        Random seed controlling deterministic generation.
    magnitude_hists : dict, optional
        Precomputed magnitude histograms keyed by filter name. If omitted,
        only ``i_mag`` is generated from the internal histogram.
    """

    def __init__(self, level=7, seed=123, magnitude_hists=None):
        super().__init__(level=level)
        self.seed = seed
        self.magnitude_hists = magnitude_hists

        self.imag_hist = np.array([
            129, 168, 180, 170, 191, 213, 233, 261, 249, 300, 338, 350, 393,
            413, 437, 476, 502, 504, 596, 633, 687, 717, 753, 850, 898, 980,
            1027, 1122, 1110, 1230, 1347, 1382, 1418, 1580, 1687, 1774, 1818,
            1972, 2058, 2173, 2275, 2419, 2544, 2679, 2905, 3062, 3135, 3286,
            3469, 3619, 3748, 3894, 4079, 4334, 4599, 4736, 4898, 5114, 5577,
            5695, 5772, 5992, 6346, 6562, 6716, 7072, 7293, 7516, 7833, 8116,
            8161, 8800, 9135, 9282, 9633, 10074, 10294, 10719, 11378, 11609,
            12084, 12509, 12647, 13454, 14090, 14407, 15194, 15699, 16437,
            16951, 17812, 18275, 19097, 20012, 20865, 21666, 22424, 23508,
            24474, 25621, 26306, 27156, 28272, 28551, 29535, 30510, 31067,
            32346, 33013, 33201,
        ])
        self.bins = np.linspace(10, 21, 111)
        self.density = np.sum(self.imag_hist) / 0.102219849198379

    def _get_trixel_stars(self, idx, time):
        """
        Generate synthetic stars for a trixel.

        Parameters
        ----------
        idx : int
            HTM trixel index.
        time : astropy.time.Time
            Observation time. Unused for the mock catalog.

        Returns
        -------
        astropy.table.Table
            Synthetic star table.
        """
        trixel = self.htm.triangle(idx)
        circle = trixel.getBoundingCircle()
        area = circle.getArea()

        rng = np.random.default_rng(idx + self.seed)
        nstar = int(rng.poisson(self.density * area))

        h = area / (2 * np.pi)
        z = rng.uniform(1 - h, 1, size=nstar)
        ph = rng.uniform(0, 2 * np.pi, size=nstar)
        rxy = np.sqrt(1 - z**2)
        x = rxy * np.cos(ph)
        y = rxy * np.sin(ph)
        r = np.array([x, y, z])

        k = np.cross([0, 0, 1], circle.getCenter())
        knorm = np.sqrt(np.sum(k**2))
        if knorm != 0:
            k /= knorm
            th = np.pi / 2 - np.arcsin(circle.getCenter()[2])
            sth, cth = np.sin(th), np.cos(th)
            r = cth * r + sth * (np.cross(k, r.T).T) + (1 - cth) * np.dot(k, r) * k[:, None]

        x, y, z = r
        ra = np.arctan2(y, x)
        dec = np.arcsin(z)
        w = trixel.contains(ra, dec)

        table = Table()
        table["ra_rad"] = ra[w]
        table["dec_rad"] = dec[w]

        indices = rng.choice(
            len(self.imag_hist),
            p=self.imag_hist / np.sum(self.imag_hist),
            size=nstar,
        )
        i_mag = self.bins[indices] + rng.uniform(0, 0.1, size=nstar)
        table["i_mag"] = i_mag[w]

        if self.magnitude_hists is not None:
            for gaia_filter, hist in self.magnitude_hists.items():
                mag_hist = hist["values"]
                bins = hist["bins"]

                if len(mag_hist) == 0 or np.sum(mag_hist) == 0:
                    continue

                indices = rng.choice(
                    len(mag_hist),
                    p=mag_hist / np.sum(mag_hist),
                    size=nstar,
                )
                mag = bins[indices] + rng.uniform(0, 0.1, size=nstar)
                table[f"{gaia_filter.lower()}_mag"] = mag[w] + vega_to_ab_offset(gaia_filter)

        table["ra"] = np.rad2deg(table["ra_rad"])
        table["dec"] = np.rad2deg(table["dec_rad"])
        return table


class GaiaStarCatalog(StarCatalog):
    """
    Star catalog backed by local Gaia trixel FITS files.

    Parameters
    ----------
    gaia_dir : str
        Directory containing Gaia HTM trixel FITS files named ``{idx}.fits``.
    level : int, optional
        HTM pixelization level.

    Notes
    -----
    Expected input files should contain columns required for apparent-coordinate
    conversion, including sky position, proper motion, parallax, and fluxes.
    """

    def __init__(self, gaia_dir, level=7):
        super().__init__(level=level)
        self.gaia_dir = gaia_dir

    def _get_trixel_stars(self, idx, time):
        """
        Load Gaia stars from a trixel and convert to apparent coordinates.

        Parameters
        ----------
        idx : int
            HTM trixel index.
        time : astropy.time.Time
            Observation time.

        Returns
        -------
        astropy.table.Table
            Star table with apparent coordinates and ``i_mag``.
        """
        filename = os.path.join(self.gaia_dir, f"{idx}.fits")
        gaia_data = Table.read(filename)

        valid = gaia_data["phot_g_mean_flux"] > 0.0
        gaia_data = gaia_data[valid]

        table = Table()
        table["ra_rad"], table["dec_rad"] = catalog_to_apparent(
            gaia_data["coord_ra"].to(u.rad).value,
            gaia_data["coord_dec"].to(u.rad).value,
            time,
            observer=None,
            pmra=gaia_data["pm_ra"].to(u.mas).value,
            pmdec=gaia_data["pm_dec"].to(u.mas).value,
            parallax=gaia_data["parallax"].to(u.arcsec).value,
        )

        table["i_mag"] = gaia_data["phot_g_mean_flux"].to(u.ABmag).value
        table["ra"] = np.rad2deg(table["ra_rad"])
        table["dec"] = np.rad2deg(table["dec_rad"])
        return table