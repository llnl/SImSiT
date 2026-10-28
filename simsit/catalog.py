"""
Star catalog interfaces for observation simulation.
"""

import os
from abc import ABC, abstractmethod

import astropy.units as u
import lsst.sphgeom as sphgeom
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time
from . import backend

from .photometry import vega_to_ab_offset

__all__ = ["StarCatalog", "MockStarCatalog", "GaiaStarCatalog"]


def _column_to_unit(column, unit, default_unit=None):
    """Return a table column as a floating-point array in ``unit``."""
    column_unit = getattr(column, "unit", None)
    mask = np.ma.getmaskarray(column)
    if column_unit is None:
        if default_unit is None:
            raise ValueError(
                f"Column without a unit requires an explicit {unit} unit."
            )
        values = np.asarray(np.ma.getdata(column), dtype=float)
        converted = (values * default_unit).to_value(unit)
    else:
        values = np.asarray(np.ma.getdata(column.to(unit).value), dtype=float)
        converted = values
    return np.where(mask, np.nan, converted)


def _histogram_sample(rng, values, bins, size):
    """Sample uniformly within histogram intervals."""
    values = np.asarray(values, dtype=float)
    bins = np.asarray(bins, dtype=float)
    if len(bins) != len(values) + 1:
        raise ValueError("Histogram bins must contain one more edge than values.")
    if (
        len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.all(np.isfinite(bins))
        or np.any(np.diff(bins) <= 0)
    ):
        raise ValueError("Histogram values and edges must define finite intervals.")
    indices = rng.choice(len(values), p=values / np.sum(values), size=size)
    return rng.uniform(bins[indices], bins[indices + 1])


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
        self.bins = np.linspace(10, 21, len(self.imag_hist) + 1)
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

        i_mag = _histogram_sample(rng, self.imag_hist, self.bins, nstar)
        table["i_mag"] = i_mag[w]

        if self.magnitude_hists is not None:
            for gaia_filter, hist in self.magnitude_hists.items():
                mag_hist = hist["values"]
                bins = hist["bins"]

                if len(mag_hist) == 0 or np.sum(mag_hist) == 0:
                    continue

                mag = _histogram_sample(rng, mag_hist, bins, nstar)
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
    conversion, including sky position, proper motion, parallax, and
    ``phot_g_mean_mag`` or a physically convertible
    ``phot_g_mean_flux``. Returned magnitudes are AB: Gaia's Vega-system
    G magnitude receives the existing ``G`` Vega-to-AB offset, while a
    physical flux density is converted directly with Astropy. The returned
    ``i_mag`` column is a documented Gaia-G-as-``i_mag`` approximation; it is
    not a Sloan/LSST i-band transformation. Existing LSST trixel files may use
    ``coord_ra``,
    ``coord_dec``, ``pm_ra``, and ``pm_dec``. These legacy LSST columns are
    interpreted as J2000 positions, matching ``catalog_to_apparent``.
    Standard Gaia files may use ``ra``, ``dec``, ``pmra``, and ``pmdec``;
    those files must also provide ``ref_epoch``. Standard Gaia positions are
    rebased from ``ref_epoch`` to J2000 before calling
    ``catalog_to_apparent``.
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

        if "phot_g_mean_mag" in gaia_data.colnames:
            g_mag = np.ma.asarray(
                gaia_data["phot_g_mean_mag"], dtype=float
            ).filled(np.nan)
            # Gaia catalogue magnitudes are Vega magnitudes; the renderer
            # consumes AB magnitudes.
            g_mag += vega_to_ab_offset("G")
        elif "phot_g_mean_flux" in gaia_data.colnames:
            try:
                g_mag = np.ma.asarray(
                    gaia_data["phot_g_mean_flux"].to(u.ABmag).value,
                    dtype=float,
                ).filled(np.nan)
                g_mag = np.where(
                    np.ma.getmaskarray(gaia_data["phot_g_mean_flux"]), np.nan, g_mag,
                )
            except (TypeError, u.UnitConversionError) as exc:
                raise ValueError(
                    "phot_g_mean_flux must be a physical flux density "
                    "convertible to ABmag; electron/s requires a "
                    "release-specific zeropoint."
                ) from exc
        else:
            raise ValueError(
                "Gaia catalog must provide phot_g_mean_mag or a physical "
                "phot_g_mean_flux."
            )

        if "coord_ra" in gaia_data.colnames:
            standard = False
            ra_name, dec_name = "coord_ra", "coord_dec"
            pmra_name, pmdec_name = "pm_ra", "pm_dec"
            coord_default = u.rad
            pm_default = u.mas / u.yr
            parallax_default = u.arcsec
        else:
            standard = True
            if "ref_epoch" not in gaia_data.colnames:
                raise ValueError(
                    "Standard Gaia ra/dec columns require ref_epoch; "
                    "raw Gaia positions must not be assumed to be J2000."
                )
            ra_name, dec_name = "ra", "dec"
            pmra_name, pmdec_name = "pmra", "pmdec"
            coord_default = u.deg
            pm_default = u.mas / u.yr
            parallax_default = u.mas

        ra = _column_to_unit(gaia_data[ra_name], u.rad, coord_default)
        dec = _column_to_unit(gaia_data[dec_name], u.rad, coord_default)
        pmra = _column_to_unit(
            gaia_data[pmra_name], u.mas / u.yr, pm_default
        )
        pmdec = _column_to_unit(
            gaia_data[pmdec_name], u.mas / u.yr, pm_default
        )
        parallax = _column_to_unit(
            gaia_data["parallax"], u.arcsec, parallax_default
        )
        if standard:
            ref_epoch = _column_to_unit(
                gaia_data["ref_epoch"], u.yr, u.yr
            )
        else:
            ref_epoch = None
        valid = np.ones(len(ra), dtype=bool)
        for values in (ra, dec, pmra, pmdec, parallax, g_mag):
            valid &= np.isfinite(values)
        if ref_epoch is not None:
            valid &= np.isfinite(ref_epoch)

        if not np.any(valid):
            return Table(
                {
                    "ra_rad": np.array([], dtype=float),
                    "dec_rad": np.array([], dtype=float),
                    "i_mag": np.array([], dtype=float),
                    "ra": np.array([], dtype=float),
                    "dec": np.array([], dtype=float),
                }
            )

        if standard:
            j2000 = SkyCoord(
                ra=ra[valid] * u.rad,
                dec=dec[valid] * u.rad,
                pm_ra_cosdec=pmra[valid] * u.mas / u.yr,
                pm_dec=pmdec[valid] * u.mas / u.yr,
                obstime=Time(ref_epoch[valid], format="jyear"),
            ).apply_space_motion(new_obstime=Time("J2000"))
            ra = j2000.ra.to_value(u.rad)
            dec = j2000.dec.to_value(u.rad)
            pmra = j2000.pm_ra_cosdec.to_value(u.mas / u.yr)
            pmdec = j2000.pm_dec.to_value(u.mas / u.yr)
        else:
            ra = ra[valid]
            dec = dec[valid]
            pmra = pmra[valid]
            pmdec = pmdec[valid]

        table = Table()
        table["ra_rad"], table["dec_rad"] = backend.catalog_to_apparent(
            ra,
            dec,
            time,
            observer=None,
            pmra=pmra,
            pmdec=pmdec,
            parallax=parallax[valid],
        )

        table["i_mag"] = g_mag[valid]
        table["ra"] = np.rad2deg(table["ra_rad"])
        table["dec"] = np.rad2deg(table["dec_rad"])
        return table
