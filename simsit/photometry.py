"""
Utilities for transforming magnitudes between photometric systems and
querying Gaia source catalogs.

References
----------
Gaia DR2 photometric transformation documentation:
https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


def convert_gaia_magnitude(gaia_g, gaia_bp, gaia_rp, target_filter="2MASS_Ks"):
    """
    Convert Gaia magnitudes to an approximate target-system magnitude.

    Parameters
    ----------
    gaia_g : float or array-like
        Magnitude in the Gaia G passband.
    gaia_bp : float or array-like
        Magnitude in the Gaia BP passband.
    gaia_rp : float or array-like
        Magnitude in the Gaia RP passband.
    target_filter : str, optional
        Target filter name. Supported values are:

        - ``"2MASS_Ks"``
        - ``"2MASS_H"``
        - ``"2MASS_J"``
        - ``"SDSS12_i"``

    Returns
    -------
    float or ndarray
        Approximate magnitude in the requested target filter.

    Notes
    -----
    These are empirical polynomial transforms based on Gaia documentation.
    They should be treated as approximate and used only within the validity
    range of the published relations.
    """
    coeff_dict = {
        "2MASS_Ks": [0.1885, -2.092, 0.1345],
        "2MASS_H": [0.1621, -1.968, 0.1328],
        "2MASS_J": [0.01883, -1.394, 0.07893],
        "SDSS12_i": [0.29676, -0.64728, 0.10141],
    }

    if target_filter not in coeff_dict:
        supported = ", ".join(coeff_dict.keys())
        raise ValueError(f"Unsupported target_filter '{target_filter}'. Supported values: {supported}")

    bp_minus_rp = np.asarray(gaia_bp) - np.asarray(gaia_rp)
    gaia_g = np.asarray(gaia_g)
    c0, c1, c2 = coeff_dict[target_filter]

    return gaia_g + c0 + c1 * bp_minus_rp + c2 * (bp_minus_rp**2)


def spectrum_to_magnitude(*args, **kwargs):
    """
    Convert a spectrum to a magnitude in a specified photometric system.

    Notes
    -----
    This function is not yet implemented.
    """
    raise NotImplementedError("spectrum_to_magnitude is not yet implemented.")


def get_gaia_sources(ra_deg, dec_deg, radius_deg, gaia_data_release=3, star_lim=-1):
    """
    Query Gaia sources within a circular sky region.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees in the ICRS.
    dec_deg : float
        Declination in degrees in the ICRS.
    radius_deg : float
        Search radius in degrees.
    gaia_data_release : int, optional
        Gaia data release to query, typically 2 or 3.
    star_lim : int, optional
        Maximum number of returned sources. Use -1 for no explicit row limit.

    Returns
    -------
    astropy.table.Table
        Query results from the Gaia archive.

    Notes
    -----
    This function performs a remote query through ``astroquery.gaia`` and
    therefore requires network access.
    """
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = star_lim
    Gaia.MAIN_GAIA_TABLE = f"gaiadr{gaia_data_release}.gaia_source"

    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame="icrs")
    job = Gaia.cone_search_async(coord, radius=u.Quantity(radius_deg, u.deg))
    return job.get_results()


def get_gaia_magnitude_histogram(
    ra_deg,
    dec_deg,
    radius_deg,
    gaia_data_release=3,
    star_lim=-1,
    nbins=100,
):
    """
    Compute histograms of Gaia G, BP, and RP magnitudes in a sky region.

    Parameters
    ----------
    ra_deg : float
        Right ascension of the query center in degrees.
    dec_deg : float
        Declination of the query center in degrees.
    radius_deg : float
        Search radius in degrees.
    gaia_data_release : int, optional
        Gaia data release to query.
    star_lim : int, optional
        Maximum number of returned sources. Use -1 for no explicit row limit.
    nbins : int, optional
        Number of histogram bins.

    Returns
    -------
    dict
        Dictionary keyed by ``"G"``, ``"BP"``, and ``"RP"``. Each entry contains
        ``"bins"`` and ``"values"`` arrays.
    """
    if nbins <= 0:
        raise ValueError("nbins must be positive.")

    catalog = get_gaia_sources(
        ra_deg,
        dec_deg,
        radius_deg,
        gaia_data_release=gaia_data_release,
        star_lim=star_lim,
    )

    gaia_filters = ["g", "bp", "rp"]
    results = {}

    for filt in gaia_filters:
        magnitude_dist = np.ma.asarray(
            catalog[f"phot_{filt}_mean_mag"], dtype=float
        ).filled(np.nan)
        magnitude_dist = magnitude_dist[np.isfinite(magnitude_dist)]

        if len(magnitude_dist) == 0:
            results[filt.upper()] = {
                "bins": np.array([]),
                "values": np.array([]),
            }
            continue

        min_mag = np.min(magnitude_dist)
        max_mag = np.max(magnitude_dist)
        if min_mag == max_mag:
            min_mag -= 0.5
            max_mag += 0.5
        bins = np.linspace(min_mag, max_mag, num=nbins + 1)
        values, _ = np.histogram(magnitude_dist, bins=bins)

        results[filt.upper()] = {
            "bins": bins,
            "values": values,
        }

    return results


def vega_to_ab_offset(filter_name):
    """
    Return the additive offset to convert Vega magnitudes to AB magnitudes.

    Parameters
    ----------
    filter_name : str
        Filter name. Supported values include:

        - ``"2MASS_Ks"``
        - ``"2MASS_H"``
        - ``"2MASS_J"``
        - ``"G"``
        - ``"BP"``
        - ``"RP"``

    Returns
    -------
    float
        Magnitude offset to add to a Vega magnitude to obtain an AB magnitude.
    """
    offsets = {
        "2MASS_Ks": 1.85,
        "2MASS_H": 1.39,
        "2MASS_J": 0.91,
        "G": 0.105,
        "BP": 0.0292,
        "RP": 0.3542,
    }

    if filter_name not in offsets:
        supported = ", ".join(offsets.keys())
        raise ValueError(f"Unsupported filter_name '{filter_name}'. Supported values: {supported}")

    return offsets[filter_name]
