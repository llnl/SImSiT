"""Independent analytic references for the image-formation primitives."""

from __future__ import annotations

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from scipy.integrate import quad
from scipy.special import erf

import galsim
import numpy as np
import pytest

import simsit.backend as backend
from simsit import Instrument, SiderealTracker
from simsit.tools import draw_sat, draw_stars


PIXEL_SCALE_ARCSEC = 1.0
SOURCE_FLUX = 1000.0
PSF_SIGMA_PIX = 1.0
REFERENCE_GSPARAMS = galsim.GSParams(
    folding_threshold=1e-6,
    maxk_threshold=1e-6,
)


def _simple_tan_wcs():
    """Return a 1 arcsec/pixel TAN WCS with a 101 by 101 image footprint."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 101
    header["NAXIS2"] = 101
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 51.0
    header["CRPIX2"] = 51.0
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    return galsim.GSFitsWCS(header=header)


def _image():
    return galsim.Image(galsim.BoundsI(-50, 50, -50, 50))


def _pixel_cdf_difference(edge, center, sigma=PSF_SIGMA_PIX):
    return 0.5 * (
        erf((edge + 0.5 - center) / (np.sqrt(2.0) * sigma))
        - erf((edge - 0.5 - center) / (np.sqrt(2.0) * sigma))
    )


def _stationary_reference(image, center, flux=SOURCE_FLUX):
    """Analytic Gaussian pixel integral, independent of GalSim drawing."""
    xx, yy = np.meshgrid(
        np.arange(image.bounds.xmin, image.bounds.xmax + 1),
        np.arange(image.bounds.ymin, image.bounds.ymax + 1),
    )
    return flux * _pixel_cdf_difference(xx, center[0]) * _pixel_cdf_difference(
        yy, center[1]
    )


def _path_flux_reference(image, path, flux=lambda _: SOURCE_FLUX):
    """Integrate a Gaussian PSF over each pixel for an arbitrary source path.

    The path and flux callables deliberately form a generic reference for
    curved-path and variable-flux APIs.  The current test uses only a
    straight path and constant flux.
    """
    result = np.zeros_like(image.array, dtype=float)
    for iy, y in enumerate(range(image.bounds.ymin, image.bounds.ymax + 1)):
        for ix, x in enumerate(range(image.bounds.xmin, image.bounds.xmax + 1)):
            def integrand(fraction):
                center_x, center_y = path(fraction)
                return (
                    flux(fraction)
                    * _pixel_cdf_difference(x, center_x)
                    * _pixel_cdf_difference(y, center_y)
                )

            result[iy, ix] = quad(
                integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10
            )[0]
    return result


def _pixel_position_to_radec(wcs, x, y):
    coord = wcs.posToWorld(galsim.PositionD(x, y))
    return coord.ra.deg, coord.dec.deg


def test_stationary_gaussian_matches_analytic_pixel_integrals():
    wcs = _simple_tan_wcs()
    center = (0.23, -0.31)
    ra, dec = _pixel_position_to_radec(wcs, *center)
    image = _image()

    draw_stars(
        Table({"ra": [ra], "dec": [dec], "nphot": [SOURCE_FLUX]}),
        t0=Time("2020-01-01"),
        exptime=1.0,
        wcs0=wcs,
        tracker=SiderealTracker(wcs.center, 0 * galsim.degrees),
        psf=galsim.Gaussian(sigma=PSF_SIGMA_PIX, gsparams=REFERENCE_GSPARAMS),
        image=image,
        nsplit=1,
    )

    reference = _stationary_reference(image, center)
    residual = image.array - reference
    assert image.array.sum() == pytest.approx(SOURCE_FLUX, abs=3e-3)
    assert np.max(np.abs(residual)) < 2e-4
    assert np.sum(np.abs(residual)) / SOURCE_FLUX < 1e-5


def test_constant_velocity_trail_matches_quad_pixel_integrals(monkeypatch):
    wcs = _simple_tan_wcs()
    start = (-8.2, -5.3)
    end = (9.1, 7.4)
    start_radec = _pixel_position_to_radec(wcs, *start)
    end_radec = _pixel_position_to_radec(wcs, *end)

    def synthetic_sky_coordinates(_orbit, times, **_kwargs):
        fractions = np.linspace(0.0, 1.0, len(times))
        ra = np.deg2rad(
            np.interp(fractions, [0.0, 1.0], [start_radec[0], end_radec[0]])
        )
        dec = np.deg2rad(
            np.interp(fractions, [0.0, 1.0], [start_radec[1], end_radec[1]])
        )
        return ra, dec, None

    monkeypatch.setattr(backend, "sky_coordinates", synthetic_sky_coordinates)
    image = _image()

    class SyntheticOrbit:
        pass

    draw_sat(
        [SyntheticOrbit()],
        t0=Time("2020-01-01"),
        exptime=1.0,
        wcs0=wcs,
        tracker=SiderealTracker(wcs.center, 0 * galsim.degrees),
        psf=galsim.Gaussian(sigma=PSF_SIGMA_PIX, gsparams=REFERENCE_GSPARAMS),
        image=image,
        observer=None,
        nphot=[SOURCE_FLUX],
        nsplit=1,
        integration=False,
    )

    reference = _path_flux_reference(
        image,
        lambda fraction: (
            start[0] + fraction * (end[0] - start[0]),
            start[1] + fraction * (end[1] - start[1]),
        ),
    )
    residual = image.array - reference
    assert image.array.sum() == pytest.approx(SOURCE_FLUX, abs=3e-3)
    assert np.max(np.abs(residual)) < 3e-5
    assert np.sum(np.abs(residual)) / SOURCE_FLUX < 1e-5


def test_adaptive_curved_variable_flux_converges_to_quad_reference(monkeypatch):
    """Check final adaptive rendering against time-integrated pixel CDFs."""
    wcs = _simple_tan_wcs()
    epoch = Time("2025-01-01")
    duration = 4.0
    start = np.array([-7.0, -3.5])

    def path(fraction):
        return (
            start[0] + 10.0 * fraction + 2.4 * fraction**2,
            start[1] + 5.0 * fraction - 1.6 * fraction**2,
        )

    def rate(fraction):
        return SOURCE_FLUX * (
            1.0 + 0.25 * fraction + 0.18 * np.sin(2.0 * np.pi * fraction)
        )

    def synthetic_sky_coordinates(_orbit, time, **_kwargs):
        fraction = (time - epoch).to_value(u.s) / duration
        ra, dec = _pixel_position_to_radec(wcs, *path(fraction))
        return ra * np.pi / 180.0, dec * np.pi / 180.0, 7.0e5

    monkeypatch.setattr(backend, "sky_coordinates", synthetic_sky_coordinates)

    def render(settings):
        image = galsim.Image(galsim.BoundsI(-20, 20, -20, 20))
        class SyntheticOrbit:
            pass

        table, _ = draw_sat(
            [SyntheticOrbit()],
            t0=epoch,
            exptime=duration,
            wcs0=wcs,
            tracker=SiderealTracker(wcs.center, 0 * galsim.degrees),
            psf=galsim.Gaussian(sigma=PSF_SIGMA_PIX, gsparams=REFERENCE_GSPARAMS),
            image=image,
            observer=None,
            rate_functions=[
                lambda time: rate((time - epoch).to_value(u.s) / duration)
            ],
            integration=settings,
        )
        return image.array.copy(), table

    reference_image = galsim.Image(galsim.BoundsI(-20, 20, -20, 20))
    reference = duration * _path_flux_reference(
        reference_image,
        path,
        flux=rate,
    )
    loose, loose_table = render({
        "pixel_tolerance": 0.35,
        "flux_relative_tolerance": 2e-3,
        "max_step_s": duration / 2,
    })
    tight, tight_table = render({
        "pixel_tolerance": 0.01,
        "flux_relative_tolerance": 1e-7,
        "max_step_s": duration / 8,
        "max_depth": 14,
        "max_samples": 4097,
    })

    yy, xx = np.indices(reference.shape)
    xx = xx + -20
    yy = yy + -20
    reference_flux = reference.sum()
    reference_centroid = np.array([
        np.sum(xx * reference) / reference_flux,
        np.sum(yy * reference) / reference_flux,
    ])

    def diagnostics(image):
        flux = image.sum()
        centroid = np.array([
            np.sum(xx * image) / flux,
            np.sum(yy * image) / flux,
        ])
        return flux, centroid, np.linalg.norm(image - reference) / reference_flux

    loose_flux, loose_centroid, loose_norm = diagnostics(loose)
    tight_flux, tight_centroid, tight_norm = diagnostics(tight)
    assert tight_table["integration_samples"][0] > loose_table["integration_samples"][0]
    assert tight_norm < loose_norm
    assert tight_norm < 1e-4
    assert tight_norm < 0.1 * loose_norm
    assert tight_flux == pytest.approx(reference_flux, rel=2e-4)
    assert np.linalg.norm(tight_centroid - reference_centroid) < 2e-3
    assert loose_flux == pytest.approx(reference_flux, rel=3e-3)
    assert np.linalg.norm(loose_centroid - reference_centroid) < 0.03


def test_ccd_noise_matches_closed_form_adu_moments():
    gain = 2.4
    quantum_efficiency = 0.73
    read_noise = 3.2
    detector = Instrument(
        image_shape=(512, 512),
        gain=gain,
        read_noise=read_noise,
        quantum_efficiency=quantum_efficiency,
        pixel_scale=PIXEL_SCALE_ARCSEC,
        aperture=1.0,
    )
    yy, xx = np.indices((512, 512))
    photons = 80.0 + 0.02 * xx + 0.03 * yy
    image = galsim.Image(photons.astype(float))
    detector.apply_noise(image, sky_phot=0.0, rng=np.random.default_rng(20261027))

    expected_mean = quantum_efficiency * photons.mean() / gain
    expected_variance = (
        (quantum_efficiency * photons).mean() + read_noise**2
    ) / gain**2 + (quantum_efficiency * photons / gain).var()
    np.testing.assert_allclose(image.array.mean(), expected_mean, rtol=3e-3)
    np.testing.assert_allclose(image.array.var(), expected_variance, rtol=8e-3)
