import astropy.units as u
import galsim
import numpy as np
import pytest
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import EarthLocation

from simsit import SiderealTracker, OrbitTracker, Instrument, Passband
from simsit import backend
from simsit.brightness import model_from_config
from simsit.tools import draw_sat, generate_orbit
from simsit.simulate import make_image


def test_variable_rate_renderer_matches_analytic_photon_integral(monkeypatch):
    epoch = Time("2025-01-01")
    center = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
    wcs = galsim.TanWCS(
        galsim.AffineTransform(1, 0, 0, 1, origin=galsim.PositionD(0, 0)),
        center, units=galsim.arcsec,
    )

    def coordinates(orbit, times, **kwargs):
        offset = (times - epoch).sec
        ra, dec = wcs.xyToradec(-2 + offset, np.zeros_like(offset), units="radians")
        return ra, dec, np.full(np.shape(offset), 7e5)

    monkeypatch.setattr("simsit.tools.backend.sky_coordinates", coordinates)
    image = galsim.Image(galsim.BoundsI(-20, 20, -20, 20))
    table, _ = draw_sat(
        [object()], t0=epoch, exptime=4, wcs0=wcs,
        tracker=SiderealTracker(center, 0 * galsim.degrees),
        psf=galsim.Gaussian(sigma=1), image=image, observer=object(),
        rate_functions=[lambda t: (200 + 100 * (t - epoch).sec) / u.s],
        ab_magnitude_functions=[lambda rate: -2.5 * np.log10(rate / 1000)],
    )
    expected = 200 * 4 + 100 * 4**2 / 2
    np.testing.assert_allclose(table["nphot"], expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(image.array.sum(), expected, rtol=1e-4)
    np.testing.assert_allclose(
        table["mean_ab_mag"], -2.5 * np.log10(expected / 4 / 1000), atol=1e-10, rtol=0,
    )


def test_photometry_model_selection_is_explicit():
    with pytest.raises(ValueError, match="photometry"):
        model_from_config(None, None)
    assert model_from_config(None, {"model": "constant"}) is None
    with pytest.raises(ValueError, match="parameters"):
        model_from_config(None, {"model": "constant", "params": {"radius_m": 1}})


def test_physical_model_drives_image_truth_not_synthetic_magnitudes(monkeypatch):
    epoch = Time("2025-01-01T12:00:00")
    observer = backend.observer_from_location(EarthLocation.from_geodetic(0*u.deg, 0*u.deg))
    site, _ = observer.getRV(epoch)
    sun = backend.ssatk.get_body("sun").position(epoch.gps)
    direction = np.asarray(sun).reshape(3) - site
    direction /= np.linalg.norm(direction)
    coord = galsim.CelestialCoord.from_xyz(*direction)
    orbit = generate_orbit(500e3, coord, 0, 7800, 0, observer, epoch)
    band = Passband("test", [500, 600], [1, 1], "Synthetic optical test response")
    instrument = Instrument(
        image_shape=[16, 16], gain=2, read_noise=1, pixel_scale=1, aperture=1,
        passband=band, distortion={"th": [0, 1], "dthdr": [1, 1]},
    )
    tracker = OrbitTracker(orbit=orbit, observer=observer, t0=epoch, rot_sky_pos0=0*galsim.degrees)

    class Catalog:
        def __init__(self, passband):
            self.band = passband

        def get_stars(self, start, end, radius, time):
            return Table(
                {"ra": [start.ra.deg], "dec": [start.dec.deg], "mag": [18.]},
                meta={"PASSBAND": self.band.name, "RESPHASH": self.band.response_hash,
                      "MAGSYS": "AB", "MAGCOL": "mag"},
            )

    monkeypatch.setattr("simsit.MockStarCatalog", Catalog)
    exposure = 0.002
    config = {"sat": {"mag_range": None}, "catalog": {"develop": True, "min_snr": 0},
         "photometry": {"model": "lambertian", "params": {
             "radius_m": 0.5, "albedo": 0.3, "include_earthshine": False,
             "include_moonshine": False, "k_extinction": 0,
         }, "spectral_relative_tolerance": 0.005}}
    parameters = {"observer": observer, "t0": epoch, "instrument": instrument, "exptime": exposure,
         "orbit": [orbit], "tracker": tracker, "zp": instrument.photon_zeropoint,
         "sat_mag": [None], "sky_sb": 18, "psf_fwhm": 2, "i_obs": 0,
         "propagator": backend.keplerian_propagator()}
    with pytest.raises(ValueError, match="synthetic"):
        make_image(config, np.random.default_rng(42), {**parameters, "sat_mag": [12.0]})
    output = make_image(config, np.random.default_rng(42), parameters)
    truth = output[4][0].data
    assert truth["nphot"][0] > 0
    expected = -2.5*np.log10(truth["nphot"][0]/exposure/instrument.photon_rate(0).value)
    np.testing.assert_allclose(truth["mag"], expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(truth["mean_ab_mag"], expected, atol=1e-10, rtol=0)
    assert "geometric radiometry" in output[0].header["PHOTO"]
