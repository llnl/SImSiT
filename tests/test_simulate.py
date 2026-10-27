from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
import yaml
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.utils import iers
from astropy.table import Table
from astropy.wcs import WCS

import simsit
from simsit.simulate import simulate


class SmallCatalog:
    def get_stars(self, start, end, radius, time):
        return Table({"ra": [start.ra.deg], "dec": [start.dec.deg], "i_mag": [18.0]})


@pytest.mark.parametrize("tracking,n_sat,magnitudes", [
    ("sidereal", 1, [12, 13]),
    ("orbit", 1, None),
    ("sidereal", 2, [12, 13]),
    ("orbit", 2, None),
])
def test_seeded_simulation_and_fits_truth(tmp_path, monkeypatch, tracking, n_sat, magnitudes):
    site = Observer(
        location=EarthLocation.from_geodetic(-106.6 * u.deg, 35.1 * u.deg, 1600 * u.m),
        name="test",
    )
    monkeypatch.setattr(Observer, "at_site", lambda name: site)
    monkeypatch.setattr(simsit, "MockStarCatalog", SmallCatalog)
    monkeypatch.setattr(iers.conf, "auto_download", False)
    config = {
        "n_obs": 2, "n_example": 1, "n_sat": n_sat, "seed": 123,
        "instrument": {
            "image_shape": [48, 48], "gain": 2, "read_noise": 3,
            "quantum_efficiency": 0.8, "pixel_scale": 1, "aperture": 1,
            "distortion": {"th": [0, 1], "dthdr": [1, 1]},
            "zeropoint": 20,
        },
        "conditions": {"psf_fwhm_range": [2, 2], "sky_range": [18, 18], "zp_range": [0, 0]},
        "cadence": {"exptime": [0.1]}, "sites": ["test"],
        "sat": {"mag_range": magnitudes},
        "tracker": {
            "type": tracking, "rot_sky_pos0": 17,
            "error": {"t_rewind": 1, "v_perturb": 0, "boresight": 0},
        },
        "catalog": {"develop": True, "min_snr": 0},
    }
    for name in ("first", "second"):
        current = deepcopy(config)
        current["outdir"] = str(tmp_path / name)
        simulate(current)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for relative in ("images/0001.fits", "images/0002.fits", "reference/reference_truth.fits"):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    with fits.open(first / "images/0001.fits") as hdus:
        assert hdus[0].header["BUNIT"] == "ADU"
        assert hdus[0].header["GAIN"] == 2
        assert np.isfinite(hdus[0].data).all()
    with open(first / "images/sample_metadata_1.yaml") as stream:
        example = list(yaml.safe_load_all(stream))
    assert len(example) == 1
    assert len(example[0]["sats"]) == n_sat
    if magnitudes is None:
        assert all(sat["mag"] is None and sat["flux"] == 0 for sat in example[0]["sats"])
    with fits.open(first / "reference/reference_truth.fits") as hdus:
        hdus.verify("exception")
        truth = hdus["SAT_0001"].data
        for suffix, ra, dec, x, y in (
            ("wcs", "ra0", "dec0", "x0_FITS", "y0_FITS"),
            ("wcst", "ra1", "dec1", "x1_FITS", "y1_FITS"),
        ):
            header = fits.getheader(first / f"reference/0001.{suffix}.fits")
            wcs = WCS(header)
            actual = wcs.all_world2pix(np.column_stack([truth[ra], truth[dec]]), 1)
            np.testing.assert_allclose(actual, np.column_stack([truth[x], truth[y]]), atol=1e-4, rtol=0)
            np.testing.assert_allclose(header["MAGZP"], 20 + 2.5 * np.log10(0.8 / 2))
