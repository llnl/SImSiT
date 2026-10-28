"""
Generate synthetic observations containing stars and satellites.
"""

import glob
import os

import astroplan
import astropy.io.fits as fits
import astropy.units as u
import galsim
import numpy as np
import simsit
import yaml
from astropy.time import Time
from tqdm import tqdm
from .tracker import transform_wcs as _transform_wcs
from . import backend
from .brightness import model_from_config as _model_from_config


def make_image(config, rng, parameters):
    """
    Generate a single simulated image with stars and satellites.

    Parameters
    ----------
    config : dict
        Simulation configuration dictionary.
    rng : numpy.random.Generator
        Random number generator.
    parameters : dict
        Observation parameters.

    Returns
    -------
    hdu : astropy.io.fits.PrimaryHDU
        Simulated image with perturbed WCS metadata.
    hdu_true_wcs_start : astropy.io.fits.PrimaryHDU
        Reference WCS metadata at exposure start.
    hdu_true_wcs_end : astropy.io.fits.PrimaryHDU
        Reference WCS metadata at exposure end.
    image_metadata : dict
        Example metadata record for the generated image.
    reference_hdulist : list
        List containing reference catalog HDUs.
    """
    observer = parameters["observer"]
    t0 = parameters["t0"]
    instrument = parameters["instrument"]
    exptime = parameters["exptime"]
    orbits = parameters["orbit"]
    tracker = parameters["tracker"]
    zp = parameters["zp"]
    sat_mags = parameters["sat_mag"]
    sky_sb = parameters["sky_sb"]
    psf_fwhm = parameters["psf_fwhm"]
    i_obs = parameters["i_obs"]
    propagator = parameters["propagator"]
    observation_mode = backend.observation_mode(
        parameters.get("observation_mode", "geometric")
    )
    photometry = config.get("photometry")
    if not isinstance(photometry, dict) or photometry.get("model") not in (
        "constant", "lambertian", "faceted",
    ):
        raise ValueError("photometry.model must be 'constant', 'lambertian', or 'faceted'.")
    if photometry["model"] == "constant":
        photo_description = "constant synthetic AB magnitude"
    else:
        photo_description = (
            f"SSATK gray {photometry['model']} geometric radiometry at arrival epoch; "
            "apparent mode only changes direction/light-time"
        )
    integration_config = config.get("integration")
    if integration_config is False:
        integration_metadata = {}
        integration_mode = "fixed"
    else:
        integration_metadata = dict(integration_config or {})
        integration_mode = "adaptive"

    if instrument.passband is None:
        raise ValueError("Image generation requires an explicit named passband response.")
    brightness = _model_from_config(instrument, photometry)
    if photometry["model"] != "constant" and (
        config["sat"].get("mag_range") is not None or any(mag is not None for mag in sat_mags)
    ):
        raise ValueError("Physical photometry cannot also use synthetic magnitudes.")
    zp_offset = zp - instrument.photon_zeropoint
    sky_phot = instrument.photon_rate(sky_sb, zeropoint_offset=zp_offset).to_value(1 / u.s)
    psf = galsim.Kolmogorov(
        fwhm=psf_fwhm,
        gsparams=galsim.GSParams(folding_threshold=1e-3),
    )

    boresight0 = tracker.get_boresight(t0)
    rot_sky_pos0 = tracker.get_rot_sky_pos(t0)
    wcs0 = instrument.get_wcs(boresight=boresight0, rot_sky_pos=rot_sky_pos0)

    print(f"{t0.iso = }")
    print(f"{boresight0.ra = }")
    print(f"{boresight0.dec = }")
    print(f"{rot_sky_pos0 = }")
    for orbit in orbits:
        print(f"{orbit.period = } s")
        print(f"{orbit.e = }")
        print(f"{orbit.i = } rad")

    if config["catalog"].get("develop", False):
        catalog = simsit.MockStarCatalog(passband=instrument.passband)
    else:
        catalog_config = config["catalog"]
        catalog = simsit.GaiaStarCatalog(
            catalog_config["gaia_dir"], passband=instrument.passband,
            magnitude_column=catalog_config.get("magnitude_column"),
            magnitude_system=catalog_config.get("magnitude_system", "AB"),
        )

    cushion = 0.05 * galsim.degrees
    radius = instrument.field_radius + cushion
    p0 = tracker.get_boresight(t0)
    p1 = tracker.get_boresight(t0 + exptime * u.s)

    stars = catalog.get_stars(p0, p1, radius, t0)
    if (
        stars.meta.get("PASSBAND") != instrument.passband.name
        or stars.meta.get("RESPHASH") != instrument.passband.response_hash
        or stars.meta.get("MAGSYS") != "AB"
        or stars.meta.get("MAGCOL") != "mag"
    ):
        raise ValueError("Star magnitudes must be AB and calibrated in the instrument response.")
    stars["nphot"] = instrument.photon_rate(
        np.asarray(stars["mag"]), zeropoint_offset=zp_offset,
    ).to_value(1 / u.s) * exptime

    length = p0.distanceTo(p1) / galsim.arcsec / instrument.pix_size
    stars["SNR"] = instrument.streak_snr(
        nphot=stars["nphot"],
        length=length,
        psf_fwhm=psf_fwhm,
        sky_phot=sky_phot,
        exptime=exptime,
    )
    stars = stars[stars["SNR"] > config["catalog"]["min_snr"]]
    print(f"{len(stars) = }")

    sat_nphots = None
    rate_functions = None
    ab_magnitude_functions = None
    if photometry["model"] == "constant":
        sat_nphots = []
        for sat_mag in sat_mags:
            if sat_mag is not None:
                sat_nphots.append(
                    float(
                        instrument.photon_rate(
                            sat_mag, zeropoint_offset=zp_offset,
                        ).to_value(1 / u.s)
                    ) * exptime
                )
            else:
                sat_nphots.append(0.0)
    else:
        calibration_ratio = (
            instrument.photon_scale(zp_offset) / instrument.photon_scale(0.0)
        )
        zero_rate = instrument.photon_rate(
            0.0, zeropoint_offset=zp_offset,
        ).to_value(1 / u.s)
        rate_functions = []
        ab_magnitude_functions = []
        for orbit in orbits:
            def rate(time, orbit=orbit):
                position, _ = backend.states(
                    orbit, time, propagator=propagator,
                )
                return brightness.photon_rate(
                    position, time, observer,
                ) * calibration_ratio

            def mean_ab_mag(rate_value, zero_rate=zero_rate):
                if rate_value <= 0:
                    return np.inf
                return -2.5 * np.log10(rate_value / zero_rate)

            rate_functions.append(rate)
            ab_magnitude_functions.append(mean_ab_mag)

    print(f"{sat_mags = }")
    print(f"{sat_nphots = }")

    image = instrument.init_image(sky_phot=sky_phot, exptime=exptime)
    stars = simsit.tools.draw_stars(
        stars,
        t0=t0,
        exptime=exptime,
        wcs0=wcs0,
        tracker=tracker,
        psf=psf,
        image=image,
        integration=config.get("integration"),
    )
    sats, wcst = simsit.tools.draw_sat(
        orbits,
        t0=t0,
        exptime=exptime,
        wcs0=wcs0,
        tracker=tracker,
        psf=psf,
        image=image,
        observer=observer,
        nphot=sat_nphots,
        propagator=propagator,
        observation_mode=observation_mode,
        rate_functions=rate_functions,
        ab_magnitude_functions=ab_magnitude_functions,
        integration=config.get("integration"),
    )

    if wcst is None:
        wcst = _transform_wcs(wcs0, p1, tracker.get_rot_sky_pos(t0 + exptime * u.s))
    if photometry["model"] == "constant":
        sats["mean_ab_mag"] = [
            np.nan if mag is None else mag for mag in sat_mags
        ]
    sats["mag"] = np.asarray(sats["mean_ab_mag"])
    instrument.apply_vignetting(image)
    instrument.apply_noise(image, sky_phot, rng)

    sample_star_dicts = []
    star_indices = np.argsort(stars["mag"])

    sigma_pix = 5.0
    sigma_sky_deg = 5.0 / 3600.0
    sigma_mag = 0.1

    for i in star_indices[:10]:
        star = stars[i]
        sample_star_dicts.append(
            {
                "x": float(star["x_FITS"] + rng.uniform(0, sigma_pix)),
                "y": float(star["y_FITS"] + rng.uniform(0, sigma_pix)),
                "flux": float(star["nphot"] * rng.uniform(1.0, 1.1)),
                "ra": float((star["ra"] + rng.uniform(0, sigma_sky_deg)) % 360.0),
                "dec": float(star["dec"] + rng.uniform(0, sigma_sky_deg)),
                "mag": float(star["mag"] + rng.uniform(0, sigma_mag)),
            }
        )

    sample_sat_dicts = []
    for sat in sats:
        sample_sat_dicts.append(
            {
                "x0": float(sat["x0_FITS"] + rng.uniform(0, sigma_pix)),
                "y0": float(sat["y0_FITS"] + rng.uniform(0, sigma_pix)),
                "x1": float(sat["x1_FITS"] + rng.uniform(0, sigma_pix)),
                "y1": float(sat["y1_FITS"] + rng.uniform(0, sigma_pix)),
                "flux": float(sat["nphot"] * rng.uniform(1.0, 1.1)),
                "ra0": float((sat["ra0"] + rng.uniform(0, sigma_sky_deg)) % 360.0),
                "dec0": float(sat["dec0"] + rng.uniform(0, sigma_sky_deg)),
                "ra1": float((sat["ra1"] + rng.uniform(0, sigma_sky_deg)) % 360.0),
                "dec1": float(sat["dec1"] + rng.uniform(0, sigma_sky_deg)),
                "mag": None if not np.isfinite(sat["mag"]) else float(sat["mag"]),
                "mean_ab_mag": (
                    None if not np.isfinite(sat["mean_ab_mag"])
                    else float(sat["mean_ab_mag"])
                ),
            }
        )

    image_metadata = {
        "file": f"{i_obs + 1:04d}.fits",
        "sats": sample_sat_dicts,
        "stars": sample_star_dicts,
    }

    star_hdu = fits.table_to_hdu(stars)
    star_hdu.name = f"STAR_{i_obs + 1:04d}"
    sat_hdu = fits.table_to_hdu(sats)
    sat_hdu.name = f"SAT_{i_obs + 1:04d}"
    reference_hdulist = [sat_hdu, star_hdu]

    hdu_true_wcs_start = fits.PrimaryHDU()
    hdu_true_wcs_end = fits.PrimaryHDU()

    for hdu_wcs in (hdu_true_wcs_start, hdu_true_wcs_end):
        hdu_wcs.header["TIMESYS"] = "TAI"
        hdu_wcs.header["DATE-BEG"] = t0.tai.isot
        hdu_wcs.header["DATE-END"] = (t0 + exptime * u.s).tai.isot
        hdu_wcs.header["TELAPSE"] = exptime
        hdu_wcs.header["TIMEUNIT"] = "s"

    location = backend.observer_location(observer)
    obsx = location.x.to(u.m).value
    obsy = location.y.to(u.m).value
    obsz = location.z.to(u.m).value

    for hdu_wcs in (hdu_true_wcs_start, hdu_true_wcs_end):
        hdu_wcs.header["OBSGEO-X"] = (obsx, "ITRS (m)")
        hdu_wcs.header["OBSGEO-Y"] = (obsy, "ITRS (m)")
        hdu_wcs.header["OBSGEO-Z"] = (obsz, "ITRS (m)")

    wcs0.writeToFitsHeader(hdu_true_wcs_start.header, image.bounds)
    wcst.writeToFitsHeader(hdu_true_wcs_end.header, image.bounds)

    hdu = fits.PrimaryHDU(image.array)
    hdu.header["TIMESYS"] = "TAI"
    hdu.header["DATE-BEG"] = t0.tai.isot
    hdu.header["DATE-END"] = (t0 + exptime * u.s).tai.isot
    hdu.header["TELAPSE"] = exptime
    hdu.header["TIMEUNIT"] = "s"
    hdu.header["OBSGEO-X"] = (obsx, "ITRS (m)")
    hdu.header["OBSGEO-Y"] = (obsy, "ITRS (m)")
    hdu.header["OBSGEO-Z"] = (obsz, "ITRS (m)")
    adu_zp = zp + 2.5 * np.log10(instrument.quantum_efficiency / instrument.gain)
    hdu.header["MAGZP"] = adu_zp + rng.uniform(-0.1, 0.1)
    for header in (hdu.header, hdu_true_wcs_start.header, hdu_true_wcs_end.header):
        header["PHOTZP"] = (zp, "AB magnitude for one incident photon/sec")
        header["GAIN"] = (instrument.gain, "electrons/ADU")
        header["RDNOISE"] = (instrument.read_noise, "electrons")
        header["QE"] = (instrument.quantum_efficiency, "electrons/photon")
        header["EXPTIME"] = (exptime, "seconds")
        header["MAGSYS"] = "AB"
        header["FRAME"] = "GCRF"
        header["OBSMODE"] = observation_mode
        header["FILTER"] = instrument.passband.name
        header["RESPHASH"] = instrument.passband.response_hash
        header["BANDREF"] = instrument.passband.reference
        header["ZPREF"] = (
            instrument.zeropoint_reference
            or "Derived from response and collecting area"
        )
        header["PHOTO"] = photo_description
        header["INTMODE"] = integration_mode
        if integration_mode == "adaptive":
            header["PIXTOL"] = integration_metadata.get("pixel_tolerance", 0.01)
            header["FLXTOL"] = integration_metadata.get("flux_relative_tolerance", 1e-4)
    hdu.header["BUNIT"] = "ADU"
    hdu_true_wcs_start.header["MAGZP"] = adu_zp
    hdu_true_wcs_end.header["MAGZP"] = adu_zp

    htmp = hdu_true_wcs_start.header
    crpix = htmp["CRPIX1"], htmp["CRPIX2"]
    cd = np.array([
        [htmp["CD1_1"], htmp["CD1_2"]],
        [htmp["CD2_1"], htmp["CD2_2"]],
    ])
    crval1, crval2 = htmp["CRVAL1"], htmp["CRVAL2"]

    shift_scale_deg = 1.0 / 60.0
    crval1 += rng.uniform(-0.25, 0.25) * shift_scale_deg
    crval2 += rng.uniform(-0.25, 0.25) * shift_scale_deg
    rot_shift = np.deg2rad(rng.uniform(-0.5, 0.5))
    cr, sr = np.cos(rot_shift), np.sin(rot_shift)
    cd = cd @ np.array([[cr, sr], [-sr, cr]])

    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRPIX1"], hdu.header["CRPIX2"] = crpix
    hdu.header["CD1_1"] = cd[0, 0]
    hdu.header["CD1_2"] = cd[0, 1]
    hdu.header["CD2_1"] = cd[1, 0]
    hdu.header["CD2_2"] = cd[1, 1]
    hdu.header["CUNIT1"] = "deg"
    hdu.header["CUNIT2"] = "deg"
    hdu.header["CRVAL1"] = crval1
    hdu.header["CRVAL2"] = crval2

    return hdu, hdu_true_wcs_start, hdu_true_wcs_end, image_metadata, reference_hdulist


def simulate(config):
    """
    Run the full simulation and write outputs to disk.

    Parameters
    ----------
    config : dict
        Simulation configuration dictionary.

    Notes
    -----
    Output directories default to ``images`` and ``reference`` but may be
    overridden with ``image_dir`` and ``reference_dir`` in the config.
    """
    image_dir = config.get("image_dir", "images")
    reference_dir = config.get("reference_dir", "reference")
    instrument = simsit.Instrument.from_config(config["instrument"])
    if instrument.passband is None:
        raise ValueError("Simulation requires an explicit named passband response.")
    photometry = config.get("photometry")
    _model_from_config(instrument, photometry)
    if photometry["model"] != "constant" and config["sat"].get("mag_range") is not None:
        raise ValueError("Physical photometry cannot also use synthetic mag_range.")

    for d in [image_dir, reference_dir]:
        path = os.path.join(config["outdir"], d)
        os.makedirs(path, exist_ok=True)

    sample_metadata = []
    example_metadata = []
    if "meta" in config:
        sample_metadata.append(config["meta"])
        example_metadata.append(config["meta"])

    reference_example_hdul = fits.HDUList([fits.PrimaryHDU()])
    reference_hdul = fits.HDUList([fits.PrimaryHDU()])
    propagator = backend.keplerian_propagator()
    observation_mode = backend.observation_mode(
        config.get("observation", {}).get("mode", "geometric")
    )

    for i_obs in tqdm(range(config["n_obs"])):
        with simsit.tools.nostdout():
            rng = np.random.default_rng(config["seed"] + i_obs)

            exptime = rng.choice(config["cadence"]["exptime"])
            site = astroplan.Observer.at_site(rng.choice(config["sites"]))

            observer = backend.observer_from_location(site.location)

            cond_cfg = config["conditions"]
            psf_fwhm = rng.uniform(*cond_cfg["psf_fwhm_range"])
            sky_sb = rng.uniform(*cond_cfg["sky_range"])
            zp_offset = rng.uniform(*cond_cfg["zp_range"])

            instrument.photon_scale(zp_offset)
            zp = instrument.photon_zeropoint + zp_offset

            print()
            print(f"{exptime = :.2f} s")
            print(f"{site.name = }")
            print(f"{psf_fwhm = :.2f} arcsec")
            print(f"{sky_sb = :.2f} mag / arcsec^2")
            print(f"{zp_offset = :.2f} mag")
            print(f"{zp = :.2f} mag")

            t_day = Time("2010-01-01") + rng.uniform(0, 365) * u.d
            t0 = simsit.tools.random_dark_time(t_day=t_day, site=site, rng=rng)

            boresight0 = simsit.tools.random_boresight(
                observer=observer,
                t0=t0,
                horizon=20 * galsim.degrees,
                rng=rng,
            )

            sat_height = rng.uniform(400e3, 800e3)
            sat_coord = simsit.tools.random_disk(
                boresight0,
                0.2 * instrument.field_radius,
                rng,
            )
            heading = rng.uniform(0, 2 * np.pi)
            vperp = rng.normal(7800, 10)
            vpar = rng.normal(10)

            tmid = t0 + 0.5 * exptime * u.s
            orbit = simsit.tools.generate_orbit(
                sat_height,
                sat_coord,
                heading,
                vperp,
                vpar,
                observer,
                tmid,
            )

            photometry = config.get("photometry")
            if not isinstance(photometry, dict) or photometry.get("model") not in (
                "constant", "lambertian", "faceted",
            ):
                raise ValueError(
                    "photometry.model must be 'constant', 'lambertian', or 'faceted'."
                )
            if photometry["model"] == "constant" and config["sat"]["mag_range"] is not None:
                sat_mag = rng.uniform(*config["sat"]["mag_range"])
            else:
                sat_mag = None

            if config["n_sat"] == 1:
                orbits = [orbit]
                sat_mags = [sat_mag]

            elif config["n_sat"] == 2:
                ra_offset_arcsec = rng.choice((-1, 1)) * psf_fwhm * rng.uniform(0.2, 1.5)
                dec_offset_arcsec = rng.choice((-1, 1)) * psf_fwhm * rng.uniform(0.2, 1.5)

                print(f"{ra_offset_arcsec = } arcsec")
                print(f"{dec_offset_arcsec = } arcsec")

                orbit2 = simsit.tools.generate_orbit(
                    sat_height,
                    sat_coord.deproject(
                        -ra_offset_arcsec * galsim.arcsec,
                        dec_offset_arcsec * galsim.arcsec,
                        projection="gnomonic",
                    ),
                    heading,
                    vperp,
                    vpar,
                    observer,
                    tmid,
                )

                if sat_mag is None:
                    sat_mag2 = None
                else:
                    mag_min, mag_max = config["sat"]["mag_range"]
                    mag_bound_1 = max(-1.5, mag_min - sat_mag)
                    mag_bound_2 = min(1.5, mag_max - sat_mag)
                    mag_offset = rng.uniform(mag_bound_1, mag_bound_2)
                    sat_mag2 = sat_mag + mag_offset
                    print(f"{mag_offset = } mag")

                orbits = [orbit, orbit2]
                sat_mags = [sat_mag, sat_mag2]

            else:
                raise ValueError("Only n_sat = 1 or n_sat = 2 are currently supported.")

            tracker_cfg = config["tracker"]
            tracking_error = tracker_cfg["error"]
            rot_sky_pos0 = tracker_cfg["rot_sky_pos0"] * galsim.degrees

            if tracker_cfg["type"] == "orbit":
                t_prev = t0 - tracking_error["t_rewind"] * u.min
                r_prev, v_prev = backend.states(orbit, t_prev, propagator=propagator)
                v_prev += rng.normal(scale=tracking_error["v_perturb"], size=3)
                tracking_orbit = backend.orbit_from_state(r_prev, v_prev, t_prev)
                tracker = simsit.OrbitTracker(
                    orbit=tracking_orbit,
                    observer=observer,
                    t0=t0,
                    rot_sky_pos0=rot_sky_pos0,
                    propagator=propagator,
                    observation_mode=observation_mode,
                )

            elif tracker_cfg["type"] == "sidereal":
                tracking_boresight = simsit.tools.random_disk(
                    boresight0,
                    instrument.field_radius * tracking_error["boresight"],
                    rng,
                )
                tracker = simsit.SiderealTracker(tracking_boresight, rot_sky_pos0)

            else:
                raise ValueError(f"Unsupported tracker type: {tracker_cfg['type']}")

            parameters = dict(
                orbit=orbits,
                observer=observer,
                t0=t0,
                instrument=instrument,
                exptime=exptime,
                tracker=tracker,
                sat_mag=sat_mags,
                sky_sb=sky_sb,
                psf_fwhm=psf_fwhm,
                zp=zp,
                i_obs=i_obs,
                propagator=propagator,
                observation_mode=observation_mode,
            )

            hdu, hdu_true_wcs_start, hdu_true_wcs_end, image_metadata, reference_hdulist = (
                make_image(
                    config,
                    rng,
                    parameters,
                )
            )

            reference_hdul.append(reference_hdulist[0])
            reference_hdul.append(reference_hdulist[1])

            if i_obs < config["n_example"]:
                reference_example_hdul.append(reference_hdulist[0])
                reference_example_hdul.append(reference_hdulist[1])
                example_metadata.append(image_metadata)

            sample_metadata.append(image_metadata)

            hdu.writeto(
                os.path.join(config["outdir"], image_dir, f"{i_obs + 1:04d}.fits"),
                overwrite=True,
            )
            hdu_true_wcs_start.writeto(
                os.path.join(config["outdir"], reference_dir, f"{i_obs + 1:04d}.wcs.fits"),
                overwrite=True,
            )
            hdu_true_wcs_end.writeto(
                os.path.join(config["outdir"], reference_dir, f"{i_obs + 1:04d}.wcst.fits"),
                overwrite=True,
            )

    with open(os.path.join(config["outdir"], reference_dir, "sample_metadata.yaml"), "w") as f:
        yaml.safe_dump_all(sample_metadata, f)

    example_metadata_filename = os.path.join(
        config["outdir"],
        image_dir,
        f"sample_metadata_{config['n_example']}.yaml",
    )
    with open(example_metadata_filename, "w") as f:
        yaml.safe_dump_all(example_metadata, f)

    reference_hdul.writeto(
        os.path.join(config["outdir"], reference_dir, "reference_truth.fits"),
        overwrite=True,
    )
    reference_example_hdul.writeto(
        os.path.join(config["outdir"], image_dir, f"reference_truth_{config['n_example']}.fits"),
        overwrite=True,
    )


def make_sky_flat(configfile):
    """
    Generate a median sky flat from simulated images.

    Parameters
    ----------
    configfile : str
        Path to a YAML configuration file.
    """
    with open(configfile, "r") as f:
        config = yaml.safe_load(f)

    image_dir = config.get("image_dir", "images")
    data_path = os.path.join(config["outdir"], image_dir)

    all_files = sorted(glob.glob(os.path.join(data_path, "????.fits")))
    if not all_files:
        raise ValueError(f"No simulated images found in {data_path}.")
    rng = np.random.default_rng(config["seed"])
    files = rng.choice(all_files, 1000, replace=False) if len(all_files) > 1000 else all_files

    all_images = []
    for filename in tqdm(files, desc="Creating sky flat", leave=True):
        image = fits.getdata(filename)
        image_flattened = image.flatten()
        median = np.median(image_flattened)
        if not np.isfinite(image_flattened).all() or not np.isfinite(median) or median <= 0:
            raise ValueError(f"Sky-flat input must be finite with a positive median: {filename}")
        normalized_image = image_flattened / median
        all_images.append(normalized_image)

    median_image = np.median(np.vstack(all_images), axis=0)
    final_image = median_image / np.mean(median_image)
    sky_flat = np.reshape(
        final_image,
        (
            config["instrument"]["image_shape"][1],
            config["instrument"]["image_shape"][0],
        ),
    )

    hdu = fits.PrimaryHDU(sky_flat)
    hdu.writeto(os.path.join(config["outdir"], image_dir, "sky_flat.fits"), overwrite=True)
