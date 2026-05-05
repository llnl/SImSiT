"""
Generate synthetic observations containing stars and satellites.
"""

import glob
import os
import random

import astroplan
import astropy.io.fits as fits
import astropy.units as u
import galsim
import numpy as np
import simsit
import ssapy
import yaml
from astropy.time import Time
from tqdm import tqdm


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

    sky_phot = 10 ** (-0.4 * (sky_sb - zp))
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
        catalog = simsit.MockStarCatalog()
    else:
        catalog = simsit.GaiaStarCatalog(config["catalog"]["gaia_dir"])

    cushion = 0.05 * galsim.degrees
    radius = instrument.field_radius + cushion
    p0 = tracker.get_boresight(t0)
    p1 = tracker.get_boresight(t0 + exptime * u.s)

    stars = catalog.get_stars(p0, p1, radius, t0)
    stars["nphot"] = 10 ** (-0.4 * (stars["i_mag"] - zp)) * exptime

    length = p0.distanceTo(p1) / galsim.arcsec / instrument.pix_size
    stars["SNR"] = instrument.streak_snr(
        nphot=stars["nphot"],
        length=length,
        psf_fwhm=psf_fwhm,
        sky_phot=sky_phot,
    )
    stars = stars[stars["SNR"] > config["catalog"]["min_snr"]]
    print(f"{len(stars) = }")

    sat_nphots = []
    for sat_mag in sat_mags:
        if sat_mag is not None:
            sat_nphots.append(10 ** (-0.4 * (sat_mag - zp)) * exptime)
        else:
            sat_nphots.append(0.0)

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
    )

    sats["i_mag"] = sat_mags
    instrument.apply_vignetting(image)
    instrument.apply_noise(image, sky_phot, rng)

    sample_star_dicts = []
    star_indices = np.argsort(stars["i_mag"])

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
                "mag": float(star["i_mag"] + rng.uniform(0, sigma_mag)),
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
                "mag": float(sat["i_mag"]),
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

    location = observer._location  # Replace with a public accessor if available.
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
    hdu.header["MAGZP"] = zp + rng.uniform(-0.1, 0.1)

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

    for d in [image_dir, reference_dir]:
        path = os.path.join(config["outdir"], d)
        os.makedirs(path, exist_ok=True)

    instrument = simsit.Instrument.from_config(config["instrument"])

    sample_metadata = []
    if "meta" in config:
        sample_metadata.append(config["meta"])

    reference_example_hdul = fits.HDUList()
    reference_hdul = fits.HDUList()
    propagator = ssapy.KeplerianPropagator()

    for i_obs in tqdm(range(config["n_obs"])):
        with simsit.tools.nostdout():
            rng = np.random.default_rng(config["seed"] + i_obs)

            exptime = rng.choice(config["cadence"]["exptime"])
            site = astroplan.Observer.at_site(rng.choice(config["sites"]))

            observer = ssapy.EarthObserver(
                lon=site.location.lon.to(u.deg).value,
                lat=site.location.lat.to(u.deg).value,
                elevation=site.location.height.to(u.m).value,
            )

            cond_cfg = config["conditions"]
            psf_fwhm = rng.uniform(*cond_cfg["psf_fwhm_range"])
            sky_sb = rng.uniform(*cond_cfg["sky_range"])
            zp_offset = rng.uniform(*cond_cfg["zp_range"])

            if "zeropoint" in config["instrument"]:
                zp = config["instrument"]["zeropoint"] + zp_offset
            else:
                zp = instrument.compute_lsst_scaled_zp() + zp_offset

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

            if config["sat"]["mag_range"] is not None:
                sat_mag = rng.uniform(*config["sat"]["mag_range"])
            else:
                sat_mag = None

            if config["n_sat"] == 1:
                orbits = [orbit]
                sat_mags = [sat_mag]

            elif config["n_sat"] == 2:
                ra_offset_arcsec = random.choice((-1, 1)) * psf_fwhm * random.uniform(0.2, 1.5)
                dec_offset_arcsec = random.choice((-1, 1)) * psf_fwhm * random.uniform(0.2, 1.5)

                print(f"{ra_offset_arcsec = } arcsec")
                print(f"{dec_offset_arcsec = } arcsec")

                orbit2 = simsit.tools.generate_orbit(
                    sat_height,
                    galsim.CelestialCoord(
                        ra=(sat_coord.ra.rad + ra_offset_arcsec * 4.84814e-6) * galsim.radians,
                        dec=(sat_coord.dec.rad + dec_offset_arcsec * 4.84814e-6) * galsim.radians,
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
                    mag_offset = random.uniform(mag_bound_1, mag_bound_2)
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
                r_prev, v_prev = ssapy.rv(orbit, t_prev, propagator=propagator)
                v_prev += rng.normal(scale=tracking_error["v_perturb"], size=3)
                tracking_orbit = ssapy.Orbit(r_prev, v_prev, t_prev)
                tracker = simsit.OrbitTracker(
                    orbit=tracking_orbit,
                    observer=observer,
                    t0=t0,
                    rot_sky_pos0=rot_sky_pos0,
                    propagator=propagator,
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
            )

            hdu, hdu_true_wcs_start, hdu_true_wcs_end, image_metadata, reference_hdulist = make_image(
                config,
                rng,
                parameters,
            )

            reference_hdul.append(reference_hdulist[0])
            reference_hdul.append(reference_hdulist[1])

            if i_obs < config["n_example"]:
                reference_example_hdul.append(reference_hdulist[0])
                reference_example_hdul.append(reference_hdulist[1])

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
        yaml.safe_dump_all(sample_metadata[: config["n_example"] + 1], f)

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

    all_files = glob.glob(os.path.join(data_path, "????.fits"))
    files = random.sample(all_files, 1000) if len(all_files) >= 1000 else all_files

    all_images = []
    for filename in tqdm(files, desc="Creating sky flat", leave=True):
        image = fits.getdata(filename)
        image_flattened = image.flatten()
        normalized_image = image_flattened / np.median(image_flattened)
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
