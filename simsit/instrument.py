"""
Telescope and detector instrument models for observation simulation.
"""

import galsim
import numpy as np

from .wcs import radial_wcs


class Instrument:
    """
    Instrument model describing detector geometry, noise, and optical response.

    Parameters
    ----------
    image_shape : tuple of int
        Image dimensions ``(nx, ny)`` in pixels.
    gain : float
        Detector gain in electrons per ADU.
    read_noise : float
        Read noise in electrons.
    pixel_scale : float
        Physical pixel size in microns.
    aperture : float
        Telescope aperture diameter in meters.
    obscuration : float, optional
        Linear fractional obscuration.
    distortion : dict, optional
        Optical distortion model with keys:

        - ``"th"``: field angle in degrees
        - ``"dthdr"``: plate scale in arcsec/micron
    vignetting : dict, optional
        Vignetting model with keys:

        - ``"th"``: field angle in degrees
        - ``"unvig"``: surviving fraction of photons
    """

    def __init__(
        self,
        *,
        image_shape,
        gain,
        read_noise,
        pixel_scale,
        aperture,
        obscuration=0.0,
        distortion=None,
        vignetting=None,
    ):
        self.image_shape = image_shape
        self.gain = gain
        self.read_noise = read_noise
        self.pixel_scale = pixel_scale
        self.aperture = aperture
        self.obscuration = obscuration
        self.distortion = distortion
        self.vignetting = vignetting

        if vignetting is not None:
            from scipy.interpolate import interp1d

            self.vigfun = interp1d(
                vignetting["th"],
                vignetting["unvig"],
                kind="cubic",
                bounds_error=False,
                fill_value=(vignetting["unvig"][0], vignetting["unvig"][-1]),
            )

    @staticmethod
    def from_config(config):
        """
        Create an Instrument from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Instrument configuration dictionary.

        Returns
        -------
        Instrument
            Configured instrument instance.
        """
        return Instrument(
            image_shape=config["image_shape"],
            gain=config["gain"],
            read_noise=config["read_noise"],
            pixel_scale=config["pixel_scale"],
            aperture=config["aperture"],
            obscuration=config.get("obscuration", 0.0),
            distortion=config.get("distortion"),
            vignetting=config.get("vignetting"),
        )

    def init_image(self, *, sky_phot, exptime):
        """
        Initialize an image with sky background.

        Parameters
        ----------
        sky_phot : float
            Sky level in photons / arcsec^2 / sec.
        exptime : float
            Exposure time in seconds.

        Returns
        -------
        galsim.Image
            Image initialized with sky background.
        """
        nx, ny = self.image_shape
        bounds = galsim.BoundsI(-nx // 2, nx // 2 - 1, -ny // 2, ny // 2 - 1)
        image = galsim.Image(bounds)

        boresight = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
        rot_sky_pos = 0 * galsim.degrees
        mock_wcs = self.get_wcs(boresight, rot_sky_pos)
        mock_wcs.makeSkyImage(image, sky_phot * exptime)
        return image

    @galsim.utilities.lazy_property
    def field_radius(self):
        """
        Angular distance from boresight to an image corner.

        Returns
        -------
        galsim.Angle
            Field radius.
        """
        boresight = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
        rot_sky_pos = 0 * galsim.degrees
        mock_wcs = self.get_wcs(boresight, rot_sky_pos)
        nx, ny = self.image_shape
        coord = mock_wcs.toWorld(galsim.PositionD(nx / 2, ny / 2))
        return coord.distanceTo(boresight)

    def apply_vignetting(self, image):
        """
        Apply vignetting to an image in place.

        Parameters
        ----------
        image : galsim.Image
            Image to modify.
        """
        if self.vignetting is None:
            return

        boresight = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
        rot_sky_pos = 0 * galsim.degrees
        mock_wcs = self.get_wcs(boresight, rot_sky_pos)

        xx, yy = image.get_pixel_centers()
        rr = np.hypot(xx, yy)
        rr *= np.sqrt(mock_wcs.pixelArea(galsim.PositionD(0, 0)))  # arcsec
        image.array[:] *= self.vigfun(rr / 3600.0)

    def apply_noise(self, image, sky_phot, rng):
        """
        Add CCD noise to an image in place.

        Parameters
        ----------
        image : galsim.Image
            Image to modify.
        sky_phot : float
            Sky level in photons / arcsec^2 / sec.
        rng : numpy.random.Generator
            Random number generator.
        """
        gsrng = galsim.BaseDeviate(rng.bit_generator.random_raw() % (2**63))
        noise = galsim.CCDNoise(
            gsrng,
            sky_level=sky_phot * self.pix_size**2,
            gain=self.gain,
            read_noise=self.read_noise,
        )
        image.addNoise(noise)

    def get_wcs(self, boresight, rot_sky_pos):
        """
        Construct a WCS for the instrument.

        Parameters
        ----------
        boresight : galsim.CelestialCoord
            Telescope pointing direction.
        rot_sky_pos : galsim.Angle
            Position angle of detector up relative to north.

        Returns
        -------
        galsim.GSFitsWCS
            World coordinate system.
        """
        if self.distortion is None:
            raise ValueError("Instrument distortion model is required to construct a WCS.")

        return radial_wcs(
            self.distortion["th"],
            np.asarray(self.distortion["dthdr"]) * self.pixel_scale,
            boresight,
            rot_sky_pos,
            n_control=8,
            sip_order=3,
        )

    @galsim.utilities.lazy_property
    def pix_size(self):
        """
        Mean angular pixel size in arcseconds.

        Returns
        -------
        float
            Mean pixel size in arcseconds.
        """
        if self.distortion is None:
            raise ValueError("Instrument distortion model is required to compute pix_size.")

        return np.mean(self.distortion["dthdr"]) * self.pixel_scale

    def compute_scaled_zp(self, reference_area, reference_zp):
        """
        Scale a reference photometric zeropoint by collecting area.

        Parameters
        ----------
        reference_area : float
            Reference collecting area in square meters.
        reference_zp : float
            Reference zeropoint expressed as electrons/sec for a source of
            fixed reference magnitude 24 AB.

        Returns
        -------
        float
            Zeropoint in AB magnitudes such that an object of magnitude ``zp``
            produces 1 photon/sec.

        Notes
        -----
        This assumes throughput is otherwise unchanged and only collecting
        area is rescaled.
        """
        radius = 0.5 * self.aperture
        area = np.pi * radius**2 * (1 - self.obscuration**2)
        zp_electrons = reference_zp * area / reference_area
        return np.log10(zp_electrons) / 0.4 + 24

    def compute_lsst_scaled_zp(self):
        """
        Compute a zeropoint by scaling an LSST i-band reference value.

        Returns
        -------
        float
            Approximate photometric zeropoint in AB magnitudes.

        Notes
        -----
        This helper is retained for backward compatibility and uses an LSST
        i-band reference value. For general use, prefer ``compute_scaled_zp``.
        """
        lsst_area = 32.4
        lsst_reference_zp = 32.36
        return self.compute_scaled_zp(lsst_area, lsst_reference_zp)

    def streak_snr(self, *, nphot, length, psf_fwhm, sky_phot):
        """
        Compute the signal-to-noise ratio for a linear streak.

        Parameters
        ----------
        nphot : array-like or float
            Number of photons in the streak.
        length : float
            Streak length in arcseconds.
        psf_fwhm : float
            PSF full width at half maximum in arcseconds.
        sky_phot : float
            Sky level in photons / arcsec^2 / sec.

        Returns
        -------
        array-like or float
            Streak signal-to-noise ratio.
        """
        neff = 2.266 * (psf_fwhm / self.pix_size) ** 2
        reff = np.sqrt(neff / np.pi)

        aeff = neff + reff * length
        var = np.asarray(nphot) / self.gain
        var += (sky_phot * self.pix_size**2 / self.gain + self.read_noise**2) * aeff
        return np.asarray(nphot) / np.sqrt(var)


# Backward-compatible aliases
Instrument.fromConfig = Instrument.from_config