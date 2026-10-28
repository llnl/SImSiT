"""
Telescope and detector instrument models for observation simulation.
"""

import galsim
import numpy as np

from .wcs import radial_wcs
from .passband import Passband as _Passband
from .passband import ab_magnitude_to_photon_rate as _ab_photon_rate
from .passband import photon_rate as _spectral_photon_rate
from .passband import calibration_reference as _calibration_reference
import astropy.units as u


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
    quantum_efficiency : float, optional
        Detected electrons per incident photon, greater than zero and at most one.
    passband : simsit.passband.Passband or dict, optional
        Named optical response with calibration provenance. Required for
        magnitude/spectrum photon conversion and the simulation pipeline.
    zeropoint : float, optional
        Empirical photon zeropoint, represented as additional achromatic
        transmission bounded so total throughput cannot exceed one.
    zeropoint_reference : str, optional
        Required calibration reference when an empirical zeropoint is supplied.
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
        quantum_efficiency=1.0,
        passband=None,
        zeropoint=None,
        zeropoint_reference=None,
    ):
        if (len(image_shape) != 2
                or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in image_shape)):
            raise ValueError("image_shape must contain two positive integers.")
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive.")
        if not np.isfinite(read_noise) or read_noise < 0:
            raise ValueError("read_noise must be finite and nonnegative.")
        if not np.isfinite(quantum_efficiency) or not 0 < quantum_efficiency <= 1:
            raise ValueError("quantum_efficiency must be greater than zero and at most one.")
        self.image_shape = image_shape
        self.gain = gain
        self.read_noise = read_noise
        self.pixel_scale = pixel_scale
        self.aperture = aperture
        self.obscuration = obscuration
        self.distortion = distortion
        self.vignetting = vignetting
        self.quantum_efficiency = quantum_efficiency
        if not np.isfinite(aperture) or aperture <= 0 or not np.isfinite(obscuration) or not 0 <= obscuration < 1:
            raise ValueError("aperture must be positive and obscuration must be in [0, 1).")
        self.passband = _Passband.from_config(passband) if isinstance(passband, dict) else passband
        if self.passband is not None and not isinstance(self.passband, _Passband):
            raise TypeError("passband must be a Passband or response configuration.")
        if zeropoint is not None and self.passband is None:
            raise ValueError("A supplied photon zeropoint requires a passband response and calibration reference.")
        self.zeropoint_reference = zeropoint_reference
        if zeropoint_reference is not None:
            _calibration_reference(zeropoint_reference)
        self.calibration_scale = 1.0
        if self.passband is not None and zeropoint is not None:
            if not np.isfinite(zeropoint) or not isinstance(zeropoint_reference, str) or not zeropoint_reference.strip():
                raise ValueError("A supplied photon zeropoint requires a finite value and calibration reference.")
            rate0 = _ab_photon_rate(0, self.passband, self.collecting_area).to_value(1 / u.s)
            self.calibration_scale = 10 ** (0.4 * zeropoint) / rate0
            self.photon_scale(0)

        if vignetting is not None:
            from scipy.interpolate import PchipInterpolator

            angles = np.asarray(vignetting["th"], dtype=float)
            throughput = np.asarray(vignetting["unvig"], dtype=float)
            if (angles.ndim != 1 or throughput.shape != angles.shape
                    or len(angles) < 2 or not np.isfinite(angles).all()
                    or not np.isfinite(throughput).all() or angles[0] != 0
                    or np.any(np.diff(angles) <= 0)
                    or np.any((throughput < 0) | (throughput > 1))):
                raise ValueError("vignetting requires increasing angles from zero and fractions in [0, 1].")
            interpolator = PchipInterpolator(angles, throughput)
            self.vigfun = lambda angle: interpolator(np.clip(angle, angles[0], angles[-1]))

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
            quantum_efficiency=config.get("quantum_efficiency", 1.0),
            passband=config.get("passband"),
            zeropoint=config.get("zeropoint"),
            zeropoint_reference=config.get("zeropoint_reference"),
        )

    @property
    def collecting_area(self):
        """Unobscured collecting area in square meters."""
        return np.pi * (self.aperture / 2)**2 * (1 - self.obscuration**2)

    def photon_scale(self, zeropoint_offset=0.0):
        """Additional achromatic transmission, bounded by physical throughput."""
        if self.passband is None:
            raise ValueError("Photon conversion requires an explicit passband response.")
        if np.ndim(zeropoint_offset) or not np.isfinite(zeropoint_offset):
            raise ValueError("Zeropoint offset must be finite.")
        scale = self.calibration_scale * 10 ** (0.4 * zeropoint_offset)
        if not np.isfinite(scale) or scale <= 0 or scale * np.max(self.passband.throughput) > 1 + 1e-12:
            raise ValueError("Photon calibration would exceed physical throughput or produce zero flux.")
        return scale

    @property
    def photon_zeropoint(self):
        """Passband AB magnitude producing one incident photon/sec before vignetting."""
        rate0 = self.photon_rate(0).to_value(1 / u.s)
        return 2.5 * np.log10(rate0)

    def photon_rate(self, magnitude, *, zeropoint_offset=0.0):
        """Incident photon rate for matching photon-weighted band AB magnitudes."""
        scale = self.photon_scale(zeropoint_offset)
        return _ab_photon_rate(magnitude, self.passband, self.collecting_area) * scale

    def spectral_photon_rate(self, wavelength, flux_lambda, *, zeropoint_offset=0.0):
        """Incident photons from a physical spectrum, with the same calibration."""
        scale = self.photon_scale(zeropoint_offset)
        return _spectral_photon_rate(wavelength, flux_lambda, self.passband, self.collecting_area) * scale

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
        xmin, ymin = -(nx // 2), -(ny // 2)
        bounds = galsim.BoundsI(xmin, xmin + nx - 1, ymin, ymin + ny - 1)
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
        ra, dec = mock_wcs.xyToradec(xx, yy, units="radians")
        radius = np.arctan2(
            np.hypot(np.cos(dec) * np.sin(ra), np.sin(dec)),
            np.cos(dec) * np.cos(ra),
        )
        image.array[:] *= self.vigfun(np.rad2deg(radius))

    def apply_noise(self, image, sky_phot, rng):
        """
        Convert expected incident photons to ADU and add CCD noise once.

        Parameters
        ----------
        image : galsim.Image
            Image to modify.
        sky_phot : float
            Retained for compatibility; sky is already included in ``image``.
        rng : numpy.random.Generator
            Random number generator.
        """
        image.array[:] *= self.quantum_efficiency / self.gain
        gsrng = galsim.BaseDeviate(int(rng.integers(1, 2**31)))
        noise = galsim.CCDNoise(
            gsrng,
            sky_level=0.0,
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

    def streak_snr(self, *, nphot, length, psf_fwhm, sky_phot, exptime=1.0):
        """
        Compute the signal-to-noise ratio for a linear streak.

        Parameters
        ----------
        nphot : array-like or float
            Number of photons in the streak.
        length : float
            Streak length in pixels.
        psf_fwhm : float
            PSF full width at half maximum in arcseconds.
        sky_phot : float
            Sky level in photons / arcsec^2 / sec.
        exptime : float, optional
            Exposure duration in seconds; defaults to one for compatibility.

        Returns
        -------
        array-like or float
            Streak signal-to-noise ratio.
        """
        neff = 2.266 * (psf_fwhm / self.pix_size) ** 2
        reff = np.sqrt(neff / np.pi)

        aeff = neff + reff * length
        signal = np.asarray(nphot, dtype=float) * self.quantum_efficiency
        var = signal + (
            sky_phot * exptime * self.pix_size**2 * self.quantum_efficiency
            + self.read_noise**2
        ) * aeff
        return np.divide(signal, np.sqrt(var), out=np.zeros_like(signal), where=var > 0)


# Backward-compatible aliases
Instrument.fromConfig = Instrument.from_config
