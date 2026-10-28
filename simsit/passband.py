"""Calibrated, bounded photometric passband calculations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import quad


AB_ZERO_FNU = (0 * u.ABmag).to(u.Jy)
_AB_ZERO_FNU = AB_ZERO_FNU
__all__ = [
    "Passband", "AB_ZERO_FNU", "calibration_reference", "photon_rate",
    "spectrum_to_magnitude", "ab_magnitude_to_photon_rate",
]


def calibration_reference(reference):
    """Validate a non-secret calibration label or public source URI."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Calibration reference must be a non-empty provenance label.")
    parsed = urlsplit(reference)
    if parsed.username is not None or parsed.password is not None or any(
        any(part in key.lower() for part in (
            "token", "api_key", "apikey", "password", "passwd", "credential",
            "signature", "authorization", "secret", "access_key",
        ))
        for section in (parsed.query, parsed.fragment)
        for key in parse_qs(section)
    ):
        raise ValueError("Calibration references must not contain credentials.")
    return reference


@dataclass(frozen=True, eq=False)
class Passband:
    """A named, tabulated response curve with explicit provenance.

    ``reference`` identifies the calibration or source of the supplied curve.
    The response is linearly interpolated between the tabulated points and is
    zero outside the tabulated domain; calculations reject spectra that would
    require that implicit outside-domain behavior.
    """

    name: str
    wavelength_nm: np.ndarray
    throughput: np.ndarray
    reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        calibration_reference(self.reference)

        wavelength = np.array(self.wavelength_nm, dtype=float, copy=True)
        response = np.array(self.throughput, dtype=float, copy=True)
        if wavelength.ndim != 1 or response.ndim != 1:
            raise ValueError("wavelength_nm and throughput must be one-dimensional")
        if wavelength.size < 2 or wavelength.size != response.size:
            raise ValueError("arrays must contain at least two matching samples")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(response)):
            raise ValueError("passband arrays must contain only finite values")
        if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
            raise ValueError("wavelength_nm must be positive and strictly increasing")
        if np.any((response < 0) | (response > 1)):
            raise ValueError("throughput must be between zero and one")
        integral = np.sum(
            0.5 * (response[:-1] + response[1:]) * np.diff(wavelength)
        )
        if integral <= 0:
            raise ValueError("throughput must have a nonzero integral")

        # A read-only view backed by bytes cannot be made writable by callers.
        wavelength_immutable = np.frombuffer(
            wavelength.tobytes(), dtype=wavelength.dtype
        ).reshape(wavelength.shape)
        response_immutable = np.frombuffer(
            response.tobytes(), dtype=response.dtype
        ).reshape(response.shape)
        object.__setattr__(self, "wavelength_nm", wavelength_immutable)
        object.__setattr__(self, "throughput", response_immutable)

    @property
    def provenance(self) -> str:
        """Alias for the explicit calibration/source label."""
        return self.reference

    @property
    def response_hash(self) -> str:
        """SHA-256 identifying the named response, independent of reference text."""
        data = self.to_config()
        data.pop("reference")
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Passband":
        """Construct a passband from a serializable configuration mapping."""
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        required = ("name", "wavelength_nm", "throughput", "reference")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"config is missing required keys: {', '.join(missing)}")
        return cls(
            name=config["name"],
            wavelength_nm=config["wavelength_nm"],
            throughput=config["throughput"],
            reference=config["reference"],
        )

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this passband."""
        return {
            "name": self.name,
            "wavelength_nm": self.wavelength_nm.tolist(),
            "throughput": self.throughput.tolist(),
            "reference": self.reference,
        }


def _wavelength_meters(wavelength: u.Quantity) -> np.ndarray:
    if np.ma.getmaskarray(wavelength).any():
        raise ValueError("wavelength samples must not be masked")
    if not isinstance(wavelength, u.Quantity) or not wavelength.unit.is_equivalent(u.m):
        raise TypeError("wavelength must be an Astropy length Quantity")
    values = np.asarray(wavelength.to_value(u.m), dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("wavelength must be a one-dimensional sample array")
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(np.diff(values) <= 0):
        raise ValueError("wavelength must be finite, positive, and strictly increasing")
    return values


def _flux_si(flux_lambda: u.Quantity, size: int) -> np.ndarray:
    if np.ma.getmaskarray(flux_lambda).any():
        raise ValueError("spectral flux samples must not be masked")
    if not isinstance(flux_lambda, u.Quantity):
        raise TypeError("flux_lambda must be an Astropy spectral-flux-density Quantity")
    try:
        flux = flux_lambda.to(u.W / (u.m**2 * u.m))
    except u.UnitConversionError as exc:
        raise TypeError("flux_lambda must have spectral-flux-density units") from exc
    values = np.asarray(flux.value, dtype=float)
    if values.ndim != 1 or values.size != size:
        raise ValueError("flux_lambda must match the wavelength sample shape")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("flux_lambda must contain finite, nonnegative values")
    return values


def _check_coverage(wavelength_m: np.ndarray, passband: Passband) -> None:
    lower, upper = passband.wavelength_nm[[0, -1]] * 1e-9
    tolerance = 16 * np.finfo(float).eps * max(abs(lower), abs(upper))
    if wavelength_m[0] > lower + tolerance or wavelength_m[-1] < upper - tolerance:
        raise ValueError("wavelength samples must cover the complete passband domain")


def _response(passband: Passband, wavelength_m: float) -> float:
    return float(
        np.interp(
            wavelength_m,
            passband.wavelength_nm * 1e-9,
            passband.throughput,
            left=0.0,
            right=0.0,
        )
    )


def _sampled_integral(
    wavelength_m: np.ndarray,
    flux: np.ndarray,
    passband: Passband,
    integrand,
) -> float:
    """Integrate a piecewise-linear spectrum and response without extrapolation."""
    _check_coverage(wavelength_m, passband)
    knots = np.unique(
        np.concatenate((wavelength_m, passband.wavelength_nm * 1e-9))
    )

    def value(x: float) -> float:
        spectrum = float(np.interp(x, wavelength_m, flux))
        return integrand(x, spectrum, _response(passband, x))

    return float(
        sum(
            quad(value, left, right, epsabs=0.0, epsrel=2e-12, limit=100)[0]
            for left, right in zip(knots[:-1], knots[1:])
        )
    )


def photon_rate(
    wavelength: u.Quantity,
    flux_lambda: u.Quantity,
    passband: Passband,
    area_m2: float | u.Quantity,
) -> u.Quantity:
    """Return incident photons per second after the optical response.

    This applies only the supplied passband throughput; detector quantum
    efficiency and gain are not included and must not be double-counted here.
    """
    wavelengths = _wavelength_meters(wavelength)
    flux = _flux_si(flux_lambda, wavelengths.size)
    area = u.Quantity(area_m2, u.m**2).to_value(u.m**2)
    if np.ndim(area) or not np.isfinite(area) or area <= 0:
        raise ValueError("area_m2 must be finite and positive")
    integral = _sampled_integral(
        wavelengths,
        flux,
        passband,
        lambda wave, density, response: density * response * wave / const.h.value / const.c.value,
    )
    return integral * area / u.s


def spectrum_to_magnitude(
    wavelength: u.Quantity,
    flux_lambda: u.Quantity,
    passband: Passband,
) -> float:
    """Return the AB magnitude of a sampled spectrum in ``passband``."""
    wavelengths = _wavelength_meters(wavelength)
    flux = _flux_si(flux_lambda, wavelengths.size)
    denominator = _sampled_integral(
        wavelengths,
        np.ones_like(flux),
        passband,
        lambda wave, _density, response: response / wave,
    )
    if denominator <= 0:
        raise ValueError("passband photon-weighted denominator must be positive")
    numerator = _sampled_integral(
        wavelengths,
        flux,
        passband,
        lambda wave, density, response: density * response * wave,
    ) / const.c.value
    if numerator == 0:
        return float("inf")
    fnu = (numerator / denominator) * (u.W / u.m**2 / u.Hz)
    return float(-2.5 * np.log10((fnu / _AB_ZERO_FNU).to_value(u.dimensionless_unscaled)))


def _integral_linear_response_over_wavelength(passband: Passband) -> float:
    """Integrate T(lambda) / lambda exactly over tabulated linear segments."""
    wavelength = passband.wavelength_nm * 1e-9
    response = passband.throughput
    total = 0.0
    for left, right, t_left, t_right in zip(
        wavelength[:-1], wavelength[1:], response[:-1], response[1:]
    ):
        width = right - left
        z = width / left
        log_ratio = np.log1p(z)
        # q(z) = (z - log1p(z)) / z, evaluated by series for narrow segments.
        if z < 1e-4:
            q = z * (
                0.5
                + z
                * (
                    -1 / 3
                    + z * (1 / 4 + z * (-1 / 5 + z * (1 / 6 - z / 7)))
                )
            )
        else:
            q = (z - log_ratio) / z
        slope = (t_right - t_left) / width
        total += t_left * log_ratio + slope * width * q
    return float(total)


def ab_magnitude_to_photon_rate(
    magnitude: float | np.ndarray,
    passband: Passband,
    area_m2: float | u.Quantity,
) -> u.Quantity:
    """Return photon rates for photon-weighted band-mean AB magnitudes.

    The conversion is exact by the definition used by
    :func:`spectrum_to_magnitude`, including for non-flat input SEDs. A
    magnitude is a band mean here, not a claim that the source spectrum is
    flat in F-nu.
    """
    magnitudes = np.asarray(magnitude, dtype=float)
    if np.isnan(magnitudes).any() or np.isneginf(magnitudes).any():
        raise ValueError("magnitude must be finite or positive infinity for zero flux")
    area = u.Quantity(area_m2, u.m**2).to_value(u.m**2)
    if np.ndim(area) or not np.isfinite(area) or area <= 0:
        raise ValueError("area_m2 must be finite and positive")
    with np.errstate(over="raise", invalid="raise"):
        try:
            fnu = (_AB_ZERO_FNU * 10 ** (-0.4 * magnitudes)).to_value(
                u.W / u.m**2 / u.Hz
            )
        except FloatingPointError as exc:
            raise ValueError("magnitude produces a nonfinite flux") from exc
    integral = _integral_linear_response_over_wavelength(passband)
    return (area * fnu * integral / const.h.value) / u.s
