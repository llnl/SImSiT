"""Bounded physical satellite brightness through public SSATK models."""

from __future__ import annotations

from collections.abc import Mapping

import astropy.units as u
import numpy as np

from . import backend
from .instrument import Instrument

__all__ = ["PhysicalBrightness", "model_from_config"]


_LAMBERTIAN_KEYS = {
    "radius_m", "albedo", "include_sun", "include_earthshine", "include_moonshine",
    "k_extinction", "solar_const", "t_sun", "r_sun_radius", "r_earth",
    "albedo_earth", "olr_earth", "t_earth_lw", "r_moon", "albedo_moon",
    "atmosphere_top_m", "f_nu_ab_zero",
}
_FACETED_KEYS = {
    "include_sun", "include_earthshine", "include_moonshine", "k_extinction",
    "solar_const", "t_sun", "r_sun_radius", "r_earth", "albedo_earth",
    "olr_earth", "t_earth_lw", "r_moon", "albedo_moon", "atmosphere_top_m",
    "f_nu_ab_zero",
    "facets", "quaternion", "self_shadowing", "shadow_epsilon",
    "check_line_of_sight",
}
_THERMAL_KEYS = {"temperature_K", "emissivity", "include_thermal"}


def model_from_config(instrument, config):
    """Validate an explicit physical or constant photometry configuration."""
    if not isinstance(config, Mapping) or config.get("model") not in ("constant", "lambertian", "faceted"):
        raise ValueError("photometry.model must be constant, lambertian or faceted.")
    allowed = {"model", "params", "spectral_relative_tolerance", "max_spectral_samples"}
    if set(config) - allowed:
        raise ValueError("Unknown photometry configuration fields.")
    if config["model"] == "constant":
        if config.get("params") or set(config) - {"model", "params"}:
            raise ValueError("Constant photometry does not use physical-model parameters.")
        return None
    params = dict(config.get("params", {}))
    if config["model"] == "faceted":
        params["facets"] = backend.facets_from_config(params.get("facets", ()))
    return PhysicalBrightness(
        instrument, model=config["model"], params=params,
        spectral_relative_tolerance=config.get("spectral_relative_tolerance", 1e-4),
        max_spectral_samples=config.get("max_spectral_samples", 257),
    )


def _finite_scalar(value, name, *, minimum=None, maximum=None, strict_min=False):
    if isinstance(value, (bool, np.bool_)) or np.ma.getmaskarray(value).any():
        raise ValueError(f"{name} must be an unmasked numeric scalar.")
    if isinstance(value, u.Quantity):
        value = value.to_value(u.dimensionless_unscaled)
    if np.ndim(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar.")
    value = float(value)
    if minimum is not None and (value <= minimum if strict_min else value < minimum):
        relation = "greater than" if strict_min else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def _passband_domain(instrument):
    if not isinstance(instrument, Instrument):
        raise TypeError("instrument must be a simsit.Instrument.")
    if instrument.passband is None:
        raise ValueError("Physical brightness requires an explicit instrument passband.")
    wavelengths = instrument.passband.wavelength_nm.astype(float) * 1e-9
    if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
        raise ValueError("Instrument passband wavelengths must be finite and positive.")
    return wavelengths


def _grid(domain, count):
    points = list(domain)
    target = min(9, count)
    while len(points) < target:
        points = sorted(points)
        gaps = np.log(points[1:]) - np.log(points[:-1])
        index = int(np.argmax(gaps))
        points.append(float(np.sqrt(points[index] * points[index + 1])))
    return np.asarray(sorted(points), dtype=float)


def _widest_log_midpoints(wavelengths, count):
    gaps = np.log(wavelengths[1:]) - np.log(wavelengths[:-1])
    order = np.argsort(gaps)[::-1][:count]
    return np.sqrt(wavelengths[order] * wavelengths[order + 1])


class PhysicalBrightness:
    """Estimate incident detector photons from an explicit SSATK brightness model.

    SSATK supplies band-integrated irradiance using gray Lambertian reflection
    and blackbody approximations for solar/planetary sources.  This class
    queries narrow bands and treats their mean as ``F_lambda``; it does not
    add spacecraft spectroscopy, orbit propagation, attitude inference, or
    detector QE/gain.  The adaptive estimate is bounded by
    ``max_spectral_samples`` and the stated relative tolerance.
    """

    def __init__(
        self,
        instrument,
        model="lambertian",
        params=None,
        spectral_relative_tolerance=1e-4,
        max_spectral_samples=257,
    ):
        self.instrument = instrument
        self.model = model
        if params is not None and not isinstance(params, Mapping):
            raise TypeError("params must be a mapping of physical-model parameters.")
        self.params = dict(params or {})
        self.spectral_relative_tolerance = _finite_scalar(
            spectral_relative_tolerance, "spectral_relative_tolerance",
            minimum=0.0, strict_min=True, maximum=1.0,
        )
        if (not isinstance(max_spectral_samples, (int, np.integer))
                or max_spectral_samples < 2):
            raise ValueError("max_spectral_samples must be an integer of at least 2.")
        self.max_spectral_samples = int(max_spectral_samples)
        domain = _passband_domain(instrument)
        if self.max_spectral_samples < len(domain):
            raise ValueError("max_spectral_samples must include every passband knot.")
        if model not in ("lambertian", "faceted"):
            raise ValueError("model must be 'lambertian' or 'faceted'.")
        allowed = _LAMBERTIAN_KEYS if model == "lambertian" else _FACETED_KEYS
        thermal_keys = _THERMAL_KEYS if model == "lambertian" else set()
        unsupported = sorted(set(self.params) - allowed - thermal_keys)
        if unsupported:
            raise ValueError(f"Unsupported {model} physical-model parameters: {unsupported}.")
        if model == "lambertian" and not {"radius_m", "albedo"} <= self.params.keys():
            raise ValueError("lambertian brightness requires explicit radius_m and albedo.")
        self._validate_params()

    def _validate_params(self):
        params = self.params
        units = {
            "radius_m": u.m, "r_sun_radius": u.m, "r_earth": u.m, "r_moon": u.m,
            "atmosphere_top_m": u.m, "shadow_epsilon": u.m,
            "temperature_K": u.K, "t_sun": u.K, "t_earth_lw": u.K,
            "solar_const": u.W / u.m**2, "olr_earth": u.W / u.m**2,
            "f_nu_ab_zero": u.W / u.m**2 / u.Hz,
        }
        for key, value in tuple(params.items()):
            if isinstance(value, u.Quantity):
                params[key] = value.to_value(units.get(key, u.dimensionless_unscaled))
        for key, value in params.items():
            if key in {
                "facets", "quaternion", "include_sun", "include_earthshine",
                "include_moonshine", "include_thermal", "self_shadowing",
                "check_line_of_sight",
            }:
                continue
            if key in {"albedo", "albedo_earth", "albedo_moon", "emissivity"}:
                _finite_scalar(value, key, minimum=0.0, maximum=1.0)
            elif key == "temperature_K" or key in {"t_sun", "t_earth_lw"}:
                _finite_scalar(value, key, minimum=0.0, strict_min=True)
            elif key == "radius_m":
                _finite_scalar(value, key, minimum=0.0, strict_min=True)
            else:
                _finite_scalar(value, key, minimum=0.0)
        for key in {
            "include_sun", "include_earthshine", "include_moonshine",
            "include_thermal", "self_shadowing", "check_line_of_sight",
        } & params.keys():
            if not isinstance(params[key], (bool, np.bool_)):
                raise TypeError(f"{key} must be boolean.")
        if self.model == "faceted":
            if "facets" not in params or "quaternion" not in params:
                raise ValueError("faceted brightness requires facets and quaternion.")
            if "temperature_K" in params or "emissivity" in params or params.get("include_thermal"):
                raise ValueError("faceted thermal emission is not exposed by SSATK 1.1.4.")
            quaternion = np.asarray(params["quaternion"], dtype=float)
            norm = np.linalg.norm(quaternion)
            if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                    or not np.isfinite(norm) or not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-8)):
                raise ValueError("quaternion must be one finite [w, x, y, z] vector.")
            if len(params["facets"]) == 0:
                raise ValueError("facets must not be empty.")
            for facet in params["facets"]:
                _finite_scalar(facet.area, "facet.area", minimum=0, strict_min=True)
                normal = np.asarray(facet.normal_body, dtype=float)
                if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
                    raise ValueError("Facet normals must be finite nonzero body-frame vectors.")
                reflectivity = facet.diffuse_reflectivity
                if isinstance(reflectivity, Mapping):
                    reflectivity = reflectivity.get("custom")
                if reflectivity is None:
                    raise ValueError("Facets require diffuse reflectivity for the custom spectral band.")
                _finite_scalar(reflectivity, "facet.diffuse_reflectivity", minimum=0, maximum=1)
        if self.model == "lambertian" and (
            "temperature_K" in params or "emissivity" in params or params.get("include_thermal")
        ) and "temperature_K" not in params:
            raise ValueError("temperature_K is required when thermal parameters are supplied.")

    def _band_irradiance(self, position, time, observer, lo, hi):
        kwargs = dict(self.params)
        kwargs.pop("temperature_K", None)
        kwargs.pop("emissivity", None)
        include_thermal = bool(
            kwargs.pop("include_thermal", "temperature_K" in self.params)
        )
        if self.model == "lambertian":
            kwargs.setdefault("radius_m", 1.0)
            reflection = backend.lambertian_reflection(
                position, time, observer, band=(lo, hi), **kwargs,
            )
        else:
            reflection = backend.faceted_reflection(
                position, time, observer, band=(lo, hi), **kwargs,
            )
        value = float(reflection["irradiance_inband_total_at_observer_W_m2"])
        if include_thermal:
            thermal = backend.thermal_emission(
                position, time, observer, band=(lo, hi),
                radius_m=kwargs.get("radius_m", 1.0),
                albedo=kwargs.get("albedo", 0.3),
                temperature_K=self.params["temperature_K"],
                emissivity=self.params.get("emissivity"),
                k_extinction=kwargs.get("k_extinction", 0.16),
                r_earth=kwargs.get("r_earth", 6378137.0),
                atmosphere_top_m=kwargs.get("atmosphere_top_m", 100000.0),
            )
            value += float(thermal["irradiance_inband_total_at_observer_W_m2"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("SSATK physical model returned invalid irradiance.")
        return value / (hi - lo)

    def _spectrum(self, position, time, observer, wavelengths):
        flux = np.empty(wavelengths.size)
        for index, wavelength in enumerate(wavelengths):
            lo = wavelength * (1.0 - 5.0e-5)
            hi = wavelength * (1.0 + 5.0e-5)
            flux[index] = self._band_irradiance(position, time, observer, lo, hi)
        return wavelengths * u.m, flux * u.W / u.m**2 / u.m

    def photon_rate(self, position_gcrf_m, time, observer):
        """Return incident photons/sec after the optical passband only."""
        position = backend._physical_position(position_gcrf_m)
        epoch = backend.as_time(time)
        if not epoch.isscalar:
            raise ValueError("Physical brightness requires one scalar epoch.")
        observer = backend._physical_observer(observer, epoch)
        domain = _passband_domain(self.instrument)
        wavelengths = _grid(domain, self.max_spectral_samples)
        previous = None
        while True:
            wavelength, flux = self._spectrum(position, epoch, observer, wavelengths)
            rate = self.instrument.spectral_photon_rate(wavelength, flux)
            value = float(rate.to_value(1 / u.s))
            if not np.isfinite(value) or value < 0:
                raise ValueError("Instrument spectral integration returned invalid photons/sec.")
            if previous is not None and (
                value == previous == 0.0
                or (
                    abs(value - previous)
                    <= self.spectral_relative_tolerance
                    * max(abs(value), abs(previous), 1e-300)
                )
            ):
                return rate.to(1 / u.s)
            if wavelengths.size >= self.max_spectral_samples:
                raise RuntimeError(
                    "Physical brightness spectral estimate did not converge within "
                    f"{self.max_spectral_samples} samples."
                )
            previous = value
            room = self.max_spectral_samples - wavelengths.size
            midpoints = _widest_log_midpoints(wavelengths, room)
            wavelengths = np.unique(np.concatenate((wavelengths, midpoints)))
