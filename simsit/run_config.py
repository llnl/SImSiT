"""Validation and provenance helpers for reproducible simulation runs.

This module deliberately does not construct simulator objects or write run
products.  It provides a small preflight boundary for simulation run
configuration and provenance.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import ntpath
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from .passband import calibration_reference as _calibration_reference


_REQUIRED = (
    "n_obs",
    "n_example",
    "n_sat",
    "seed",
    "instrument",
    "conditions",
    "cadence",
    "tracker",
    "sites",
    "sat",
    "catalog",
)
_TOP_LEVEL_KEYS = set(_REQUIRED) | {
    "outdir", "image_dir", "reference_dir", "meta", "photometry",
    "integration", "observation",
}
_SECRET_PARTS = (
    "credential",
    "secret",
    "password",
    "token",
    "apikey",
    "authorization",
    "accesskey",
    "privatekey",
    "clientsecret",
    "refreshtoken",
    "bearer",
    "signature",
    "passwd",
    "cookie",
    "session",
    "jwt",
    "private",
)
_DEPENDENCIES = (
    "simsit",
    "numpy",
    "scipy",
    "matplotlib",
    "scikit-learn",
    "galsim",
    "astropy",
    "astroplan",
    "ssatk",
    "llnl-ssapy",
    "ssa-data-core",
    "ssa-data-gravity",
    "ssa-data-lunar",
    "ssa-data-lunar-gravity",
    "astropy-iers-data",
    "jplephem",
    "sgp4",
    "astroquery",
    "lsst-sphgeom",
    "PyYAML",
    "tqdm",
)
_INSTRUMENT_KEYS = {
    "image_shape", "gain", "read_noise", "pixel_scale", "aperture",
    "obscuration", "distortion", "vignetting", "quantum_efficiency",
    "passband", "zeropoint", "zeropoint_reference",
}
_DISTORTION_KEYS = {"th", "dthdr"}
_VIGNETTING_KEYS = {"th", "unvig"}
_CONDITIONS_KEYS = {"psf_fwhm_range", "sky_range", "zp_range"}
_CADENCE_KEYS = {"exptime"}
_TRACKER_KEYS = {"type", "rot_sky_pos0", "error"}
_TRACKER_ERROR_KEYS = {"t_rewind", "v_perturb", "boresight"}
_SAT_KEYS = {"mag_range"}
_CATALOG_KEYS = {
    "develop", "gaia_dir", "level", "magnitude_column", "magnitude_system",
    "min_snr",
}
_PASSBAND_KEYS = {"name", "wavelength_nm", "throughput", "reference"}
_OBSERVATION_KEYS = {"mode"}
_INTEGRATION_KEYS = {
    "pixel_tolerance", "flux_relative_tolerance", "max_step_s", "max_depth",
    "max_samples",
}
_PHOTOMETRY_KEYS = {
    "model", "params", "spectral_relative_tolerance", "max_spectral_samples",
}
_PHOTOMETRY_MODELS = {"constant", "lambertian", "faceted"}
_LAMBERTIAN_PARAMS = {
    "radius_m", "albedo", "include_sun", "include_earthshine", "include_moonshine",
    "k_extinction", "solar_const", "t_sun", "r_sun_radius", "r_earth",
    "albedo_earth", "olr_earth", "t_earth_lw", "r_moon", "albedo_moon",
    "atmosphere_top_m", "f_nu_ab_zero", "temperature_K", "emissivity",
    "include_thermal",
}
_FACETED_PARAMS = {
    "include_sun", "include_earthshine", "include_moonshine", "k_extinction",
    "solar_const", "t_sun", "r_sun_radius", "r_earth", "albedo_earth",
    "olr_earth", "t_earth_lw", "r_moon", "albedo_moon", "atmosphere_top_m",
    "f_nu_ab_zero", "facets", "quaternion", "self_shadowing",
    "shadow_epsilon", "check_line_of_sight",
}
_FACET_KEYS = {
    "area", "normal_body", "center_of_pressure", "cd", "cr",
    "specular_reflectivity", "diffuse_reflectivity", "thermal_reemission",
    "vertices_body", "name", "cl",
}
_PROVENANCE_PATH_KEYS = {
    "outdir", "image_dir", "reference_dir", "gaia_dir", "path",
    "input_path", "output_path", "filename", "file",
}
_METADATA_SECRET_KEYS = {
    "author", "email", "home", "username", "user", "owner", "path",
    "file", "filename", "directory", "notebook", "workdir", "cwd",
}


def _error(path: str, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise _error(path, "must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise _error(path, "must be positive")
    if nonnegative and result < 0:
        raise _error(path, "must be nonnegative")
    return result


def _integer(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error(path, "must be an integer")
    if positive and value <= 0:
        raise _error(path, "must be positive")
    if nonnegative and value < 0:
        raise _error(path, "must be nonnegative")
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be a mapping")
    return value


def _known_keys(value: Mapping[str, Any], path: str, allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed, key=str)
    if unknown:
        raise _error(path, f"unknown key {unknown[0]!r}")


def _validate_relative_directory(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "must be a nonempty relative directory")
    if os.path.isabs(value) or ntpath.isabs(value) or ntpath.splitdrive(value)[0]:
        raise _error(path, "must be relative")
    if "\\" in value:
        raise _error(path, "must use relative POSIX path components")
    parts = Path(value).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise _error(path, "must be relative and must not contain '.', '..', or empty components")


def _finite_vector(value: Any, path: str, length: int) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise _error(path, f"must contain exactly {length} values")
    for index, item in enumerate(value):
        _finite(item, f"{path}[{index}]")


def _validate_facet(value: Any, path: str) -> None:
    facet = _mapping(value, path)
    _known_keys(facet, path, _FACET_KEYS)
    for key in ("area", "cd", "cr"):
        if key in facet:
            _finite(facet[key], f"{path}.{key}", positive=True)
    if "area" not in facet or "normal_body" not in facet:
        raise _error(path, "requires area and normal_body")
    _finite_vector(facet["normal_body"], f"{path}.normal_body", 3)
    if all(float(item) == 0 for item in facet["normal_body"]):
        raise _error(f"{path}.normal_body", "must not be zero")
    if "center_of_pressure" in facet:
        _finite_vector(facet["center_of_pressure"], f"{path}.center_of_pressure", 3)
    for key in (
        "specular_reflectivity", "diffuse_reflectivity", "thermal_reemission",
    ):
        if key in facet:
            number = _finite(facet[key], f"{path}.{key}")
            if not 0 <= number <= 1:
                raise _error(f"{path}.{key}", "must be a fraction in [0, 1]")
    if (
        "specular_reflectivity" in facet
        and "diffuse_reflectivity" in facet
        and facet["specular_reflectivity"] + facet["diffuse_reflectivity"] > 1
    ):
        raise _error(path, "specular_reflectivity plus diffuse_reflectivity must be at most one")
    if "cl" in facet:
        _finite(facet["cl"], f"{path}.cl")
    if "vertices_body" in facet:
        vertices = facet["vertices_body"]
        if not isinstance(vertices, (list, tuple)) or len(vertices) < 3:
            raise _error(f"{path}.vertices_body", "must contain at least three vertices")
        for index, vertex in enumerate(vertices):
            _finite_vector(vertex, f"{path}.vertices_body[{index}]", 3)
    if "name" in facet and not isinstance(facet["name"], str):
        raise _error(f"{path}.name", "must be a string")


def _validate_photometry_params(model: str, params: Mapping[str, Any]) -> None:
    allowed = set() if model == "constant" else (
        _LAMBERTIAN_PARAMS if model == "lambertian" else _FACETED_PARAMS
    )
    _known_keys(params, "photometry.params", allowed)
    for key, value in params.items():
        if key in {
            "include_sun", "include_earthshine", "include_moonshine",
            "include_thermal", "self_shadowing", "check_line_of_sight",
        }:
            if not isinstance(value, bool):
                raise _error(f"photometry.params.{key}", "must be boolean")
        elif key in {"facets", "quaternion"}:
            continue
        else:
            _finite(value, f"photometry.params.{key}", nonnegative=True)
    if model == "lambertian" and not {"radius_m", "albedo"} <= params.keys():
        raise _error("photometry.params", "lambertian requires radius_m and albedo")
    if "radius_m" in params:
        _finite(params["radius_m"], "photometry.params.radius_m", positive=True)
    for key in {"albedo", "albedo_earth", "albedo_moon", "emissivity"} & params.keys():
        value = _finite(params[key], f"photometry.params.{key}")
        if not 0 <= value <= 1:
            raise _error(f"photometry.params.{key}", "must be a fraction in [0, 1]")
    for key in {"temperature_K", "t_sun", "t_earth_lw"} & params.keys():
        _finite(params[key], f"photometry.params.{key}", positive=True)
    if model == "faceted":
        if "facets" not in params or "quaternion" not in params:
            raise _error("photometry.params", "faceted requires facets and quaternion")
        facets = params["facets"]
        if not isinstance(facets, (list, tuple)) or not facets:
            raise _error("photometry.params.facets", "must be a nonempty list")
        for index, facet in enumerate(facets):
            _validate_facet(facet, f"photometry.params.facets[{index}]")
        quaternion = params["quaternion"]
        _finite_vector(quaternion, "photometry.params.quaternion", 4)
        norm = math.sqrt(sum(float(item) ** 2 for item in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise _error("photometry.params.quaternion", "must have unit norm")
    if model == "lambertian" and (
        "temperature_K" in params or "emissivity" in params
    ) and "temperature_K" not in params:
        raise _error(
            "photometry.params",
            "temperature_K is required with thermal parameters",
        )


def _metadata_pathlike(value: str) -> bool:
    lowered = value.lower()
    return (
        os.path.isabs(value)
        or ntpath.isabs(value)
        or ntpath.splitdrive(value)[0]
        or value.startswith(("~", "./", "../"))
        or "/" in value
        or "\\" in value
        or ".ipynb" in lowered
        or "notebook" in lowered
    )


def _validate_metadata(value: Any, path: str) -> dict[str, Any]:
    metadata = _mapping(value, path)
    safe: dict[str, Any] = {}
    for key, item in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise _error(path, "keys must be nonempty strings")
        normalized = "".join(character for character in key.lower() if character.isalnum())
        metadata_key_parts = {
            "".join(character for character in item_name if character.isalnum())
            for item_name in _METADATA_SECRET_KEYS
        }
        if _is_secret_key(key) or any(part in normalized for part in metadata_key_parts):
            raise _error(f"{path}.{key}", "metadata key is not allowed")
        if isinstance(item, bool) or item is None:
            safe[key] = item
        elif _is_number(item):
            _finite(item, f"{path}.{key}")
            safe[key] = item
        elif isinstance(item, str):
            if _metadata_pathlike(item):
                raise _error(f"{path}.{key}", "path-like metadata is not allowed")
            if _has_unsafe_url(item):
                raise _error(f"{path}.{key}", "URL metadata is not allowed")
            safe[key] = item
        else:
            raise _error(f"{path}.{key}", "must be a scalar JSON value")
    return safe


def _range(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _error(path, "must contain exactly two values")
    low = _finite(value[0], f"{path}[0]", positive=positive, nonnegative=nonnegative)
    high = _finite(value[1], f"{path}[1]", positive=positive, nonnegative=nonnegative)
    if low > high:
        raise _error(path, "must be ordered from low to high")


def _series(value: Any, path: str, *, minimum: int = 2, fractions: bool = False) -> None:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise _error(path, f"must contain at least {minimum} values")
    values = [_finite(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if any(right <= left for left, right in zip(values, values[1:])):
        raise _error(path, "must be strictly increasing")
    if fractions and any(value < 0 or value > 1 for value in values):
        raise _error(path, "must contain fractions in [0, 1]")


def _fractions(value: Any, path: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise _error(path, "must contain at least two values")
    for index, item in enumerate(value):
        number = _finite(item, f"{path}[{index}]")
        if not 0 <= number <= 1:
            raise _error(f"{path}[{index}]", "must be a fraction in [0, 1]")


def _validate_detector(instrument: Mapping[str, Any]) -> None:
    path = "instrument"
    _known_keys(instrument, path, _INSTRUMENT_KEYS)
    if "passband" not in instrument:
        raise _error(path, "missing required key 'passband'")
    for key in ("image_shape", "gain", "read_noise", "pixel_scale", "aperture"):
        if key not in instrument:
            raise _error(path, f"missing required key {key!r}")
    shape = instrument["image_shape"]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape)
    ):
        raise _error("instrument.image_shape", "must contain two positive integers")
    _finite(instrument["gain"], "instrument.gain", positive=True)
    _finite(instrument["read_noise"], "instrument.read_noise", nonnegative=True)
    _finite(instrument["pixel_scale"], "instrument.pixel_scale", positive=True)
    _finite(instrument["aperture"], "instrument.aperture", positive=True)
    obscuration = instrument.get("obscuration", 0.0)
    _finite(obscuration, "instrument.obscuration", nonnegative=True)
    if obscuration >= 1:
        raise _error("instrument.obscuration", "must be less than one")
    qe = instrument.get("quantum_efficiency", 1.0)
    qe = _finite(qe, "instrument.quantum_efficiency", positive=True)
    if qe > 1:
        raise _error("instrument.quantum_efficiency", "must be at most one")
    if "zeropoint" in instrument:
        _finite(instrument["zeropoint"], "instrument.zeropoint")
        if (
            not isinstance(instrument.get("zeropoint_reference"), str)
            or not instrument["zeropoint_reference"].strip()
        ):
            raise _error(
                "instrument.zeropoint_reference",
                "is required when instrument.zeropoint is supplied",
            )
    if "zeropoint_reference" in instrument:
        try:
            _calibration_reference(instrument["zeropoint_reference"])
        except ValueError as exc:
            raise _error("instrument.zeropoint_reference", str(exc)) from exc

    if "distortion" not in instrument:
        raise _error(path, "missing required key 'distortion'")
    distortion = _mapping(instrument["distortion"], "instrument.distortion")
    _known_keys(distortion, "instrument.distortion", _DISTORTION_KEYS)
    for key in ("th", "dthdr"):
        if key not in distortion:
            raise _error("instrument.distortion", f"missing required key {key!r}")
    _series(distortion["th"], "instrument.distortion.th")
    if distortion["th"][0] != 0:
        raise _error("instrument.distortion.th", "must start at zero")
    derivatives = distortion["dthdr"]
    if not isinstance(derivatives, (list, tuple)) or len(derivatives) != len(distortion["th"]):
        raise _error("instrument.distortion.dthdr", "must match distortion.th in shape")
    for index, item in enumerate(derivatives):
        _finite(item, f"instrument.distortion.dthdr[{index}]", positive=True)

    vignetting = instrument.get("vignetting")
    if vignetting is not None:
        vignetting = _mapping(vignetting, "instrument.vignetting")
        _known_keys(vignetting, "instrument.vignetting", _VIGNETTING_KEYS)
        if "th" not in vignetting or "unvig" not in vignetting:
            raise _error("instrument.vignetting", "requires th and unvig")
        _series(vignetting["th"], "instrument.vignetting.th")
        if vignetting["th"][0] != 0:
            raise _error("instrument.vignetting.th", "must start at zero")
        fractions = vignetting["unvig"]
        if not isinstance(fractions, (list, tuple)) or len(fractions) != len(vignetting["th"]):
            raise _error("instrument.vignetting.unvig", "must match vignetting.th in shape")
        _fractions(fractions, "instrument.vignetting.unvig")


def _validate_passband(value: Any, path: str) -> None:
    config = _mapping(value, path)
    _known_keys(config, path, _PASSBAND_KEYS)
    for key in ("name", "wavelength_nm", "throughput", "reference"):
        if key not in config:
            raise _error(path, f"missing required key {key!r}")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise _error(f"{path}.name", "must be a nonempty string")
    wavelength = config["wavelength_nm"]
    if not isinstance(wavelength, (list, tuple)) or len(wavelength) < 2:
        raise _error(f"{path}.wavelength_nm", "must contain at least two values")
    wavelength_values = [
        _finite(item, f"{path}.wavelength_nm[{index}]", positive=True)
        for index, item in enumerate(wavelength)
    ]
    if any(right <= left for left, right in zip(wavelength_values, wavelength_values[1:])):
        raise _error(f"{path}.wavelength_nm", "must be strictly increasing")
    throughput = config["throughput"]
    if not isinstance(throughput, (list, tuple)) or len(throughput) != len(wavelength):
        raise _error(f"{path}.throughput", "must match wavelength_nm in shape")
    throughput_values = []
    for index, item in enumerate(throughput):
        number = _finite(item, f"{path}.throughput[{index}]")
        if not 0 <= number <= 1:
            raise _error(f"{path}.throughput[{index}]", "must be a fraction in [0, 1]")
        throughput_values.append(number)
    integral = sum(
        (right_wavelength - left_wavelength) * (right_flux + left_flux) / 2
        for left_wavelength, right_wavelength, left_flux, right_flux in zip(
            wavelength_values,
            wavelength_values[1:],
            throughput_values,
            throughput_values[1:],
        )
    )
    if integral <= 0:
        raise _error(f"{path}.throughput", "must have a nonzero integral")
    if not isinstance(config["reference"], str) or not config["reference"].strip():
        raise _error(f"{path}.reference", "must be a nonempty string")
    try:
        _calibration_reference(config["reference"])
    except ValueError as exc:
        raise _error(f"{path}.reference", str(exc)) from exc


def _validate_optional_models(config: Mapping[str, Any]) -> None:
    instrument = _mapping(config["instrument"], "instrument")
    _validate_passband(instrument["passband"], "instrument.passband")

    photometry = _mapping(config["photometry"], "photometry")
    _known_keys(photometry, "photometry", _PHOTOMETRY_KEYS)
    if photometry.get("model") not in _PHOTOMETRY_MODELS:
        raise _error("photometry.model", "must be 'constant', 'lambertian', or 'faceted'")
    params = _mapping(photometry.get("params", {}), "photometry.params")
    _validate_photometry_params(photometry["model"], params)
    if "spectral_relative_tolerance" in photometry:
        _finite(
            photometry["spectral_relative_tolerance"],
            "photometry.spectral_relative_tolerance",
            positive=True,
        )
    if "max_spectral_samples" in photometry:
        _integer(
            photometry["max_spectral_samples"],
            "photometry.max_spectral_samples",
            positive=True,
        )
    if photometry["model"] != "constant" and config["sat"]["mag_range"] is not None:
        raise _error("sat.mag_range", "must be null for physical photometry")

    if "integration" in config and config["integration"] not in (None, False):
        integration = _mapping(config["integration"], "integration")
        _known_keys(integration, "integration", _INTEGRATION_KEYS)
        for key in ("pixel_tolerance", "flux_relative_tolerance"):
            if key in integration:
                _finite(integration[key], f"integration.{key}", positive=True)
        if "flux_relative_tolerance" in integration and integration["flux_relative_tolerance"] > 1:
            raise _error("integration.flux_relative_tolerance", "must be at most one")
        if "max_step_s" in integration:
            if integration["max_step_s"] is not None:
                _finite(integration["max_step_s"], "integration.max_step_s", positive=True)
        if "max_depth" in integration:
            _integer(integration["max_depth"], "integration.max_depth", nonnegative=True)
        max_samples = integration.get("max_samples", 4097)
        if "max_samples" in integration:
            _integer(max_samples, "integration.max_samples", positive=True)
        if max_samples < 5:
            raise _error("integration.max_samples", "must be at least five")
        max_step_s = integration.get("max_step_s")
        required_samples = 33
        if max_step_s is not None:
            required_samples = max(
                math.ceil(exptime / max_step_s) * 4 + 1
                for exptime in config["cadence"]["exptime"]
            )
        if max_samples < required_samples:
            raise _error(
                "integration.max_samples",
                f"must be at least {required_samples} for configured cadence",
            )

    if "observation" in config:
        observation = _mapping(config["observation"], "observation")
        _known_keys(observation, "observation", _OBSERVATION_KEYS)
        if observation.get("mode") not in ("geometric", "apparent"):
            raise _error("observation.mode", "must be 'geometric' or 'apparent'")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied, validated run configuration.

    The top-level and known nested sections are strict.  This function never
    mutates its argument.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config: must be a mapping")
    result = copy.deepcopy(dict(config))
    _known_keys(result, "config", _TOP_LEVEL_KEYS)
    if "outdir" not in result:
        raise _error("config", "missing required key 'outdir'")
    for key in _REQUIRED:
        if key not in result:
            raise _error("config", f"missing required key {key!r}")
    if "photometry" not in result:
        raise _error("config", "missing required key 'photometry'")

    _integer(result["n_obs"], "n_obs", positive=True)
    _integer(result["n_example"], "n_example", nonnegative=True)
    if result["n_example"] > result["n_obs"]:
        raise _error("n_example", "must not exceed n_obs")
    _integer(result["n_sat"], "n_sat", positive=True)
    if result["n_sat"] not in (1, 2):
        raise _error("n_sat", "must be 1 or 2")
    _integer(result["seed"], "seed", nonnegative=True)

    _validate_detector(_mapping(result["instrument"], "instrument"))
    conditions = _mapping(result["conditions"], "conditions")
    _known_keys(conditions, "conditions", _CONDITIONS_KEYS)
    for key in ("psf_fwhm_range", "sky_range", "zp_range"):
        if key not in conditions:
            raise _error("conditions", f"missing required key {key!r}")
        _range(conditions[key], f"conditions.{key}", positive=key == "psf_fwhm_range")

    cadence = _mapping(result["cadence"], "cadence")
    _known_keys(cadence, "cadence", _CADENCE_KEYS)
    if "exptime" not in cadence:
        raise _error("cadence", "missing required key 'exptime'")
    exptime = cadence["exptime"]
    if not isinstance(exptime, (list, tuple)) or not exptime:
        raise _error("cadence.exptime", "must be a nonempty list")
    for index, value in enumerate(exptime):
        _finite(value, f"cadence.exptime[{index}]", positive=True)

    tracker = _mapping(result["tracker"], "tracker")
    _known_keys(tracker, "tracker", _TRACKER_KEYS)
    if tracker.get("type") not in ("sidereal", "orbit"):
        raise _error("tracker.type", "must be 'sidereal' or 'orbit'")
    _finite(tracker.get("rot_sky_pos0"), "tracker.rot_sky_pos0")
    errors = _mapping(tracker.get("error"), "tracker.error")
    _known_keys(errors, "tracker.error", _TRACKER_ERROR_KEYS)
    required_errors = (
        ("t_rewind", "v_perturb") if tracker["type"] == "orbit" else ("boresight",)
    )
    for key in required_errors:
        if key not in errors:
            raise _error("tracker.error", f"missing required key {key!r}")
    for key, value in errors.items():
        _finite(value, f"tracker.error.{key}", nonnegative=True)

    sites = result["sites"]
    if not isinstance(sites, (list, tuple)) or not sites:
        raise _error("sites", "must be a nonempty list")
    if any(not isinstance(site, str) or not site.strip() for site in sites):
        raise _error("sites", "must contain nonempty strings")

    sat = _mapping(result["sat"], "sat")
    _known_keys(sat, "sat", _SAT_KEYS)
    if "mag_range" not in sat:
        raise _error("sat", "missing required key 'mag_range'")
    if sat["mag_range"] is not None:
        _range(sat["mag_range"], "sat.mag_range")

    catalog = _mapping(result["catalog"], "catalog")
    _known_keys(catalog, "catalog", _CATALOG_KEYS)
    if "develop" in catalog and not isinstance(catalog["develop"], bool):
        raise _error("catalog.develop", "must be boolean")
    catalog_path = catalog.get("gaia_dir")
    if not catalog.get("develop", False) and (
        not isinstance(catalog_path, (str, os.PathLike)) or not str(catalog_path).strip()
    ):
        raise _error("catalog", "requires a nonempty gaia_dir when develop is false")
    if catalog_path is not None and not isinstance(catalog_path, (str, os.PathLike)):
        raise _error("catalog.path", "must be path-like")
    if "level" in catalog:
        _integer(catalog["level"], "catalog.level", positive=True)
    if "magnitude_column" in catalog and (
        not isinstance(catalog["magnitude_column"], str)
        or not catalog["magnitude_column"].strip()
    ):
        raise _error("catalog.magnitude_column", "must be a nonempty string")
    if "magnitude_system" in catalog and catalog["magnitude_system"] not in (
        "AB",
        "flux_density",
    ):
        raise _error("catalog.magnitude_system", "must be 'AB' or 'flux_density'")
    if "min_snr" not in catalog:
        raise _error("catalog", "missing required key 'min_snr'")
    _finite(catalog["min_snr"], "catalog.min_snr", nonnegative=True)
    if not catalog.get("develop", False):
        for key in ("magnitude_column", "magnitude_system"):
            if key not in catalog:
                raise _error("catalog", f"missing required key {key!r}")

    if not isinstance(result["outdir"], str) or not result["outdir"].strip():
        raise _error("outdir", "must be a nonempty path")
    for key in ("image_dir", "reference_dir"):
        if key in result:
            _validate_relative_directory(result[key], key)
    if "meta" in result:
        result["meta"] = _validate_metadata(result["meta"], "meta")
    _validate_optional_models(result)
    return result


def _is_secret_key(key: Any) -> bool:
    normalized = "".join(character for character in str(key).lower() if character.isalnum())
    return any(part in normalized for part in _SECRET_PARTS)


def _has_unsafe_url(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        return True
    return bool(parsed.query or parsed.fragment)


def _path_digest(value: Any) -> str:
    encoded = os.fspath(value).encode("utf-8", errors="surrogateescape")
    return f"<path:{hashlib.sha256(encoded).hexdigest()[:16]}>"


def _redact(value: Any, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            checked = _redact(item, str(key))
            redacted[str(key)] = "<redacted>" if _is_secret_key(key) else checked
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item, key) for item in value]
    if isinstance(value, Path):
        return _path_digest(value)
    if key in _PROVENANCE_PATH_KEYS and isinstance(value, (str, os.PathLike)):
        return _path_digest(value)
    if isinstance(value, str) and _has_unsafe_url(value):
        raise ValueError(
            "URL userinfo or query/fragment is not allowed in provenance values"
        )
    return value


def _canonical_json(value: Any) -> bytes:
    if isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key != "meta"}
    return json.dumps(
        _redact(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=os.fspath,
    ).encode("utf-8")


def _safe_input_label(path: Path) -> str:
    return f"input{path.suffix.lower()}"


def _input_path_digest(path: Path) -> str:
    return hashlib.sha256(
        os.fspath(path).encode("utf-8", errors="surrogateescape")
    ).hexdigest()[:16]


def _is_credential_input_path(path: Path) -> bool:
    name = path.name.lower()
    if name in {"id_rsa", "id_ed25519"} or name.endswith((".env", ".pem", ".key")):
        return True
    protected_dirs = {".ssh", "credential-export", "credentials", "secrets"}
    if any(part.lower() in protected_dirs for part in path.parts):
        return True
    return any(_is_secret_key(part) for part in path.parts)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_inputs(
    input_paths: Iterable[os.PathLike[str] | str],
) -> list[dict[str, str]]:
    """Hash explicit, resolved input files without retaining raw paths."""
    inputs = []
    seen = set()
    for raw_path in input_paths:
        path = Path(raw_path).resolve(strict=True)
        if path in seen:
            continue
        seen.add(path)
        if _is_credential_input_path(path):
            raise ValueError("credential-like input paths are not allowed in provenance")
        if not path.is_file():
            raise FileNotFoundError(os.fspath(path))
        inputs.append({
            "path": _safe_input_label(path),
            "path_sha256": _input_path_digest(path),
            "sha256": _file_sha256(path),
        })
    return sorted(
        inputs,
        key=lambda item: (item["path"], item["path_sha256"], item["sha256"]),
    )


def _source_provenance() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    modules = {
        os.fspath(module.relative_to(package_root)): _file_sha256(module)
        for module in sorted(package_root.glob("*.py"))
    }
    return {
        "module": "simsit.run_config",
        "source_sha256": _file_sha256(Path(__file__)),
        "module_sha256": modules,
    }


def build_provenance(
    config: Mapping[str, Any],
    input_paths: Iterable[os.PathLike[str] | str] = (),
) -> dict[str, Any]:
    """Build deterministic, non-network provenance for a run."""
    validated = validate_config(config)
    versions: dict[str, str | None] = {}
    for package in _DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    return {
        "schema_version": 1,
        "source": _source_provenance(),
        "dependency_versions": versions,
        "config_sha256": hashlib.sha256(_canonical_json(validated)).hexdigest(),
        "input_sha256": fingerprint_inputs(input_paths),
    }


def write_provenance(
    root: os.PathLike[str] | str,
    provenance: Mapping[str, Any],
    filename: str = "provenance.json",
) -> Path:
    """Atomically write a JSON provenance manifest below an output root."""
    output_root = Path(root)
    if not output_root.is_dir():
        raise NotADirectoryError(os.fspath(output_root))
    if (
        Path(filename).name != filename
        or not filename
        or Path(filename).suffix.lower() != ".json"
    ):
        raise ValueError("provenance filename must be a simple JSON filename")
    target = output_root / filename
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_root,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(provenance, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def write_manifest(
    root: os.PathLike[str] | str,
    provenance: Mapping[str, Any],
    *,
    status: str,
    products: Iterable[os.PathLike[str] | str] = (),
    stale_outputs_retained: bool = False,
    product_records: Iterable[Mapping[str, str]] | None = None,
) -> Path:
    """Write lifecycle state and the current run's product inventory atomically."""
    if status not in {"running", "complete", "failed"}:
        raise ValueError("manifest status must be running, complete, or failed")
    output_root = Path(root)
    if product_records is None:
        product_records = []
        for raw_path in sorted({Path(path) for path in products}, key=os.fspath):
            path = raw_path.resolve(strict=True)
            relative = path.relative_to(output_root.resolve(strict=True))
            if not path.is_file():
                raise ValueError(f"manifest product is not a regular file: {path}")
            product_records.append({
                "path": relative.as_posix(),
                "sha256": _file_sha256(path),
            })
    else:
        product_records = sorted(
            [dict(record) for record in product_records],
            key=lambda record: record["path"],
        )
    payload = dict(provenance)
    payload.update({
        "status": status,
        "products": product_records,
        "stale_outputs_retained": bool(stale_outputs_retained),
    })
    if stale_outputs_retained:
        payload["stale_outputs_note"] = (
            "Files not listed in products were retained and are not current-run products."
        )
    return write_provenance(output_root, payload)


def _contained_output_path(root: Path, relative: Any) -> Path:
    _validate_relative_directory(relative, "output directory")
    root_resolved = root.resolve(strict=False)
    candidate = root / relative
    candidate_resolved = candidate.resolve(strict=False)
    if (
        candidate_resolved != root_resolved
        and root_resolved not in candidate_resolved.parents
    ):
        raise ValueError("output directory symlink escapes output root")
    return candidate


def _reject_existing_symlinks(root: Path) -> None:
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                if entry.is_symlink():
                    raise ValueError(
                        "output root must not contain symlinks, including unrelated files"
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry_path)


def prepare_output(
    root: os.PathLike[str] | str,
    overwrite: bool = False,
    relative_dirs: Iterable[os.PathLike[str] | str] = (),
    *,
    require_empty: bool = True,
    create: bool = True,
) -> Path:
    """Preflight an output root and contained subdirectories safely."""
    path = Path(root)
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("output root must not contain symlinks")
    if path.is_symlink():
        raise ValueError("output root must not be a symlink")
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(os.fspath(path))
    if path.is_dir():
        _reject_existing_symlinks(path)
    relative_paths = []
    for relative in relative_dirs:
        candidate = _contained_output_path(path, relative)
        current = path
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                raise ValueError("output directory must not contain symlinks")
        relative_paths.append(candidate)
    if require_empty and path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"output root is nonempty: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
        for candidate in relative_paths:
            candidate.mkdir(parents=True, exist_ok=True)
    return path
