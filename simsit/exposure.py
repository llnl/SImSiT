"""Bounded adaptive temporal integration for moving point sources.

The result is a sequence of line segments.  ``offsets_s`` and
``positions_px`` contain the segment boundaries, while ``segment_photons``
contains the photon integral for each interval.  This deliberately does not
render an image.

The estimator is an adaptive sampling heuristic, not a rigorous guarantee for
unresolved discontinuities or arbitrarily narrow spikes.  Use an explicit
``max_step_s`` (and suitable callback regularity) when such features matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import astropy.units as u
import numpy as np
from astropy.time import Time, TimeDelta

from .backend import as_time

__all__ = ["ExposureIntegration", "sample_exposure"]


@dataclass(frozen=True)
class ExposureIntegration:
    """Immutable adaptive exposure samples and photon weights."""

    offsets_s: np.ndarray
    positions_px: np.ndarray
    segment_photons: np.ndarray
    total_photons: float
    estimated_flux_error: float
    evaluated_sample_count: int


@dataclass(frozen=True)
class _Sample:
    position: np.ndarray
    rate: float


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ma.getmaskarray(value).any():
        raise ValueError(f"{name} must be an unmasked numeric scalar.")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if array.ndim != 0 or not np.isfinite(array.item()):
        raise ValueError(f"{name} must be a finite scalar.")
    return float(array.item())


def _position(value: object) -> np.ndarray:
    if np.ma.getmaskarray(value).any():
        raise ValueError("position samples must not be masked.")
    try:
        if isinstance(value, u.Quantity):
            value = value.to_value(u.pix)
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("position callback must return two finite pixels.") from exc
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError("position callback must return two finite pixels.")
    return array.copy()


def _rate(value: object) -> float:
    if np.ma.getmaskarray(value).any():
        raise ValueError("rate samples must not be masked.")
    if isinstance(value, u.Quantity):
        try:
            value = value.to_value(1 / u.s)
        except (TypeError, ValueError, u.UnitConversionError) as exc:
            raise ValueError("rate callback Quantity must have units of 1/s.") from exc
    result = _finite_scalar(value, "rate")
    if result < 0:
        raise ValueError("rate callback must return a non-negative finite rate.")
    return result


def _simpson(values: tuple[float, float, float]) -> float:
    return (values[0] + 4.0 * values[1] + values[2]) / 6.0


def _seconds(value, name):
    if isinstance(value, TimeDelta):
        value = value.sec
    elif isinstance(value, u.Quantity):
        value = value.to_value(u.s)
    return _finite_scalar(value, name)


def _immutable(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def sample_exposure(
    position_callable: Callable[[Time], object],
    rate_callable: Callable[[Time], object],
    t0: Time,
    exposure_s: float,
    *,
    pixel_tolerance: float = 0.01,
    flux_relative_tolerance: float = 1e-4,
    max_step_s: float | None = None,
    max_depth: int = 12,
    max_samples: int = 4097,
) -> ExposureIntegration:
    """Adaptively sample one exposure using bounded temporal subdivision.

    Callbacks receive scalar :class:`astropy.time.Time` values.  A rate may be
    a scalar in photons/s or an Astropy quantity convertible to ``1/s``.
    Simpson estimates are used for photon weights.  ``estimated_flux_error``
    is the sum of absolute coarse/fine Simpson differences in photons.
    """
    if not callable(position_callable) or not callable(rate_callable):
        raise TypeError("position_callable and rate_callable must be callable.")
    try:
        epoch = as_time(t0)
    except (TypeError, ValueError) as exc:
        raise ValueError("t0 must be a finite scalar Astropy Time.") from exc
    if not isinstance(epoch, Time) or not epoch.isscalar:
        raise ValueError("t0 must be a finite scalar Astropy Time.")
    duration = _seconds(exposure_s, "exposure_s")
    if duration <= 0:
        raise ValueError("exposure_s must be positive.")
    pixel_tol = _finite_scalar(pixel_tolerance, "pixel_tolerance")
    flux_tol = _finite_scalar(flux_relative_tolerance, "flux_relative_tolerance")
    if pixel_tol <= 0 or not 0 < flux_tol <= 1:
        raise ValueError("pixel tolerance must be positive and relative flux tolerance in (0, 1].")
    if max_step_s is None:
        max_step = duration / 8.0
    else:
        max_step = _seconds(max_step_s, "max_step_s")
        if max_step <= 0:
            raise ValueError("max_step_s must be positive.")
    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)):
        raise ValueError("max_depth must be a non-negative integer.")
    if max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")
    if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)):
        raise ValueError("max_samples must be at least five.")
    if max_samples < 5:
        raise ValueError("max_samples must be at least five.")

    cache: dict[float, _Sample] = {}

    def evaluate(offset: float) -> _Sample:
        # Exact binary subdivision offsets make shared nodes cacheable without
        # altering the boundary offsets returned to the caller.
        key = float(offset)
        if key not in cache:
            if len(cache) >= max_samples:
                raise RuntimeError(
                    "adaptive exposure integration exceeded max_samples before "
                    "meeting tolerances"
                )
            time = epoch + key * u.s
            if not time.isscalar or not np.isfinite(time.gps):
                raise ValueError("callback evaluation time must be finite and scalar.")
            sample = _Sample(_position(position_callable(time)), _rate(rate_callable(time)))
            cache[key] = sample
        return cache[key]

    def interval(a: float, b: float, depth: int) -> tuple[list[tuple[float, float, float]], float]:
        h = b - a
        q1, mid, q3 = a + h / 4.0, a + h / 2.0, a + 3.0 * h / 4.0
        sa, sq1, smid, sq3, sb = (
            evaluate(a), evaluate(q1), evaluate(mid), evaluate(q3), evaluate(b)
        )
        coarse = h * _simpson((sa.rate, smid.rate, sb.rate))
        fine = h * (sa.rate + 4.0 * sq1.rate + 2.0 * smid.rate + 4.0 * sq3.rate + sb.rate) / 12.0
        if not np.isfinite(coarse) or not np.isfinite(fine):
            raise ValueError("Photon integrals overflowed; reduce the rate or exposure duration.")
        flux_error = abs(fine - coarse)

        chord = sb.position - sa.position
        fractions = (0.25, 0.5, 0.75)
        departure = max(
            np.linalg.norm(sample.position - (sa.position + fraction * chord))
            for sample, fraction in zip((sq1, smid, sq3), fractions)
        )
        midpoint = (sa.position + sb.position) / 2
        samples = (sa, sq1, smid, sq3, sb)
        weights = h / 12 * np.array([1, 4, 2, 4, 1]) * np.array([s.rate for s in samples])
        if fine > 0:
            positions = np.array([s.position for s in samples])
            centroid = np.sum(weights[:, None] * positions, axis=0) / fine
            centroid_shift = np.linalg.norm(centroid - midpoint)
            second_moment = np.sum(weights * np.sum((positions - midpoint)**2, axis=1)) / fine
            width_shift = np.sqrt(abs(second_moment - np.dot(chord, chord) / 12))
        else:
            centroid_shift = width_shift = 0.0
        local_budget = max(abs(fine), abs(coarse), np.finfo(float).tiny)
        acceptable = (
            departure <= pixel_tol
            and centroid_shift <= pixel_tol
            and width_shift <= pixel_tol
            and flux_error <= flux_tol * local_budget
        )
        if acceptable:
            return [(a, b, fine)], flux_error
        if depth >= max_depth:
            raise RuntimeError(
                "adaptive exposure integration reached max_depth before meeting tolerances"
            )
        left, left_error = interval(a, mid, depth + 1)
        right, right_error = interval(mid, b, depth + 1)
        return left + right, left_error + right_error

    seed_ratio = duration / max_step
    if not np.isfinite(seed_ratio) or np.ceil(seed_ratio) * 4 + 1 > max_samples:
        raise RuntimeError("max_step_s requires more seed samples than max_samples.")
    count = int(np.ceil(seed_ratio))
    seeds = np.linspace(0.0, duration, count + 1)
    segments: list[tuple[float, float, float]] = []
    error = 0.0
    for start, end in zip(seeds[:-1], seeds[1:]):
        part, part_error = interval(float(start), float(end), 0)
        segments.extend(part)
        error += part_error

    offsets = np.array([segment[0] for segment in segments] + [segments[-1][1]], dtype=float)
    positions = np.array([evaluate(offset).position for offset in offsets], dtype=float)
    photons = np.array([segment[2] for segment in segments], dtype=float)
    total = float(photons.sum())
    if not np.isfinite(total) or total < 0 or not np.isfinite(error):
        raise RuntimeError("adaptive exposure integration produced invalid photon totals.")
    if error > flux_tol * max(abs(total), np.finfo(float).tiny):
        raise RuntimeError(
            "adaptive exposure integration failed its global flux error budget"
        )
    return ExposureIntegration(
        _immutable(offsets), _immutable(positions), _immutable(photons),
        total, float(error), len(cache)
    )
