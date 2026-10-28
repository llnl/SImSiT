# SImSiT

[![Documentation Status](https://readthedocs.org/projects/simsit/badge/?version=latest)](https://simsit.readthedocs.io/en/latest/)

SImSiT (Satellite Image Simulation Toolkit) is a Python package for generating
synthetic telescope images of stars and satellites. It provides configurable
models for instrument response, world coordinate systems, tracking behavior,
observation cadence, and catalog-based source generation.

Built on top of [GalSim](https://github.com/GalSim-developers/GalSim) and
[SSATK](https://github.com/LLNL/ssatk), SImSiT can be used to create realistic
simulated image data for developing and testing algorithms related to satellite
imaging, detection, photometry, calibration, and related analysis workflows.

Documentation, installation instructions, and usage examples are available at:
[https://simsit.readthedocs.io](https://simsit.readthedocs.io)

![Example SImSiT output](./docs/source/example_out.png)

Example simulated output from SImSiT for low-Earth-orbit satellite observations.
Left: sidereal tracking, where the satellite appears streaked.
Right: target tracking, where stars appear streaked and the tracked satellite
appears nearly point-like.

## External catalog data

Some simulation workflows use locally stored Gaia catalog files. These catalogs
are not distributed with SImSiT and must be obtained separately. See the
documentation for expected file layout and configuration options.

## Simulation Conventions

SImSiT is the image-generation layer. SSATK supplies orbit propagation,
observer geometry, constants, and the astrodynamics interface. Its shared
geometry and brightness models can feed this layer without duplicating PSF,
WCS, or detector rendering. Python 3.10 or newer is required.

- Orbit states are in GCRF, in meters and meters/second. Observation times are
  explicit Astropy `Time` objects; FITS dates use TAI. Bare cadence durations
  are seconds. Generated orbit heights refer to a WGS72 spherical radius,
  not height above terrain or the WGS84 ellipsoid.
- The `simsit.backend` boundary uses SSATK exclusively; quantities convert to
  SI and numeric epochs explicitly mean GPS seconds. Satellite directions
  default to geometric lines of sight; `observation.mode: apparent` enables
  iterative light-time and observer-velocity aberration for both tracking
  and rendering. `FRAME` and `OBSMODE` are recorded in FITS headers.
  Gaia stars use geocentric apparent-coordinate approximations.
  Catalog `mag` values identify their actual passband and magnitude system;
  raw Gaia G is never substituted for i or another response.
- Magnitudes and surface brightness passed to the renderer are AB values in
  the same passband as the supplied zeropoint. The zeropoint is the magnitude
  producing one incident photon/second before vignetting. Random satellite
  magnitudes are synthetic inputs, not a physical brightness prediction.
- Rendering accumulates expected incident photons, applies vignetting, then
  converts photons to electrons with `quantum_efficiency` (default 1).
  Shot noise is sampled once; Gaussian read noise is in electrons. Division
  by `gain` (electrons/ADU) produces the final image in ADU.
- FITS images record `BUNIT`, `GAIN`, `RDNOISE`, `QE`, `EXPTIME`, `MAGSYS`,
  and the incident-photon zeropoint `PHOTZP`. `MAGZP` is in ADU/second and
  intentionally perturbed in the delivered image header; reference WCS
  headers retain its true value. Truth `nphot` and sample metadata `flux`
  remain pre-vignetting incident-photon counts. Missing satellite magnitudes
  are NaN in FITS truth and null in YAML.
- A configuration seed reproduces the simulation under the same dependency
  versions. Cadence returns only exposures contained in the usable pass.

## Calibrated Passbands

Simulation requires `instrument.passband` with a name, positive increasing
`wavelength_nm` samples, throughput fractions, and a calibration `reference`.
Curves are supplied by the caller; the example top-hats are explicitly
synthetic, not measured instrument calibrations.

The photon-weighted AB convention uses the exact -48.60 zero point, matching
Astropy's `ABmag`. `simsit.photometry.spectrum_to_magnitude` folds a
unit-bearing sampled spectrum through the response without extrapolation.
`Instrument.photon_rate` and `spectral_photon_rate` use that same response
and unobscured collecting area. An empirical photon zeropoint additionally
requires `zeropoint_reference`; it acts as bounded achromatic transmission.
Quantum efficiency and gain are applied later, exactly once.

With real catalogs, set `catalog.magnitude_column` to a matching AB magnitude
column, or select a physical band-mean flux density with
`magnitude_system: flux_density`. Input FITS metadata must contain
`PASSBAND`, the exact `RESPHASH`, `MAGSYS: AB`, and `MAGCOL`.
`synthetic_photometry` copies a supplied spectral catalog, adds a new AB
column and these tags without altering native measurements.
Single Gaia G measurements cannot determine other-band fluxes without
additional calibrated data. Native Gaia queries retain their actual Vega/AB
system and are not automatically eligible for rendering.

Truth magnitudes now use `mag`, not the misleading `i_mag` field. FITS
products record `FILTER`, `RESPHASH`, `BANDREF` and `ZPREF` alongside the
photon/electron/ADU contract.

## Satellite Photometry

Every configuration selects `photometry.model`: `constant`, `lambertian`,
or `faceted`. `constant` retains synthetic AB magnitudes, including the
`mag_range: null` empty-target option. Physical models require explicit
SSATK parameters:

```yaml
sat:
  mag_range: null
photometry:
  model: lambertian
  params:
    radius_m: 1.0
    albedo: 0.2
    include_earthshine: false
    include_moonshine: false
```

Physical rates use generated-orbit GCRF states at each arrival epoch and fold
the public SSATK gray Lambertian/faceted model through the configured response.
Fixed exposure segments use Simpson integration of cached scalar rates. Truth
stores integrated incident-photon `nphot` and exposure-mean `mean_ab_mag`;
detector QE and gain are applied only during rendering.

SSATK radiometry assumes gray Lambertian reflection and blackbody source
spectra. It is not a calibrated spacecraft spectrum, specular-glint model, or
relativistic flux model. Apparent observation mode changes direction through
SSATK light-time and aberration handling; radiometry remains the geometric
model evaluated at the explicit arrival state. Facet geometry, attitude, and
material coefficients are caller supplied.

## Reproducible Run Lifecycle

Run a validated configuration with:

```bash
simsit-run config/sidereal.yaml
simsit-run config/sidereal.yaml --overwrite
```

Configuration validation, site resolution, and non-development Gaia-directory
checks complete before the output root is created. Output directories must be
relative and contained, and existing nonempty roots are refused unless
`--overwrite` is explicit. Before any output write, all existing entries below
the root are scanned without following directory links; any symlink, including
an unrelated link, causes refusal. This assumes a stable, owned output
directory and does not claim adversarial concurrent-filesystem protection.
Overwrite never deletes unrelated or stale files.

Each run writes an atomic `provenance.json` manifest. It records source-module
hashes, dependency versions, explicit input hashes, lifecycle `status`
(`running`, `complete`, or `failed`), and only products written by the current
run. Gaia provenance records the individual FITS trixels actually read, not
the entire catalog directory. Files retained from earlier runs are documented
as stale and are intentionally omitted from the current product inventory.
Inputs are hashed once when consumed and products once after successful write;
intermediate and final manifest updates reuse those cached hashes.

The legacy `make_sky_flat(configfile, *, overwrite=False)` helper is separate
from this full `simulate`/`simsit-run` provenance lifecycle. It requires
explicit overwrite for an existing `sky_flat.fits`, applies the strict
no-symlink output policy, permits unrelated parent files, and does not replace
the parent run manifest.

## Validation

Run the offline reference and end-to-end tests with
`python -m pytest -m "not integration"`. Live Gaia queries are separately
selected with `python -m pytest -m integration`; they require network access.
GitHub CI tests the offline suite on Python 3.10 and 3.12 and also installs
and tests the built wheel outside the source checkout.

## License

SImSiT is distributed under the terms of the MIT license. See
[LICENSE](https://github.com/LLNL/simsit/blob/main/LICENSE) and
[NOTICE](https://github.com/LLNL/simsit/blob/main/NOTICE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-2009738
