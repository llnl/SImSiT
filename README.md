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
- Satellite directions use geometric lines of sight without light-time
  correction. Gaia stars use geocentric apparent-coordinate approximations.
  The legacy `i_mag` catalog field is AB Gaia G as an i-band proxy, not a
  calibrated i-band color transformation.
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
