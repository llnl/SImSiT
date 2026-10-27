===============
Getting Started
===============

This page shows how to generate simulated telescope images with SImSiT.

SImSiT is an image simulation library for stars and satellites. A typical
workflow consists of:

1. choosing or editing a configuration file,
2. running the simulation script, and
3. inspecting the generated image and reference outputs.

Quick start
===========

To generate a basic example dataset, run:

.. code-block:: bash

   python scripts/simulate.py config/sidereal.yaml

This creates output under the directory specified by ``outdir`` in the
configuration file. By default, simulated images are written to an ``images/``
subdirectory and reference products are written to a ``reference/``
subdirectory.

Configuration
=============

Example configuration files are provided in the ``config/`` directory.

Common settings include:

- ``outdir``: output directory for generated files
- ``n_obs``: number of simulated observations
- ``n_example``: number of example/reference records to include in exported
  example metadata products
- ``instrument``: detector and optical model parameters
- ``conditions``: observing conditions such as seeing, sky brightness, and
  zeropoint offsets
- ``catalog``: star catalog configuration
- ``tracker``: boresight and tracking model configuration

Example configurations
======================

Sidereal-tracking example
-------------------------

.. code-block:: bash

   python scripts/simulate.py config/sidereal.yaml

This produces simulated observations using a sidereal-tracking configuration.

Target-tracking example
-----------------------

.. code-block:: bash

   python scripts/simulate.py config/target.yaml

This produces simulated observations using a target-tracking configuration.

Empty-field target-tracking example
-----------------------------------

.. code-block:: bash

   python scripts/simulate.py config/empty_target.yaml

This example can be used to generate target-tracking images without a simulated
satellite source.

Output products
===============

A typical simulation writes:

- ``images/*.fits``: simulated image products
- ``images/sample_metadata_<N>.yaml``: example metadata for a subset of images
- ``images/reference_truth_<N>.fits``: reference truth tables for a subset
- ``reference/*.wcs.fits``: reference WCS at exposure start
- ``reference/*.wcst.fits``: reference WCS at exposure end
- ``reference/sample_metadata.yaml``: metadata for the full simulated set
- ``reference/reference_truth.fits``: reference truth tables for the full set

External catalog data
=====================

Some simulations use locally stored Gaia catalog files. These catalog files are
not distributed with SImSiT and must be obtained separately.

If you are using ``GaiaStarCatalog``, configure:

- ``catalog.gaia_dir``: path to the local Gaia catalog directory

The expected directory layout and required columns should match the format used
by the ``GaiaStarCatalog`` reader in the Python package.

Raw Gaia tables use ``ra``/``dec`` in degrees, ``pmra``/``pmdec`` in
milliarcseconds/year, ``parallax`` in milliarcseconds, and ``ref_epoch`` in
Julian years. Legacy LSST-style trixels use J2000 ``coord_ra``/``coord_dec``
and unit-bearing ``pm_ra``/``pm_dec``/``parallax`` columns. Magnitudes may
come from Gaia's Vega ``phot_g_mean_mag`` (converted to AB), or from
physical flux-density units such as nJy in ``phot_g_mean_flux``. Raw
electron/second flux alone is insufficient without a release-specific
photometric zeropoint. The legacy ``i_mag`` output is an AB Gaia G-band
proxy, not a calibrated i-band color transformation.

Units and reproducibility
=========================

Orbit states use GCRF meters and meters/second; observation times are Astropy
``Time`` objects and FITS dates are TAI. Satellite positions use SSATK's
geometric line of sight without light-time correction. Gaia stellar
positions use the geocentric apparent-coordinate approximation. These
approximations should not be interpreted as precision astrometry.

Source magnitudes and sky surface brightness must be AB quantities in the
same passband as the configured zeropoint. That zeropoint defines one
incident photon/second before vignetting. Expected photon images receive
vignetting first, then quantum efficiency (electrons/photon, default 1).
Shot noise is added once, read noise is in electrons, and gain
(electrons/ADU) converts the final image to ADU. ``nphot`` in truth tables
and ``flux`` in sample metadata remain incident-photon counts.

FITS records the conversion factors and ``PHOTZP``. ``MAGZP`` uses
ADU/second, with deliberate calibration perturbation in the image header
and the true value in the reference headers. Missing satellite magnitudes
are NaN in FITS and null in YAML. The configuration seed controls all
simulation randomness under fixed dependency versions.

Notes
=====

The configuration files in ``config/`` are examples and may reflect particular
instrument or observing assumptions. Review them before using them as templates
for new simulations.
