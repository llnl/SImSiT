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

   simsit-run config/sidereal.yaml

This creates output under the directory specified by ``outdir`` in the
configuration file. By default, simulated images are written to an ``images/``
subdirectory and reference products are written to a ``reference/``
subdirectory. From a source checkout that has not installed the console
script, the equivalent command is:

.. code-block:: bash

   PYTHONPATH=. python -m simsit.cli config/sidereal.yaml

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
- ``photometry.model``: required satellite source model, ``constant``,
  ``lambertian``, or ``faceted``

Example configurations
======================

Sidereal-tracking example
-------------------------

.. code-block:: bash

   simsit-run config/sidereal.yaml

This produces simulated observations using a sidereal-tracking configuration.

Target-tracking example
-----------------------

.. code-block:: bash

   simsit-run config/target.yaml

This produces simulated observations using a target-tracking configuration.

Empty-field target-tracking example
-----------------------------------

.. code-block:: bash

   simsit-run config/empty_target.yaml

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
and unit-bearing ``pm_ra``/``pm_dec``/``parallax`` columns.
Native Gaia queries retain Vega ``phot_g_mean_mag`` or AB from physical
flux-density units such as nJy. Raw electron/second flux alone is
insufficient without a release-specific photometric zeropoint.
The output is named ``mag`` with explicit ``MAGSYS`` and ``PASSBAND``;
G is not an i-band proxy and native queries are not calibrated render inputs.

Named response curves
=====================

Simulation requires an explicit ``instrument.passband``:

.. code-block:: yaml

   passband:
     name: synthetic_optical
     wavelength_nm: [400.0, 800.0]
     throughput: [1.0, 1.0]
     reference: Synthetic top-hat example, not a measured response

For scientific use, supply a measured response and its provenance instead.
The provided examples use mock catalogs and these explicitly synthetic curves.
Zeropoints are derived from the response and unobscured collecting area.
Optional empirical ``zeropoint`` values require ``zeropoint_reference`` and
cannot make total transmission exceed one. Conditions offsets describe
additional achromatic transmission changes, not a change of passband.

``simsit.photometry.spectrum_to_magnitude`` integrates physical
wavelength/flux-density quantities over the response. The complete response
domain must be covered; no stellar spectrum or color is guessed. It uses the
photon-weighted AB system and exact -48.60 zero point.

To prepare an astrometric catalog that also contains unit-bearing spectral
vector columns:

.. code-block:: python

   from simsit.photometry import synthetic_photometry

   calibrated = synthetic_photometry(
       spectral_catalog, passband,
       wavelength_column="wavelength", flux_column="flux",
       magnitude_column="band_mag",
   )

This returns a copy with the new column and ``PASSBAND``, ``RESPHASH``,
``MAGSYS`` and ``MAGCOL`` metadata. Original photometry is preserved.
Store the calibrated trixels separately and configure
``catalog.magnitude_column: band_mag``. All four metadata values must match
the selected response. Band-mean physical flux-density columns can instead
use ``catalog.magnitude_system: flux_density``; raw Gaia G is not used for
another band. Response identity and calibration references are recorded
in ``FILTER``, ``RESPHASH``, ``BANDREF`` and ``ZPREF`` FITS keywords.

Satellite photometry
====================

Select the satellite source model explicitly in every configuration. Constant
photometry retains the synthetic AB-magnitude behavior:

.. code-block:: yaml

   sat:
     mag_range: [12.0, 15.0]
   photometry:
     model: constant

The physical Lambertian model requires explicit geometry and material
parameters:

.. code-block:: yaml

   sat:
     mag_range: null
   photometry:
     model: lambertian
     params:
       radius_m: 1.0
       albedo: 0.2
       include_earthshine: false
       include_moonshine: false
     spectral_relative_tolerance: 1.0e-4
     max_spectral_samples: 257

Physical models evaluate public SSATK radiometry from generated-orbit GCRF
states at each arrival epoch. Fixed exposure segments use Simpson integration
of cached scalar photon-rate callbacks. Truth tables record integrated
incident ``nphot`` and exposure-mean ``mean_ab_mag``. The bounded spectral
estimate uses SSATK gray Lambertian/faceted reflection and blackbody source
assumptions; it does not infer spacecraft geometry or material properties.
In apparent observation mode, SSATK changes direction using light-time and
aberration, while radiometry remains the geometric model evaluated at the
explicit arrival state.

Run lifecycle and provenance
============================

The CLI validates the complete configuration before creating the output root.
It resolves all configured observing sites and, for a non-development Gaia
catalog, checks that ``catalog.gaia_dir`` exists and is a directory before
writing anything.

Output roots are non-destructive by default. A nonempty root is rejected:

.. code-block:: bash

   simsit-run config/sidereal.yaml --overwrite

``--overwrite`` is explicit and overwrites products with the same current-run
names without deleting unrelated files. Relative ``image_dir`` and
``reference_dir`` values must remain contained in ``outdir``. Before any
directory or manifest write, every existing entry beneath ``outdir`` is
checked without following directory links; any symlink, including an
unrelated, dangling, file, or directory link, causes refusal. This assumes
the output directory is stable and owned by the run process; concurrent
filesystem changes are outside the guarantee.

Every started run writes an atomic ``provenance.json`` manifest. Its lifecycle
status is ``running``, ``complete``, or ``failed``. A complete manifest lists
only files written by that run and includes their hashes. A failed manifest
records products written before the failure. Existing files retained during an
overwrite are documented as stale and are not included in the current product
inventory.

Provenance includes source-module hashes, dependency versions, the
configuration digest, and explicit input hashes. The CLI fingerprints its YAML
configuration and, for non-development Gaia runs, only the FITS trixels
actually read. It does not hash an entire potentially large catalog directory.
Input labels are sanitized; raw personal filesystem paths are not stored in the
manifest. Each input is hashed once when first consumed, and each product is
hashed once after its successful write. Intermediate ``running`` updates and
the final lifecycle state reuse those cached records rather than rehashing all
earlier files.

Limits
======

The manifest is a reproducibility and file-integrity record, not a scientific
validation report. Matching hashes and dependency versions do not guarantee
identical results across operating systems, hardware, numerical libraries, or
external site/catalog changes. Retained stale files are not deleted and must
be removed or archived separately when a clean output directory is required.

The example curves and mock catalogs are synthetic. Physical satellite
photometry uses the configured public SSATK gray models and caller-supplied
geometry/material parameters; it is not a calibrated spacecraft spectrum or a
detectability claim. Real Gaia rendering still requires calibrated passband
metadata and compatible magnitude columns.

Sky-flat helper
===============

The legacy ``make_sky_flat(configfile, *, overwrite=False)`` helper is separate
from the full ``simulate``/``simsit-run`` provenance lifecycle. It refuses to
replace an existing ``sky_flat.fits`` unless ``overwrite=True`` and never
creates or replaces a parent-run provenance manifest. It applies the same
strict no-symlink output policy while allowing unrelated existing files in the
parent run directory.

Units and reproducibility
=========================

Orbit states use GCRF meters and meters/second; observation times are Astropy
``Time`` objects and FITS dates are TAI. The ``simsit.backend`` module is
the sole astrodynamics boundary. Numeric epochs there are explicitly GPS
seconds, not implicit calendar dates; quantity-valued states are converted
to SI and non-GCRF states must be transformed before constructing an orbit.
Satellite positions default to geometric lines of sight. Set
``observation.mode: apparent`` to use iterative light-time and
observer-velocity aberration consistently for satellite tracking and
rendering. ``FRAME`` and ``OBSMODE`` record these conventions in image
and truth-WCS headers. Neither mode adds atmospheric refraction. Gaia stellar
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
