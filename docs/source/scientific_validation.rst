Independent Scientific Image References
=======================================

This validation is intentionally separate from the feature tests.  It
compares the image renderer with integrals written independently of the
renderer, rather than comparing two calls through the same implementation.
The test sources are ``tests/test_scientific_images.py`` and
``tests/test_scientific_brightness.py``; the reportable results are the
pytest output for those files.

Reference construction
----------------------

The tests use a synthetic Gaia-like ``astropy.table.Table`` containing
right-ascension, declination, and photon-count columns, a
``galsim.GSFitsWCS`` TAN projection with a one arcsec/pixel scale, and a
fixed ``SiderealTracker``.  The PSF is a circular Gaussian with
``sigma = 1 arcsec``.  Each stationary-source pixel is evaluated from the
closed-form Gaussian cumulative distribution function (CDF), using the
error function at the four pixel edges.

The trail reference integrates the product of the two pixel CDF differences
with ``scipy.integrate.quad`` over the analytically specified, constant
velocity pixel path.  The backend sky-coordinate function is monkeypatched
only to supply those analytically constructed world positions.  The actual
renderer, PSF convolution, local WCS, and pixel accumulation remain active;
this isolates image formation and does not test an SSATK propagation oracle.
The path/flux reference helper accepts arbitrary path and flux callables so
that the curved-path and variable-flux interfaces can reuse a correct
reference.  The adaptive test below covers those interfaces through the final
rendering path.

For reference quality, the tests set
``folding_threshold = maxk_threshold = 1e-6``.  They compare both maximum
per-pixel absolute residual and global L1 residual, and require conservation
of the 1000 incident photons.  A supervisor one-off measurement for the
stationary case was maximum absolute residual
``1.1323e-5 photons/pixel`` and relative L1 residual ``1.2194e-7`` at that
setting.  The same comparison with default GalSim settings measured
``0.0241 photons/pixel`` and relative L1 residual ``0.000327``.  These
values are numerical-setting diagnostics, not timing measurements.  The
committed test fixture uses a different subpixel position and, on the
validation environment, measures ``1.765e-4 photons/pixel`` maximum residual
and ``7.21e-6`` relative L1; its explicit limits are set to that reproducible
fixture rather than to the separate one-off probe.

The CCD reference uses a large, deterministic synthetic photon field and
the closed-form moments in electrons/ADU.  For gain :math:`g`, quantum
efficiency :math:`q`, read noise :math:`r`, and incident photon field
:math:`p`, the expected mean is :math:`q E[p]/g`.  The expected spatial
variance is

.. math::

   \operatorname{Var}(q p/g) + (q E[p] + r^2)/g^2.

The seeded draw is checked against both moments, adding coverage beyond a
single deterministic image comparison.

Physical brightness and rendering
---------------------------------

The physical-brightness references use a 5800 K blackbody with unit radius,
unit emissivity, zero albedo, and sun, Earthshine, and moonshine disabled.
The expected incident rate is calculated independently from Planck's law,
the geometric dilution :math:`(R/d)^2`, the collecting area, and a
piecewise-linear synthetic passband response.  The independently evaluated
integrand is

.. math::

   A (R/d)^2 \frac{\pi 2 h c^2}{\lambda^5}
   \left[\exp\left(\frac{h c}{\lambda k T}\right)-1\right]^{-1}
   T_{\rm band}(\lambda) \frac{\lambda}{h c}.

The test then supplies the public ``PhysicalBrightness.photon_rate`` callback
to the public ``draw_sat`` rate interface.  It checks both the exposure
integral and the total rendered incident-photon image.  This is a pipeline
check, not an oracle copied from the physical model: the Planck law and
piecewise-linear response interpolation are implemented independently, while
the package's SSATK radiometry and spectral sampling remain under test.
Quantum efficiency, gain, read noise, atmospheric extinction, phase changes,
attitude, and orbit propagation are intentionally outside this reference.
The fixed sky-coordinate callback isolates image formation and makes no
propagation-accuracy claim.

Adaptive curved-path and flux reference
---------------------------------------

The adaptive reference uses a synthetic quadratic pixel path and a positive
smooth sinusoidal-plus-linear photon rate.  For every pixel it integrates the
Gaussian PSF pixel CDF product with ``scipy.integrate.quad`` over normalized
exposure time, multiplying by the exposure duration to obtain photons.  It
then compares the loose and tight ``draw_sat(integration=...)`` results by
total flux, flux-weighted centroid, and the image L2 norm normalized by the
independent total flux.  Tightening the pixel and flux tolerances must
increase the adaptive sample count and reduce that normalized per-pixel norm.

This is a final-rendering reference: it exercises the adaptive
``draw_sat`` path, not merely ``sample_exposure``.  The sky-coordinate
callback supplies analytically constructed world positions solely to isolate
image formation; it is not an orbit-propagation oracle.  It does not cover
real orbit curvature, attitude,
non-Gaussian PSFs, variable distance, discontinuous rates, and makes no claims
about adaptive behavior outside the stated tolerances.
In the validation run, the loose and tight configurations
used 17 and 129 integration samples, respectively.  Their normalized
per-pixel L2 norms were ``3.98e-3`` and ``5.87e-5``; centroid errors were
``2.32e-2`` and ``3.63e-4`` pixels, with a common relative flux residual of
``1.74e-6`` against the 4500-photon independent reference.  The test requires
the tight normalized L2 norm to be below ``1e-4`` and below one tenth of the
loose norm; the observed ratio is ``1.47e-2``.

Qualification matrix
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Concern
     - Reference in this validation
     - Qualification
   * - Frames
     - TAN sky coordinates
     - Synthetic, with no astrometric frame-orientation claim
   * - Passbands
     - Independent piecewise-linear response integration
     - Tests response integration for a synthetic curve; does not calibrate
       throughput or establish provenance for a real instrument
   * - SSATK gray models
     - Thermal gray radiometry at 5800 K, checked against Planck's law
     - Thermal gray assumptions only; no reflected-light, orbit, attitude, or
       general brightness-model validation
   * - Sampling
     - Exact stationary, constant-velocity, and adaptive synthetic
       curved/variable-flux integrals
     - Adaptive reference is verified through final rendering; real orbit and
       attitude sampling remain outside scope
   * - PSF
     - Analytic Gaussian, 1 arcsec
     - Gaussian reference only
   * - Detector
     - CCD mean/variance in ADU
     - Moment validation, not a detector calibration
   * - Physical brightness
     - Independent Planck-law and response quadrature
     - Does not validate nonthermal spectra, phase-dependent reflection,
       faceted attitude, variable distance, or adaptive spectral convergence
   * - Pipeline rendering
     - Public rate callback and integrated image flux
     - Fixed synthetic position; no orbit or attitude claim

Interpretation and limits
-------------------------

The synthetic positions and photon counts are test inputs, not real
observations, calibrated detections, or a substitute for instrument
characterization.  Passing these tests supports the pixel integration,
constant-trail, photon-conservation, and CCD moment paths under the stated
assumptions.  It does not establish detectability, atmospheric refraction,
catalog completeness, SSATK orbit accuracy, passband throughput, PSF
adequacy for a real instrument, or the validity of a gray satellite
brightness model.  The brightness test additionally does not validate
nonthermal spectra, phase-dependent reflection, faceted attitude, variable
distance, or adaptive spectral convergence beyond the configured tolerance.
All positions, spectra, and photon counts used here are synthetic test inputs,
not real observations.
