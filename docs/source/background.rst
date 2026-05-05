==========
Background
==========

Summary
-------

SImSiT (Satellite Image Simulation Toolkit) is a Python package for generating
synthetic telescope images containing stars and satellites. It is built on top
of GalSim (:cite:t:`Rowe2015`) and SSAPy (:cite:t:`SSAPy2023`; :cite:t:`Yeager2023`)
and is intended to support the development and testing of algorithms for
satellite imaging, detection, calibration, and related analysis tasks.

SImSiT supports multiple observing configurations, including sidereal tracking
(where stars remain approximately point-like and fast-moving satellites appear
streaked) and target tracking (where the tracked satellite appears nearly
point-like and background stars may appear streaked). By enabling realistic
simulation of telescope, detector, and tracking effects, SImSiT helps users
study satellite observations under a range of observing conditions.

SImSiT has also been used in studies of closely separated satellites in
ground-based astronomical images (:cite:t:`Pruett2023`).

Statement of need
-----------------

The astronomy community has developed many image simulation tools, including
mission- or telescope-specific packages such as
`Romanisim <https://romanisim.readthedocs.io/en/latest/>`_, which generates
high-fidelity images for the Nancy Grace Roman Space Telescope
(:cite:t:`Spergel2015`), as well as end-to-end simulation frameworks for
instruments such as the Vera C. Rubin Observatory LSSTCam
(:cite:t:`Peng2013`).

More general packages also exist for astronomical scene generation and optical
simulation, including GalSim (:cite:t:`Rowe2015`), SkyMaker
(:cite:t:`Bertin2009`), and ray-tracing frameworks such as PhoSim
(:cite:t:`Peterson2015`). Separately, software packages have been developed for
modeling satellite observational characteristics (:cite:t:`Fankhauser2023`).

SImSiT is intended to bridge these domains by combining realistic astronomical
image simulation with satellite orbit-based motion and observation geometry in
a single package.

With the growth of large astronomical imaging surveys such as the Vera C. Rubin
Observatory LSST (:cite:t:`Ivezi2019`) and the Zwicky Transient Facility
(:cite:t:`Bellm2019`), along with the increasing number of visible satellites
(:cite:t:`Mroz2022`), realistic simulation of satellite-containing images has
become increasingly important for both astronomy and space-domain applications.

Method
------

SImSiT combines GalSim-based image rendering with SSAPy-based orbital and
observational geometry calculations.

GalSim enables users to configure observing parameters such as detector
properties, optical distortion, vignetting, and point-spread functions.
These models can be used either to approximate existing instruments or to
explore prospective observing configurations.

SSAPy provides orbit propagation and related geometry calculations, allowing
users to simulate satellites in a range of orbit classes and observing
scenarios.

SImSiT can generate star fields from Gaia-based catalog inputs and produce
simulated image products together with associated reference metadata such as
WCS information and source tables.