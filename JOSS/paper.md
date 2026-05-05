---
title: 'SImSiT: The Satellite Image Simulation Toolkit'
tags:
  - Python
  - astronomy
  - satellites
  - space domain awareness
authors:
  - name: LLNL SImSiT team
    affiliation: 1
  - name: Peter McGill
    orcid: 0000-0002-1052-6749
    corresponding: true
    affiliation: 1
  - name: Denvir Higgins
    affiliation: 3
    orcid: 0000-0002-7579-1092
  - name: Travis Yeager
    affiliation: 3
    orcid: 0000-0002-2582-0190
  
affiliations:
 - name: Space Science Institute, Lawrence Livermore National Laboratory, 7000 East Ave., Livermore, CA 94550, USA
   index: 1
  
date: 17 October 2024
bibliography: paper.bib
  
aas-doi:
aas-journal:
---
  
# Summary
  
The Satellite Image Simulation Toolkit (SImSiT) is a Python software package designed
to generate diverse and realistic satellite imaging scenarios. SImSiT is built on top
of GalSim [@Rowe2015] and SSAPy [@SSAPy2023;@Yeager2023] and serves as a toolkit for
simulating imaging data that supports the development and testing of algorithms for
satellite detection, calibration, and characterization. SImSiT provides a suite of
simulation tools that allow users to replicate various satellite observation conditions,
including: Sidereal tracking (where the sensor tracks fixed stars, causing the satellite
to appear as a streak in the image), and Target tracking (where the sensor follows the
satellite during exposure), as shown in Figure 1. By enabling the creation of scenarios
that mimic real-world satellite operations, SImSiT facilitates advancements in
satellite image data processing and the study of satellite behavior under different
observational parameters. SImSiT has been used to benchmark and test machine
learning algorithms that aim to identify closely separated satellites in
ground-based astronomical images [@Pruett2023].

# Statement of need
  
The astronomy community has built many specialized image simulation tools. These include tools specific
to a given mission or telescope, such as Roman-I-Sim[^1], which generates high-fidelity images for the
upcoming Nancy Grace Roman Space Telescope [@Spergel2015], as well as end-to-end image simulations
of the ground-based Vera C. Rubin Observatory's LSSTCam [e.g.,@Peng2013].
The community has also built more general software tools that can simulate detailed telescope designs
using ray-tracing methods—PhoSim[@Peterson2015]—and generate general astronomical scenes, for example,
GalSim [@Rowe2015] and SkyMaker [@Bertin2009]. Moreover, software packages that model detailed observational
characteristics of satellites have also been developed [@Fankhauser2023]. However, no software packages that
combine general image simulations of satellites with realistic orbits currently exist—SImSiT fills this gap.
  
With the advent of large astronomical imaging surveys such as LSST [@Ivezi2019] and The Zwicky Transient
Facility [@Bellm2019] in the coming years, and the proliferation of satellites visible in those surveys [@Mroz2022],
simulating images with satellites has become increasingly important for the broad astronomy and space communities.
  
  
[^1]: [https://romanisim.readthedocs.io/en/latest/](https://romanisim.readthedocs.io/en/latest/)
  
![Example of simulated image output from SImSiT of satellites in low-Earth orbit observed in the optical i-band filter using a ground-based sensor:
Left: An example image taken with sidereal (star) tracking, where the fast-moving satellite appears as a streak across the image.
Right: An example image taken with satellite tracking, where the stars appear as streaks, and the satellite appears as a point source.](example_out.png)
  
# Method
  
SImSiT combines the functionality of GalSim and SSAPy. GalSim enables users to customize a wide
range of observing parameters, such as detector settings, optical system properties, and realistic
atmospheric conditions. Specifically, users can manually configure sensor parameters, such as vignetting
and optical distortion, based on real and characterized sensors or simulate and test future sensor designs.
  
The functionality provided by SSAPy enables users to simulate the trajectories of satellites in a wide
variety of realistic orbits, such as low-Earth or geostationary orbits. Users can also use this
functionality to simulate arbitrary satellite maneuvers between sensor exposures.
  
SImSiT generates realistic astronomical background scenes using star positions,
fluxes, and colors from the Gaia Data Release 2 catalog [@GaiaCollaboration2018].
SImSiT also provides tools to convert Gaia optical magnitudes into other passbands,
such as Short Wave Infrared (SWIR), to generate realistic astronomical scenes
in different wavelengths. SImSiT models satellites as a Lambertian sphere with
spherically symmetric emissivity. Users can specify the brightness of a satellite in a
given photometric filter, its size (radius), and albedo.
  
# Acknowledgements
  
`SImSiT` depends on numpy [@Harris2020], scipy [@Virtanen2020], matplotlib [@Hunter2007], SSAPy [@SSAPy2023], and
pandas [@Pandas2010].
This work was performed under the auspices of the U.S.
Department of Energy by Lawrence Livermore National
Laboratory (LLNL) under Contract DE-AC52-07NA27344. The document number is LLNL-JRNL-XXXX and the code number is LLNL-CODE-XXXX.
`SImSiT` was developed under LLNL’s Laboratory Directed Research and Development Program under projects 19-SI-004 and 22-ERD-054.
  
# References