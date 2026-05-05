# SImSiT

[![Documentation Status](https://readthedocs.org/projects/simsit/badge/?version=latest)](https://simsit.readthedocs.io/en/latest/)

SImSiT (Satellite Image Simulation Toolkit) is a Python package for generating
synthetic telescope images of stars and satellites. It provides configurable
models for instrument response, world coordinate systems, tracking behavior,
observation cadence, and catalog-based source generation.

Built on top of [GalSim](https://github.com/GalSim-developers/GalSim) and
[SSAPy](https://github.com/LLNL/SSAPy), SImSiT can be used to create realistic
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

## License

SImSiT is distributed under the terms of the MIT license. See
[LICENSE](https://github.com/LLNL/simsit/blob/main/LICENSE) and
[NOTICE](https://github.com/LLNL/simsit/blob/main/NOTICE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-2009738