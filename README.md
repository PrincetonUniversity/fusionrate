[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

fusionrate
==========

This python module contains data and functions to calculate and quickly look up fusion reaction rate coefficients.

It has data on

- Cross sections
- Rate coefficients

and their derivatives, as functions of temperature, for the major fusion-power relevant reactions.

Usage
=====

I'm still working on the API; this may change.

```
>>> from fusionrate import Reaction
>>> dt = Reaction("D+T")
>>> temperature = 10 # keV
>>> cs = dt.cross_section(temperature)  # millibarns

# Maxwellian rate coefficient
>>> rc = dt.rate_coefficient(temperature)  # in cm³/s
>>> temperatures = np.logspace(-2,4) # keV
>>> rc = dt.rate_coefficient(temperatures)  # in cm³/s

# To get derivatives of rate coefficients, in  cm³/s/keV
>>> rc_derivs = dtreaction.rate_coefficient(temperature, derivatives=True)
```
Also see the example scripts provided.

Installation
============

Eventually this package should be able to be installed from `PyPi`, using pip:

`pip install fusionrate`

Citing
======

If you use this package in your research, please cite it (via Zenodo, eventually).

Licensing
=========

Like its parent project, FAROES, this package is released under the MIT license.
If you're interested in using this package under a different license, let's talk.
