[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

FusRate
=======

This python module contains data and functions to calculate fusion reaction rate coefficients.

It has data on

- Cross sections
- Rate coefficients

and their derivatives, as functions of temperature, for the major fusion-power relevant reactions.

Usage
=====

I'm still thinking about the API; this may change.

```
>>> from fusrate import Reaction
>>> dt = Reaction("D+T")
>>> temperature = 10 # keV
>>> cs = dt.cross_section(temperature)  # millibarns
>>> rc = dt.rate_coefficient(temperature)  # in m³/s
>>> temperatures = np.logspace(-2,4) # keV
>>> rc = dt.rate_coefficient(temperatures)  # in m³/s

# To get derivatives of rate coefficients, in  m³/s/keV
>>> rc_derivs = dtreaction.rate_coefficient(temperature, derivatives=True)
```
Also see the example scripts provided.

Installation
============

Eventually this package should be able to be installed from `PyPi`, using pip:

`pip install fusrate`

Citing
======

If you use this package in your research, please cite it (via Zenodo; link above).

Licensing
=========

If you're interested in using this package under a different license, let's talk.
