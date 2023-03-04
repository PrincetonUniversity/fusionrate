[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

fusionrate
==========

This python module contains data and functions to calculate and quickly look up fusion reaction rate coefficients.

It has data on

- Cross sections
- Rate coefficients

and their derivatives, as functions of temperature, for the major fusion-power relevant reactions.

This package is a part of the [FAROES](https://github.com/Plasmacontrol/FAROES) project, and is released separately for convenience.

Usage
=====

Note that the API is still in alpha.

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
>>> rc_derivs = dt.rate_coefficient(temperature, derivatives=True)
```
Also see the example scripts provided.

Installation
============

For now, install manually using pip:

`pip install -e fusionrate`.

Eventually this package should be able to be installed from `PyPi`, using pip:

`pip install fusionrate`

Handling of unreasonable numerical inputs
=========================================
The package tries to handle broken or unreasonable inputs silently, in a reasonable way, rather than crashing. 
Negative or infinite energies or temperatures resolve to NaN.

Zero energies or temperatures should return zero cross sections and rate coefficients.
However, the derivatives with respect to energies or temperatures may be small but nonzero,
in order to help a gradient-based optimizer realize that positive temperatures increase fusion rates.

Citing
======

If you use this package in your research, please cite it (via Zenodo, eventually).

Licensing
=========

Like its parent project, FAROES, this package is released under the MIT license.

Funding
=======

This code is developed using funding from the Department of Energy,
including Contract No. DE-AC02-09CH11466.

The developer grants to the Government, and others acting on its behalf,
a nonexclusive, paid-up, irrevocable, world-wide license in such copyrighted data to reproduce,
prepare derivative works, distribute copies to the public,
and perform publicly and display publicly, by or on behalf of the Government.
