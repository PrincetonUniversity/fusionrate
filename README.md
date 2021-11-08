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

I'm still thinking about the API; this will change.

```
>>> from fusrate import reaction
>>> dtreaction = reaction("D+T")
>>> ratecoeff = dtreaction.maxwellian(datasource='BoschHale')
>>> t_k = 900 # Kelvin
>>> ratecoeff(t_k) # in m³/s
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
