import numpy as np

from fusrate.constants import atomic_mass_unit as amu
from fusrate.constants import kiloelectronvolt as keV


def v_th(T, m):
    r"""Thermal velocity

    Parameters
    ----------
    T : array_like,
        Temperature in keV
    m : array_like,
        mass in amu

    Returns
    -------
    velocity in m/s
    """
    return np.sqrt(keV * T / (m * amu))


def reduced_mass(m1, m2):
    r"""For two interacting particles

    Parameters
    ----------
    m1, m2 : float

    Returns
    -------
    float
    """
    μ = m1 * m2 / (m1 + m2)
    return μ
