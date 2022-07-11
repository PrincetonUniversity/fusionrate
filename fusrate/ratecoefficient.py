from fusrate.constants import amu, keV
from cubature import cubature

import numba

import numpy as np


@numba.njit
def v_th(T_keV, m_amu):
    r"""Jitted (for compatibility) 1 keV thermal velocity"""
    return np.sqrt(keV * T_keV / (m_amu * amu))


@numba.njit
def reduced_mass(m1, m2):
    r"""Reduced mass"""
    μ = m1 * m2 / (m1 + m2)
    return μ


def makef3d(σ, m1, m2):
    r"""Integrand-making function
    Does the integration in Z1 R2-Z2 space, not Z1 ρ2-θ2 space

    It's a bit more natural to express the maxwellian in this space

    Returns a function which can be used at any temperature

    Integration limits:
    Z1: 0 to oo
    R2: 0 to oo
    Z2: -oo to oo

    """

    μ = reduced_mass(m1, m2)
    leading_factor = 2 ** (7 / 2) / np.pi

    @numba.njit
    def com_energy_keVU(v1z, v2r, v2z):
        energy_part = (
            np.square(v1z) + v2r ** 2 - 2 * v1z * v2z + np.square(v2z)
        )
        com_energy = amu * μ * energy_part / (keV)

        return com_energy

    def f(u_array, T1_keV, T2_keV):
        u1z, u2r, u2z = u_array.T

        vth1 = v_th(T1_keV, m1)
        vth2 = v_th(T2_keV, m2)

        maxwellians = np.exp(-np.square(u1z) - np.square(u2r) - np.square(u2z))
        com_e_keV = com_energy_keVU(u1z * vth1, u2r * vth2, u2z * vth2)
        cross_section = σ(com_e_keV)
        jacobian = np.square(u1z) * u2r
        relative_v = np.sqrt(
            np.square(vth2 * u2r) + np.square(vth1 * u1z - vth2 * u2z)
        )

        return (
            leading_factor
            * cross_section
            * relative_v
            * jacobian
            * maxwellians
        )

    return f


class MaxwellianRateCoefficientCalculator:
    def __init__(self, rcore, σ):
        self.rcore = rcore
        self.m_beam = rcore.m_beam
        self.m_tar = rcore.m_tar

        self.f = makef3d(σ, self.m_beam, self.m_tar)
        h = 8
        self.xmin = np.array([0, 0, -h], np.float64)
        self.xmax = np.array([h, h, h], np.float64)
        self.reactivity = np.vectorize(self.reactivity, otypes=['float'])

    def reactivity(self, T_keV):
        val, err = cubature(
            self.f,
            3,
            1,
            self.xmin,
            self.xmax,
            args=(T_keV, T_keV),
            vectorized=True,
            relerr=1e-05,
            maxEval=50000,
            adaptive="h",
        )
        return val


if __name__ == "__main__":
    from fusrate.endf import ENDFCrossSection
    from fusrate.reaction import ReactionCore

    rc = ReactionCore("D+T")
    cs = ENDFCrossSection(rc)
    mwrc = MaxwellianRateCoefficientCalculator(rc, cs.cross_section)
    my_t = 40
    mwrc.reactivity(my_t)
