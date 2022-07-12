from fusrate.constants import amu, keV

from cubature import cubature
import numba
import numpy as np


@numba.njit
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


@numba.njit
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


def makef3d(σ, m1, m2):
    r"""Integrand-making function

    Does the integration in Z1 R2-Z2 space, not Z1 ρ2-θ2 space
    It's a bit more natural to express the maxwellian in this space
    Returns a function which can be used at any temperature

    Integration limits:
    Z1: 0 to oo
    R2: 0 to oo
    Z2: -oo to oo

    Parameters
    ----------
    σ : function
        Cross section function
    m1, m2 : float
        reactant masses in amu

    Returns
    -------
    function f
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

    def f(u_array, vth1, vth2):
        r"""Reactivity integrand

        Parameters
        ----------
        u_array : array_like
            n x 3 array of velocities
        vth1, vth2 : floats
            Thermal velocities in m/s. Passed as fixed arguments.

        Returns
        -------
        reactivity integrand

        Notes
        -----
        We can't @numba.njit this function because the cross section cannot be
        jitted.
        """
        u1z, u2r, u2z = u_array.T

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
    def __init__(self, rcore, σ, relerr=1e-5, maxeval=1e5):
        self.rcore = rcore
        self.m_a = rcore.m_beam
        self.m_b = rcore.m_tar

        self.f = makef3d(σ, self.m_a, self.m_b)
        h = 8
        self.xmin = np.array([0, 0, -h], np.float64)
        self.xmax = np.array([h, h, h], np.float64)
        self.reactivity = np.vectorize(self.reactivity, otypes=["float"])

        self.relerr = relerr
        self.maxeval = maxeval

    def set_relerr(self, relerr):
        self.relerr = relerr

    def set_maxeval(self, maxeval):
        self.maxeval = maxeval

    def reactivity(self, T):
        r"""
        Parameters
        ----------
        T : array_like,
            Temperatures in keV

        Returns
        -------
        array_like of rate coefficients, cm³/s
        """
        barn_meters_squared_to_cubic_centimeter = 1e-22

        vth_a = v_th(T, self.m_a)
        vth_b = v_th(T, self.m_b)

        val, err = cubature(
            self.f,
            3,
            1,
            self.xmin,
            self.xmax,
            args=(vth_a, vth_b),
            vectorized=True,
            relerr=self.relerr,
            maxEval=self.maxeval,
            adaptive="h",
        )
        val_cm3_s = val * barn_meters_squared_to_cubic_centimeter
        return val_cm3_s


if __name__ == "__main__":
    from fusrate.endf import ENDFCrossSection
    from fusrate.reaction import ReactionCore
    import matplotlib.pyplot as plt

    rc = ReactionCore("D+T")
    cs = ENDFCrossSection(rc)
    my_t = 40

    rcs = []

    relerrs = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
    mwrc = MaxwellianRateCoefficientCalculator(
        rc, cs.cross_section, relerr=1e-3, maxeval=1e7
    )
    for rl in relerrs:
        mwrc.set_relerr(rl)
        rcs.append(mwrc.reactivity(my_t))

    rcs = np.array(rcs)

    plt.loglog(relerrs, np.abs(rcs-rcs[-1])/rcs[-1])
    plt.xlabel("Desired accuracy")
    plt.ylabel("Apparent accuracy")
    plt.show()