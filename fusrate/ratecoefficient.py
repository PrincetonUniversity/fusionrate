from fusrate.constants import atomic_mass_unit as amu
from fusrate.constants import kiloelectronvolt as keV
from fusrate.constants import millibarn_meters_squared_to_cubic_centimeter

from cubature import cubature
import numba
import numpy as np


@numba.njit(cache=True)
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


@numba.njit(cache=True)
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


def makef_simplemaxwellian(σ, m1, m2):
    r"""Integrand-making function

    Does the integration in Z1 R2-Z2 space.
    It's a bit more natural to express the maxwellian in this space and there
    is no direct trigometry.

    Integration limits:
    Z1: 0 to ∞
    R2: 0 to ∞
    Z2: -∞ to ∞

    Parameters
    ----------
    σ : function
        Cross section function, which takes as a single argument energy in keV
        and returns the cross section in millibarns
    m1, m2 : float
        reactant masses in amu

    Returns
    -------
    functions f, x_limits
    """

    μ = reduced_mass(m1, m2)
    leading_factor = 2 ** (7 / 2) / np.pi

    @numba.njit(cache=True)
    def com_energy_keV(y1z, y2r, y2z):
        r"""Center of mass energy

        Parameters
        ----------
        y's: velocities, in m/s, but normalized by √2

        Returns
        -------
        Energy in keV
        """
        energy_part = (
            np.square(y1z) + y2r ** 2 - 2 * y1z * y2z + np.square(y2z)
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
        com_e = com_energy_keV(u1z * vth1, u2r * vth2, u2z * vth2)
        cross_section = σ(com_e)
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

    def x_limits(h):
        r"""Limits function corresponding to the integration strategy

        Parameters
        ----------
        h : number
            multiples of the thermal velocity to integrate out to
        """
        xmin = np.array([0, 0, -h], np.float64)
        xmax = np.array([h, h, h], np.float64)
        return xmin, xmax

    return f, x_limits


def makef_bimaxwellian(σ, m1, m2):
    r"""Integrand-making function for T⊥ ≠ T‖

    Parameters
    ----------
    σ : function
        Cross section function, which takes as a single argument energy in keV
        and returns the cross section in millibarns
    m1, m2 : float
        reactant masses in amu

    Returns
    -------
    functions f, x_limits
    """
    μ = reduced_mass(m1, m2)

    @numba.njit(cache=True)
    def sq_norm_rel_v(y1r, y1z, y2x, y2y, y2z):
        r"""Energy-like term

        Parameters
        ----------
        y's: velocities, in m/s, but normalized by √2

        Returns
        -------
        v_rel² / 2, in m²/s²
        """
        return np.square(y1r - y2x) + np.square(y2y) + np.square(y1z - y2z)

    @numba.njit(cache=True)
    def com_energy_keV(squared_normalized_relative_velocity):
        r"""Center of mass energy

        Parameters
        ----------
        squared_normalized_relative_velocity : array_like,
            v_rel² / 2, in m²/s²

        Returns
        -------
        Energy in keV
        """
        com_energy = amu * μ * squared_normalized_relative_velocity / (keV)
        return com_energy

    def f(u, vth1_perp, vth1_par, vth2_perp, vth2_par):
        u1r, u1z, u2x, u2y, u2z = u.T

        leading_factor = 2 ** (7 / 2) / np.pi ** (2)

        maxwellfactor = np.exp(-np.sum(np.square(u), axis=1))
        jacobian = u1r
        squared_normalized_relative_velocity = sq_norm_rel_v(
            u1r * vth1_perp,
            u1z * vth1_par,
            u2x * vth2_perp,
            u2y * vth2_perp,
            u2z * vth2_par,
        )
        com_e = com_energy_keV(squared_normalized_relative_velocity)
        cross_section = σ(com_e)
        relative_v = np.sqrt(squared_normalized_relative_velocity)
        return (
            leading_factor
            * cross_section
            * maxwellfactor
            * relative_v
            * jacobian
        )

    def x_limits(h):
        r"""Limits function corresponding to the integration strategy

        Parameters
        ----------
        h : number
            multiples of the thermal velocity, normalized by √2,
            to integrate out to.

        Examples
        -------
        h = 5 gives integration limits of ±5/√2 v_th
        """
        xmin = np.array([0, 0, -h, 0, -h], np.float64)
        xmax = np.array([h, h, h, h, h], np.float64)
        return xmin, xmax

    return f, x_limits


class RateCoefficientIntegrator:
    def __init__(self, rcore, σ, integrand_maker, relerr, maxeval, h):
        self.rcore = rcore
        self.m_a = rcore.m_beam
        self.m_b = rcore.m_tar

        self.relerr = relerr
        self.maxeval = maxeval
        self.h = h

        self.f, xlimits = integrand_maker(σ, self.m_a, self.m_b)
        self.xmin, self.xmax = xlimits(h)

        self.reactivity = np.vectorize(self.reactivity, otypes=["float"])


class RateCoefficientIntegratorBiMaxwellian(RateCoefficientIntegrator):
    r"""Seperate perp and parallel temperatures, no drifts"""

    def __init__(self, rcore, σ, relerr=1e-4, maxeval=1e7, h=8):
        super().__init__(rcore, σ, makef_bimaxwellian, relerr, maxeval, h)

    def reactivity(self, T_perp, T_par):
        r"""Rate coefficent

        Parameters
        ----------
        T_perp : array_like,
            Temperatures in keV

        T_parallel : array_like,
            Temperatures in keV.

        Returns
        -------
        array_like of rate coefficients, cm³/s
        """
        vth_a_perp = v_th(T_perp, self.m_a)
        vth_b_perp = v_th(T_perp, self.m_b)

        vth_a_par = v_th(T_par, self.m_a)
        vth_b_par = v_th(T_par, self.m_b)

        val, err = cubature(
            self.f,
            5,
            1,
            self.xmin,
            self.xmax,
            args=(vth_a_perp, vth_a_par, vth_b_perp, vth_b_par),
            vectorized=True,
            relerr=self.relerr,
            maxEval=self.maxeval,
            adaptive="h",
        )
        val_cm3_s = val * millibarn_meters_squared_to_cubic_centimeter
        return val_cm3_s


class RateCoefficientIntegratorMaxwellian(RateCoefficientIntegrator):
    r"""Isotropic, single ion temperature, no drifts"""

    def __init__(self, rcore, σ, relerr=1e-5, maxeval=1e5, h=8):
        super().__init__(rcore, σ, makef_simplemaxwellian, relerr, maxeval, h)

    def reactivity(self, T):
        r"""Rate coefficent

        Parameters
        ----------
        T : array_like,
            Temperatures in keV

        Returns
        -------
        array_like of rate coefficients, cm³/s
        """

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
        val_cm3_s = val * millibarn_meters_squared_to_cubic_centimeter
        return val_cm3_s


if __name__ == "__main__":
    from fusrate.endf import ENDFCrossSection
    from fusrate.reaction import ReactionCore
    import matplotlib.pyplot as plt

    rc = ReactionCore("D+T")
    cs = ENDFCrossSection(rc)
    my_t = np.logspace(0, 4, 20)

    mwrc = RateCoefficientIntegratorMaxwellian(
        rc, cs.cross_section, relerr=1e-4, maxeval=5e4, h=8
    )
    σv = mwrc.reactivity(my_t)

    mwrc2 = RateCoefficientIntegratorBiMaxwellian(
        rc, cs.cross_section, relerr=1e-5, maxeval=3e7, h=8
    )
    σv2 = mwrc2.reactivity(my_t, my_t)

    # plt.loglog(my_t, σv)
    # plt.loglog(my_t, σv2)
    # plt.show()

    plt.loglog(my_t, σv / σv2)
    plt.show()
