from cubature import cubature
import numpy as np
import scipy.interpolate

from fusrate.constants import atomic_mass_unit as amu
from fusrate.constants import kiloelectronvolt as keV
from fusrate.constants import millibarn_meters_squared_to_cubic_centimeter
from fusrate.physics import reduced_mass
from fusrate.physics import v_th


# This is my velocity-based implementation
def makef_simplemaxwellian(σ, m1, m2, extramult=1):
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
    leading_factor = extramult * 2 ** (7 / 2) / np.pi

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
            np.square(y1z) + y2r**2 - 2 * y1z * y2z + np.square(y2z)
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
        ratecoeff integrand
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


# This is an energy-based implementation based on Elijah's thesis,
# appendix A.2, Equation (A.17)
# I don't quite understand how it works.
def makef_simplermaxwellian(σ, m1, m2, extramult=1):
    r"""Even better integrand function

    Does the integration in energy space.

    Integration limits:
    U: 0 to ∞

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
    μ = reduced_mass(m1, m2) * amu
    leading_factor = 2 * np.sqrt(2 * keV / (np.pi * μ))
    leading_factor *= extramult

    def f(un_array, T):
        r"""Reactivity integrand

        Parameters
        ----------
        un_array : array_like
            n x 1 array of energies normalized to the temperature
        T : floats
            Temperature in keV. Passed as fixed arguments.
        """
        un = un_array[:, 0]
        maxwellian = np.exp(-un)
        cross_section = σ(un * T)
        jacobian = un

        return leading_factor * cross_section * jacobian * maxwellian

    def x_limits(h):
        r"""Limits function corresponding to the integration strategy

        Parameters
        ----------
        h : number
            multiples of the thermal velocity to integrate out to
        """
        xmin = np.array(
            [
                0,
            ],
            np.float64,
        )
        xmax = np.array(
            [
                h,
            ],
            np.float64,
        )
        return xmin, xmax

    return f, x_limits


def makef_bimaxwellian(σ, m1, m2, extramult=1):
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
    leading_factor = 2 ** (7 / 2) / np.pi ** (2) * extramult

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


# Should find a way to make this return a RCIMaxwellian or BiMaxwellian
# or whatever else, by passing it a Distribution name
class RateCoefficientIntegrator:
    def __init__(
        self, rcore, σ, integrand_maker, relerr, maxeval, h, extramult=None
    ):
        self.rcore = rcore
        self.m_a = rcore.m_beam
        self.m_b = rcore.m_tar

        self.relerr = relerr
        self.maxeval = maxeval
        self.h = h

        self.extramult = extramult

        self.f, xlimits = integrand_maker(σ, self.m_a, self.m_b, extramult)
        self.xmin, self.xmax = xlimits(h)

        self.ratecoeff = np.vectorize(self.ratecoeff, otypes=["float"])


class RateCoefficientIntegratorMaxwellian(RateCoefficientIntegrator):
    r"""Isotropic, single ion temperature, no drifts"""

    def __init__(self, rcore, σ, relerr=1e-6, maxeval=1e4, h=20, extramult=1):
        super().__init__(
            rcore, σ, makef_simplermaxwellian, relerr, maxeval, h, extramult
        )

    def ratecoeff(self, T, **kwargs):
        r"""Rate coefficent

        Parameters
        ----------
        T : array_like,
            Temperatures in keV

        **kwargs :
            Named arguments are ignored.
            Is a dump for parameters like 'derivatives',
            which are not yet supported

        Returns
        -------
        array_like of rate coefficients, cm³/s
        """
        t_factor = T ** (1 / 2)
        val, err = cubature(
            self.f,
            1,
            1,
            self.xmin,
            self.xmax,
            args=(T,),
            vectorized=True,
            relerr=self.relerr,
            maxEval=self.maxeval,
            adaptive="h",
        )
        val *= t_factor
        val_cm3_s = val * millibarn_meters_squared_to_cubic_centimeter
        return val_cm3_s / self.extramult


class RateCoefficientIntegratorBiMaxwellian(RateCoefficientIntegrator):
    r"""Seperate perp and parallel temperatures, no drifts"""

    def __init__(self, rcore, σ, relerr=1e-4, maxeval=1e7, h=8, extramult=1):
        super().__init__(
            rcore, σ, makef_bimaxwellian, relerr, maxeval, h, extramult
        )

    def ratecoeff(self, T_perp, T_par, **kwargs):
        r"""Rate coefficent

        Parameters
        ----------
        T_perp : array_like,
            Temperatures in keV

        T_parallel : array_like,
            Temperatures in keV.

        **kwargs :
            Named arguments are ignored.
            Is a dump for parameters like 'derivatives',
            which are not yet supported

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
        return val_cm3_s / self.extramult

if __name__ == "__main__":
    from fusrate.endf import ENDFCrossSection
    from fusrate.reaction import ReactionCore
    import matplotlib.pyplot as plt
    import inspect

    rc = ReactionCore("D+T")
    cs = ENDFCrossSection(rc)
    my_t = np.logspace(0, 3, 100)

    t1, t2 = np.meshgrid(my_t, my_t)

    mwrc = RateCoefficientIntegratorMaxwellian(
        rc, cs.cross_section, relerr=1e-4, maxeval=5e4, h=10, extramult=1e5
    )
    σv = mwrc.ratecoeff(my_t)

    plt.loglog(my_t, σv)
    plt.show()
