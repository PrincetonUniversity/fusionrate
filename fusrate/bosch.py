import numpy as np

from fusrate.reactionnames import bosch_name_resolver
from fusrate.reactionnames import DDHE3_NAME
from fusrate.reactionnames import DDT_NAME
from fusrate.reactionnames import DHE3_NAME
from fusrate.reactionnames import DT_NAME


class BoschCrossSection:
    r"""Cross section and derivative for four common reactions

    References
    ----------
    ..[1] Bosch, H.-S.; Hale, G. M.
          Improved Formulas for Fusion Cross-Sections and
          Thermal Reactivities. Nuclear Fusion 1992, 32 (4).
    """
    COEFFICIENTS = {
        DT_NAME: {
            "Bg": 34.3827,
            "a": [
                [6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0],
                [-1.4714e6, 0.0, 0.0, 0.0, 0.0],
            ],
            "b": [
                [6.38e1, -9.95e-1, 6.981e-5, 1.728e-4],
                [-8.4127e-3, 4.7983e-6, -1.0748e-9, 8.5184e-14],
            ],
            "range": [[0.5, 550], [550, 4700]],
            "transition": 530,
        },
        DHE3_NAME: {
            "Bg": 68.7508,
            "a": [
                [5.7501e6, 2.5226e3, 4.5566e1, 0, 0],
                [-8.3993e5, 0.0, 0.0, 0.0, 0.0],
            ],
            "b": [
                [-3.1995e-3, -8.5530e-6, 5.9014e-8, 0],
                [-2.6830e-3, 1.1633e-6, -2.1332e-10, 1.425e-14],
            ],
            "range": [[0.3, 900], [900, 4800]],
            "transition": 900,
        },
        DDT_NAME: {
            "Bg": 31.3970,
            "a": [5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10],
            "b": [0, 0, 0, 0],
            "range": [0.5, 5000],
        },
        DDHE3_NAME: {
            "Bg": 31.3970,
            "a": [5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9],
            "b": [0, 0, 0, 0],
            "range": [0.5, 4900],
        },
    }

    def __init__(self, raw_reaction_name, energy_range="full"):
        self.reaction_name = bosch_name_resolver(raw_reaction_name)
        coeffs = self.COEFFICIENTS[self.reaction_name]
        Bg = coeffs["Bg"]
        a = coeffs["a"]
        b = coeffs["b"]
        has_multiple_ranges = type(a[0]) == list
        if not has_multiple_ranges:
            if energy_range != "full":
                raise ValueError(
                    "This reaction only has one energy range. "
                    "Keyword energy_range should be set to 'full' or left "
                    "unspecified."
                )
            self.calculator = BoschCrossSectionCalc(Bg, a, b)
        elif has_multiple_ranges:
            if energy_range == "full":
                self.calculator = BoschHybridCrossSectionCalc(
                    Bg, a, b, coeffs["transition"]
                )
            elif energy_range == "lower":
                self.calculator = BoschCrossSectionCalc(Bg, a[0], b[0])
            elif energy_range == "upper":
                self.calculator = BoschCrossSectionCalc(Bg, a[1], b[1])
            else:
                raise ValueError(
                    f"Unknown energy range '{energy_range}'; choices are 'full', 'upper', and 'lower'."
                )

    @classmethod
    def provides_reactions(cls):
        r"""List of canonical reaction names"""
        return list(cls.COEFFICIENTS.keys())

    def cross_section(self, e):
        r"""Cross section at some energy

        Parameters
        -----------
        e: array_like,
            keV, energy

        Returns
        -------
        σ : array_like
            mb
        """
        return self.calculator.cross_section(e)

    def derivative(self, e):
        r"""Derivative w.r.t. energy of cross section

        Parameters
        -----------
        e: array_like,
            keV, energy

        Returns
        -------
        dσ_de : array_like
            mb/keV
        """
        return self.calculator.dcrosssection_de(e)

    def canonical_reaction_name(self):
        return self.reaction_name

    @property
    def prescribed_range(self):
        r"""Energy range in keV over which the reaction is valid

        Returns
        -------
        a two-element list [low, high]
        """
        r = self.COEFFICIENTS[self.reaction_name]["range"]
        # for the two reactions with 'upper' and 'lower' ranges
        if type(r[0]) == list:
            r = [r[0][0], r[-1][-1]]
        return r

    @property
    def parameters(self):
        return (("Energy", self.prescribed_range, "keV"))


class BoschRateCoeff:
    r"""Maxwellian-averaged rate coefficients for four common reactions

    References
    ----------
    ..[1] Bosch, H.-S.; Hale, G. M.
          Improved Formulas for Fusion Cross-Sections and
          Thermal Reactivities. Nuclear Fusion 1992, 32 (4).
    """
    COEFFICIENTS = {
        DT_NAME: {
            "Bg": 34.3827,
            "mrc²": 1124656,
            "c": [
                1.17302e-9,
                1.51361e-2,
                7.51886e-2,
                4.60643e-3,
                1.35000e-2,
                -1.06750e-4,
                1.36600e-5,
            ],
            "range": [0.2, 100],
        },
        DHE3_NAME: {
            "Bg": 68.7508,
            "mrc²": 1124572,
            "c": [
                5.51036e-10,
                6.41918e-3,
                -2.02896e-3,
                -1.91080e-5,
                1.35776e-4,
                0,
                0,
            ],
            "range": [0.5, 190],
        },
        DDHE3_NAME: {
            "Bg": 31.3970,
            "mrc²": 937814,
            "c": [5.43360e-12, 5.85778e-3, 7.68222e-3, 0, -2.96400e-6, 0, 0],
            "range": [0.2, 100],
        },
        DDT_NAME: {
            "Bg": 31.3970,
            "mrc²": 937814,
            "c": [5.65718e-12, 3.41267e-3, 1.99167e-3, 0, 1.05060e-5, 0, 0],
            "range": [0.2, 100],
        },
    }

    def __init__(self, raw_reaction_name):
        self.reaction_name = bosch_name_resolver(raw_reaction_name)
        coeffs = self.COEFFICIENTS[self.reaction_name]
        Bg = coeffs["Bg"]
        mrc2 = coeffs["mrc²"]
        c = coeffs["c"]
        self.calculator = BoschRateCoeffCalc(Bg, mrc2, c)

    @classmethod
    def provides_reactions(cls):
        r"""List of canonical names for reactions implemented."""
        return list(cls.COEFFICIENTS.keys())

    def rate_coefficient(self, t):
        r"""Rate coefficient <σv> at some temperature

        Assumes both species are Maxwellian with equal velocity and temperature

        Parameters
        -----------
        t: array_like,
            keV, temperature

        Returns
        -------
        <σv> : array_like
            cm³/s
        """
        return self.calculator.ratecoeff(t)

    def derivative(self, t):
        r"""Derivative of rate coefficient <σv>

        Assumes both species are Maxwellian with equal velocity and temperature

        Parameters
        -----------
        t: array_like,
            keV, temperature

        Returns
        -------
        d<σv>/dTemperature : array_like
            cm³/s/keV
        """
        return self.calculator.dratecoeff_dt(t)

    def canonical_reaction_name(self):
        return self.reaction_name

    @property
    def prescribed_range(self):
        return self.COEFFICIENTS[self.reaction_name]["range"]

    @property
    def parameters(self):
        return (("Temperature", self.prescribed_range, "keV"))


class BoschHybridCrossSectionCalc:
    r"""Separate fits for two energy ranges

    References
    ----------
    ..[1] Bosch, H.-S.; Hale, G. M.
          Improved Formulas for Fusion Cross-Sections and
          Thermal Reactivities. Nuclear Fusion 1992, 32 (4).
    """

    def __init__(self, bg, a, b, transition):
        self.lower_calc = BoschCrossSectionCalc(bg, a[0], b[0])
        self.upper_calc = BoschCrossSectionCalc(bg, a[1], b[1])
        self.transition_energy = transition

    def cross_section(self, e):
        r"""Equation (8)

        Parameters
        ----------
        e : array_like
            keV, c.o.m. energy

        Returns
        -------
        σ: array_like
           cm²
        """
        is_lower = e <= self.transition_energy
        σ_lower = self.lower_calc.cross_section(e)
        σ_upper = self.upper_calc.cross_section(e)
        σ = is_lower * σ_lower + np.bitwise_not(is_lower) * σ_upper
        return σ

    def dcrosssection_de(self, e):
        r"""Equation (8)

        Parameters
        ----------
        e : array_like
            keV, c.o.m. energy

        Returns
        -------
        dσ_de: array_like
           cm² / keV
        """
        is_lower = e <= self.transition_energy
        σ_lower = self.lower_calc.dcrosssection_de(e)
        σ_upper = self.upper_calc.dcrosssection_de(e)
        print(σ_lower)
        print(σ_upper)
        σ = is_lower * σ_lower + np.bitwise_not(is_lower) * σ_upper
        return σ


class BoschCrossSectionCalc:
    r"""Calculates cross sections of the Bosch-Hale type"""

    def __init__(self, bg, a, b):
        self.bg = bg
        self.a1 = a[0]
        self.a2 = a[1]
        self.a3 = a[2]
        self.a4 = a[3]
        self.a5 = a[4]
        self.b1 = b[0]
        self.b2 = b[1]
        self.b3 = b[2]
        self.b4 = b[3]

    def s(self, e):
        r"""Equation (9)

        Parameters
        ----------
        e : array_like
            keV, c.o.m. energy

        Returns
        -------
        "S values": array_like
        """
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        b1 = self.b1
        b2 = self.b2
        b3 = self.b3
        b4 = self.b4
        numer = a1 + e * (a2 + e * (a3 + e * (a4 + e * a5)))
        denom = 1 + e * (b1 + e * (b2 + e * (b3 + e * b4)))
        return numer / denom

    def ds_de(self, e):
        r"""Derivative of Equation (9)

        Parameters
        ----------
        e : array_like
            keV, c.o.m. energy

        Returns
        -------
        dS/de: array_like
            1/keV
        """
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        b1 = self.b1
        b2 = self.b2
        b3 = self.b3
        b4 = self.b4
        numer = a1 + e * (a2 + e * (a3 + e * (a4 + e * a5)))
        denom = 1 + e * (b1 + e * (b2 + e * (b3 + e * b4)))
        ns1 = a3 + e * (a4 + a5 * e)
        ds1 = b2 + e * (b3 + b4 * e)
        term1 = (
            -(b1 + e * ds1 + e * (ds1 + e * (b3 + 2 * b4 * e)))
            * numer
            / denom**2
        )
        term2 = a2 + e * ns1 + e * (ns1 + e * (a4 + 2 * a5 * e)) / denom
        return term1 + term2

    def cross_section(self, e):
        r"""Equation (8)

        Parameters
        ----------
        e : array_like
            keV, c.o.m. energy

        Returns
        -------
        σ: array_like
           cm²
        """

        s = self.s(e)
        return s / (e * np.exp(self.bg / np.sqrt(e)))

    def dcrosssection_de(self, e):
        r"""Derivative of Equation (8)

        Parameters
        ----------
        e : array_like
            keV, c.o.m. energy

        Returns
        -------
        dσ/de: array_like
            cm²/keV
        """
        exp_term = np.exp(-self.bg / np.sqrt(e))
        exp_term * (
            (self.bg - 2 * e ** (1 / 2)) * self.s(e)
            + 2 * e ** (3 / 2) * self.ds_de(e)
        ) / (2 * e ** (5 / 2))
        return np.zeros_like(e)


class BoschRateCoeffCalc:
    r"""Calculates Maxwell-averaged rate coefficient

    Todo: test whether the 'optimizations' are really any faster;
    test other methods (compilation?) for speeding up.

    References
    ----------
    ..[1] Bosch, H.-S.; Hale, G. M.
          Improved Formulas for Fusion Cross-Sections and
          Thermal Reactivities. Nuclear Fusion 1992, 32 (4).
    """

    def __init__(self, bg, mrc2, cd):
        self.bg2 = bg**2
        self.mrc2 = mrc2
        self.c1 = cd[0]
        self.c2 = cd[1]
        self.c3 = cd[2]
        self.c4 = cd[3]
        self.c5 = cd[4]
        self.c6 = cd[5]
        self.c7 = cd[6]
        if self.c6 == 0 and self.c7 == 0:
            if self.c4 == 0:
                self.theta = self.ddfunc
                self.dtheta = self.ddderiv
            else:
                self.theta = self.hefunc
                self.dtheta = self.hederiv
        else:
            self.theta = self.dtfunc
            self.dtheta = self.dtderiv

    def xi(self, θ):
        r"""Equation (14)

        .. math::
           \xi = (B_g^2 / (4 \theta))^{1/3}

        Parameters
        ----------
        θ: array_like
        """
        bg2 = self.bg2
        ξ = (bg2 / (4 * θ)) ** (1 / 3)
        return ξ

    def dxi_dtheta(self, θ):
        r"""Derivative"""
        bg2 = self.bg2
        return -(bg2 ** (1 / 3)) / (3 * 2 ** (2 / 3) * θ ** (4 / 3))

    def dtfunc(self, t):
        r"""Equation (13)

        .. math::
           \theta = T /
               (1 - (T(C2 + T(C4 + T C6)))/(1 + T(C3 + T (C5 + T C7))))

        Parameters
        ----------
        t: array_like
           keV, temperature

        Returns
        -------
        θ: array_like
        """
        numer = t * (self.c2 + t * (self.c4 + t * self.c6))
        denom = 1 + t * (self.c3 + t * (self.c5 + t * self.c7))
        data = t / (1 - numer / denom)
        return data

    def dtderiv(self, t):
        r"""Derivative of Equation (13)"""
        c2 = self.c2
        c3 = self.c3
        c4 = self.c4
        c5 = self.c5
        c6 = self.c6
        c7 = self.c7
        odd_denom = 1 + t * (c3 + t * (c5 + t * c7))
        a = c4 + c6 * t
        b = c2 + t * a
        c = 1 - t * b / odd_denom
        d1 = (
            t
            * b
            * (c3 + t * (c5 + c7 * t) + t * (c5 + 2 * c7 * t))
            / odd_denom**2
        )
        d2 = -t * (c4 + 2 * c6 * t) / odd_denom
        d3 = -b / odd_denom
        return -(t / c**2) * (d1 + d2 + d3) + 1 / c

    def hefunc(self, t):
        r"""Equation (13), specialized

        Parameters
        ----------
        t: array_like
           keV, temperature

        Returns
        -------
        θ: array_like
        """
        numer = t * (self.c2 + t * self.c4)
        denom = 1 + t * (self.c3 + t * self.c5)
        return t / (1 - numer / denom)

    def hederiv(self, t):
        r"""Derivative of Equation (13), specialized"""
        c2 = self.c2
        c3 = self.c3
        c4 = self.c4
        c5 = self.c5
        smdenom = 1 + t * (c3 + c5 * t)
        a = c2 + c4 * t
        b = 1 - t * a / smdenom
        d1 = t * a * (c3 + 2 * c5 * t) / smdenom**2
        d2 = c4 * t / smdenom
        d3 = a / smdenom
        return -(t / b**2) * (d1 - d2 - d3) + 1 / b

    def ddfunc(self, t):
        r"""Equation (13), specialized

        Parameters
        ----------
        t: array_like
           keV, temperature

        Returns
        -------
        θ: array_like
        """
        numer = t * self.c2
        denom = 1 + t * (self.c3 + t * self.c5)
        return t / (1 - numer / denom)

    def ddderiv(self, t):
        r"""Derivative of Equation (13), specialized"""
        c2 = self.c2
        c3 = self.c3
        c5 = self.c5
        smdenom = 1 + t * (c3 + c5 * t)
        a = 1 - c2 * t / smdenom
        d1 = c2 * t * (c3 + 2 * c5 + t) / smdenom**2
        d2 = c2 / smdenom
        return -(t / a**2) * (d1 - d2) + 1 / a

    def ratecoeff(self, t):
        r"""Equation (12)

        Parameters
        ----------
        t: array_like
           keV, temperature

        Returns
        -------
        <σv>: array_like
           cm³/s
        """
        c1 = self.c1
        θ = self.theta(t)
        ξ = self.xi(θ)
        mrc2 = self.mrc2
        root_term = np.sqrt(ξ / (mrc2 * t**3))
        exp_term = np.exp(-3 * ξ)
        return c1 * θ * root_term * exp_term

    def dratecoeff_dt(self, t):
        r"""Derivative of Equation (12)

        Parameters
        ----------
        t: array_like
           keV, temperature

        Returns
        -------
        d<σv>/dt: array_like
           cm³/(s keV)
        """
        c1 = self.c1
        θ = self.theta(t)
        ξ = self.xi(θ)
        mrc2 = self.mrc2
        root_term = np.sqrt(ξ / (mrc2 * t**3))
        exp_term = np.exp(-3 * ξ)
        dθ_dt = self.dtheta(t)
        dξ_dθ = self.dxi_dtheta(θ)
        droot_dξ = 1 / (2 * t ** (3 / 2) * np.sqrt(mrc2 * ξ))
        droot_dt = -(3 / 2) * np.sqrt(ξ / mrc2) * t ** (-5 / 2)
        dexp_dξ = -3 * exp_term

        term1 = dθ_dt * root_term * exp_term
        term2 = θ * (droot_dξ * dξ_dθ * dθ_dt + droot_dt) * exp_term
        term3 = θ * root_term * dexp_dξ * dξ_dθ * dθ_dt
        return c1 * (term1 + term2 + term3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bh = BoschCrossSection("T(d,n)4He")
    energy_range = bh.prescribed_range()
    e1 = np.logspace(*np.log10(energy_range), 500)
    sigma = bh.cross_section(e1)
    plt.loglog(e1, sigma)

    plt.show()
