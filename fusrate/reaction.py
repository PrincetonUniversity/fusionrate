import fusrate.reactionnames as rn
from fusrate.bosch import BoschCrossSection
from fusrate.bosch import BoschReactivity
from fusrate.endf import ENDFCrossSection
from fusrate.ion_data import ion_mass
from fusrate.ratecoefficient import RateCoefficientIntegratorMaxwellian


class ReactionCore:
    r"""Basic, cross-section-independent reaction data

    Beam and Target refer to the formally specified reactants in the canonical
    name, Target(Beam, product1)product2.
    This is important because ENDF cross sections are given in a frame where
    the target is in the lab frame, and we want to convert to the
    center-of-mass frame.

    I'm not sure if product1 and product2 have specific names.
    """

    def __init__(self, name):
        self.name = rn.name_resolver(name)

        self.beam, self.target = rn.reactants(self.name)

        self.m_beam = ion_mass(self.beam)
        self.m_tar = ion_mass(self.target)

        self.bt_to_com = self.m_tar / (self.m_beam + self.m_tar)

    def canonical_name(self):
        return self.name

    def reactants(self):
        return self.beam, self.target

    def reactant_masses(self):
        return self.m_beam, self.m_tar

    def beam_target_to_com_factor(self):
        return self.bt_to_com


class Reaction:
    r"""Main reaction class"""

    def __init__(self, name):
        self.rcore = ReactionCore(name)
        name = self.rcore.canonical_name()

        self.cross_analytic_call = self._no_cross_analytic
        self.reactivity_analytic_call = self._no_reactivity_analytic

        # the Bosch paper is the only source of analytic fits

        self.has_analytic_fit = name in BoschCrossSection.provides_reactions()
        if self.has_analytic_fit:
            self.bh_cross = BoschCrossSection(name)
            self.cross_analytic_call = self.bh_cross.cross_section
            self.bh_react = BoschReactivity(name)
            self.reactivity_analytic_call = self.bh_react.reactivity

        self.cross_sec_interpolator = ENDFCrossSection(name).cross_section

        self.reactivity_integrator = RateCoefficientIntegratorMaxwellian(
            self.rcore, self.cross_sec_interpolator
        ).reactivity

    def canonical_name(self):
        return self.rcore.canonical_name()

    def reactants(self):
        return self.rcore.reactants()

    def reactant_masses(self):
        return self.rcore.reactant_masses()

    def beam_target_to_com_factor(self):
        return self.rcore.beam_target_to_com_factor()

    ### cross section functions

    def cross_section(self, e):
        r"""Look up the interpolated cross section from ENDF data

        Parameters
        ----------
        e : array_like,
          energies in keV

        Returns
        -------
        Cross sections in mb
        """
        return self.cross_sec_interpolator(e)

    def cross_section_d(self, e):
        pass

    def cross_section_analytic_fit(self, e):
        r"""Fit from Bosch-Hale

        Parameters
        ----------
        e : array_like,
          energies in keV

        Returns
        -------
        Cross sections in mb
        """
        return self.cross_analytic_call(e)

    def _no_cross_analytic(self, *T, **kwargs):
        r"""There is no implemented analytic formation to the reactivity.

        Please do not call the function reactivity_analytic, or it will throw
        an error.
        """
        raise NotImplementedError(
            f"There is no implemented analytic reactivity"
            f" function for {self.canonical_name()}."
        )

    ### reactivity functions

    def reactivity(self, T):
        pass

    def reactivity_integration(self, T):
        r"""Perform an integration"""
        return self.reactivity_integrator(T)

    def reactivity_analytic_fit(self, T):
        r"""Use the analytic fit"""
        return self.reactivity_analytic_call(T)

    def reactivity_d(self, T):
        pass

    def _no_reactivity_analytic(self, *T, **kwargs):
        r"""There is no implemented analytic formation to the reactivity.

        Please do not call the function reactivity_analytic, or it will throw
        an error.
        """
        raise NotImplementedError(
            f"There is no implemented analytic reactivity"
            f" function for {self.canonical_name()}."
        )


if __name__ == "__main__":
    import numpy as np

    r = Reaction("D+T")
    ts = np.array([10, 20, 30])
    s = r.reactivity_integration(ts)
    print(s)
    s = r.reactivity_analytic_fit(ts)
    print(s)
