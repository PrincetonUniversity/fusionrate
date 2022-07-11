from boschhale import BoschHaleCrossSection
from boschhale import BoschHaleReactivity

from ion_data import ion_mass

import fusrate.reactionnames as rn


# canonical reaction names


class ReactionCore:
    r"""Basic, cross-section-independent reaction data"""

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
    r"""
    """

    def __init__(self, name):
        # try bh name resolver. If that fails,
        # try pB11 name resolver. If that fails, try
        # # try pLi6
        self.rcore = ReactionCore(name)
        name = self.rcore.canonical_name()

        self.cross_analytic_call = self._no_reactivity_analytic
        if name in BoschHaleCrossSection.provides_reactions():
            self.bh_cross = BoschHaleCrossSection(name)
            self.cross_analytic_call = self.bh_cross.cross_section

        self.reactivity_analytic_call = self._no_reactivity_analytic
        if name in BoschHaleReactivity.provides_reactions():
            self.bh_react = BoschHaleReactivity(name)
            self.reactivity_analytic_call = self.bh_react.reactivity

    def canonical_name(self):
        return self.rcore.canonical_name()

    def reactants(self):
        return self.rcore.reactants()

    def reactant_masses(self):
        return self.rcore.reactant_masses()

    def beam_target_to_com_factor(self):
        return self.rcore.beam_target_to_com_factor()

    def cross_section(self, e):
        return self.cross_analytic_call(e)

    def cross_section_d(self, e):
        pass

    def cross_section_analytic_fit(self, e):
        return self.cross_analytic_call(e)

    def reactivity(self, T):
        pass

    def reactivity_analytic_fit(self, T):
        return self.reactivity_analytic_call(T)

    def reactivity_d(self, T):
        pass

    def _no_reactivity_analytic(self, *T, **kwargs):
        r"""There is no implemented analytic formation to the reactivity.

        Please do not call the function reactivity_analytic, or it will throw
        an error.
        """
        raise NotImplementedError(f"There is no implemented analytic reactivity"
            f" function for {self.canonical_name()}.")


if __name__ == "__main__":
    import numpy as np
    r = Reaction("D+T")
    s = r.cross_section_analytic(np.array([1,2,3]))
    print(s)

    # do the thing
