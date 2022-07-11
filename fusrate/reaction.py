from boschhale import BoschHaleCrossSection
from boschhale import BoschHaleReactivity

import fusrate.reactionnames as rn
# canonical reaction names


class Reaction:
    def __init__(self, name):
        # try bh name resolver. If that fails,
        # try pB11 name resolver. If that fails, try
        # # try pLi6
        self.name = rn.name_resolver(name)

        self.beam, self.target = rn.reactants(name)

        self.m_beam = ion_mass(self.beam)
        self.m_tar = ion_mass(self.target)

        self.bt_to_com = self.beam_target_to_com_factor()

    def canonical_name(self):
        return self.name

    def reactants(self):
        return self.beam, self.target

    def reactant_masses(self):
        return self.m_beam, self.m_tar

    def beam_target_to_com_factor(self):
        return self.m_tar / (self.m_beam + self.m_tar)

    def cross_section(self, e):
        pass

    def reactivity(self, T):
        pass


if __name__ == "__main__":
    pass
    # do the thing
