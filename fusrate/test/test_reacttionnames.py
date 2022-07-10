import fusrate.reactionnames as rn
from fusrate.reactionnames import DT_NAME
from fusrate.reactionnames import DHE3_NAME
from fusrate.reactionnames import DDT_NAME
from fusrate.reactionnames import DDHE3_NAME

from fusrate.reactionnames import PLI6_NAME
from fusrate.reactionnames import PB11_NAME

import unittest

class TestReactionNameParsing(unittest.TestCase):
    PROT = 'H'
    DEU = 'D'
    TRIT = 'T'
    HE3 = '³He'
    LI6 = '⁶Li'
    B11 = '¹¹B'

    def test_t0(self):
        assert self.TRIT == rn.reaction_target_species(DT_NAME)

    def test_b0(self):
        assert self.DEU == rn.reaction_beam_species(DT_NAME)

    def test_t1(self):
        assert self.HE3 == rn.reaction_target_species(DHE3_NAME)

    def test_b1(self):
        assert self.DEU == rn.reaction_beam_species(DHE3_NAME)

    def test_t2(self):
        assert self.DEU == rn.reaction_target_species(DDT_NAME)

    def test_b2(self):
        assert self.DEU == rn.reaction_beam_species(DDT_NAME)

    def test_t3(self):
        assert self.DEU == rn.reaction_target_species(DDHE3_NAME)

    def test_b3(self):
        assert self.DEU == rn.reaction_beam_species(DDHE3_NAME)

    def test_t4(self):
        assert self.LI6 == rn.reaction_target_species(PLI6_NAME)

    def test_b4(self):
        assert self.PROT == rn.reaction_beam_species(PLI6_NAME)

    def test_t5(self):
        assert self.LI6 == rn.reaction_target_species(PLI6_NAME)

    def test_b5(self):
        assert self.PROT == rn.reaction_beam_species(PLI6_NAME)

    def test_t6(self):
        assert self.B11 == rn.reaction_target_species(PB11_NAME)

    def test_b6(self):
        assert self.PROT == rn.reaction_beam_species(PB11_NAME)

class TestBoschHaleReactionNameResolver(unittest.TestCase):
    def test_DT_0(self):
        assert DT_NAME == rn.bosch_hale_name_resolver(DT_NAME)

    def test_DT_1(self):
        assert DT_NAME == rn.bosch_hale_name_resolver("DT")

    def test_DT_2(self):
        assert DT_NAME == rn.bosch_hale_name_resolver("D-T")

    def test_DT_3(self):
        assert DT_NAME == rn.bosch_hale_name_resolver("D+T")

    def test_DT_4(self):
        assert DT_NAME == rn.bosch_hale_name_resolver("D+T→n+α")

    def test_DT_5(self):
        assert DT_NAME == rn.bosch_hale_name_resolver("D+T→α+n")

    def test_DT_6(self):
        assert DT_NAME == rn.bosch_hale_name_resolver("D+T→a+n")

    def test_DT_7(self):
        assert DT_NAME == rn.bosch_hale_name_resolver("D+T→n+a")

    def test_DH3_0(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver(DHE3_NAME)

    def test_DH3_1(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("DHe3")

    def test_DH3_2(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("DHe")

    def test_DH3_3(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("D3He")

    def test_DH3_4(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("D+3He")

    def test_DH3_5(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("D+³He")

    def test_DH3_6(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("D+³He→p+⁴He")

    def test_DH3_7(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("D+³He→p+α")

    def test_DH3_8(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("D+³He→α+p")

    def test_DH3_9(self):
        assert DHE3_NAME == rn.bosch_hale_name_resolver("D+³He->a+p")

    def test_DDT_0(self):
        assert DDT_NAME == rn.bosch_hale_name_resolver(DDT_NAME)

    def test_DDT_1(self):
        assert DDT_NAME == rn.bosch_hale_name_resolver("D+D→p+T")

    def test_DDT_2(self):
        assert DDT_NAME == rn.bosch_hale_name_resolver("D+D→T+p")

    def test_DDT_3(self):
        assert DDT_NAME == rn.bosch_hale_name_resolver("²H+²H→³H+¹H")

    def test_DDT_4(self):
        assert DDT_NAME == rn.bosch_hale_name_resolver("²H+²H→¹H+³H")

    def test_DDHe3_0(self):
        assert DDHE3_NAME == rn.bosch_hale_name_resolver(DDHE3_NAME)

    def test_DDHe3_1(self):
        assert DDHE3_NAME == rn.bosch_hale_name_resolver("D(d,n)3He")

    def test_DDHe3_2(self):
        assert DDHE3_NAME == rn.bosch_hale_name_resolver("D+D→n+3He")

    def test_DDHe3_3(self):
        assert DDHE3_NAME == rn.bosch_hale_name_resolver("²H+²H→n+3He")

    def test_DDHe3_4(self):
        assert DDHE3_NAME == rn.bosch_hale_name_resolver("²H+²H→3He+n")

    def test_bad_name(self):
        self.assertRaises(ValueError, rn.bosch_hale_name_resolver, "bad")


if __name__ == "__main__":
    unittest.main()
