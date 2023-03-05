import fusionrate.reactionnames as rn
from fusionrate.reactionnames import DT_NAME
from fusionrate.reactionnames import DHE3_NAME
from fusionrate.reactionnames import DDT_NAME
from fusionrate.reactionnames import DDHE3_NAME

from fusionrate.reactionnames import PLI6_NAME
from fusionrate.reactionnames import PB11_NAME

import unittest


class TestReactionNameParsing(unittest.TestCase):
    PROT = 'H'
    DEU = 'D'
    TRIT = 'T'
    HE3 = '³He'
    LI6 = '⁶Li'
    B11 = '¹¹B'

    def test_t0(self):
        assert self.TRIT == rn.target_species(DT_NAME)

    def test_b0(self):
        assert self.DEU == rn.beam_species(DT_NAME)

    def test_r0(self):
        assert self.DEU, self.TRIT == rn.reactants(DT_NAME)

    def test_t1(self):
        assert self.HE3 == rn.target_species(DHE3_NAME)

    def test_b1(self):
        assert self.DEU == rn.beam_species(DHE3_NAME)

    def test_t2(self):
        assert self.DEU == rn.target_species(DDT_NAME)

    def test_b2(self):
        assert self.DEU == rn.beam_species(DDT_NAME)

    def test_t3(self):
        assert self.DEU == rn.target_species(DDHE3_NAME)

    def test_b3(self):
        assert self.DEU == rn.beam_species(DDHE3_NAME)

    def test_t4(self):
        assert self.LI6 == rn.target_species(PLI6_NAME)

    def test_b4(self):
        assert self.PROT == rn.beam_species(PLI6_NAME)

    def test_t5(self):
        assert self.LI6 == rn.target_species(PLI6_NAME)

    def test_b5(self):
        assert self.PROT == rn.beam_species(PLI6_NAME)

    def test_t6(self):
        assert self.B11 == rn.target_species(PB11_NAME)

    def test_b6(self):
        assert self.PROT == rn.beam_species(PB11_NAME)


class TestBoschReactionNameResolver(unittest.TestCase):
    def test_DT_0(self):
        assert DT_NAME == rn.bosch_name_resolver(DT_NAME)

    def test_DT_1(self):
        assert DT_NAME == rn.bosch_name_resolver("DT")

    def test_DT_2(self):
        assert DT_NAME == rn.bosch_name_resolver("D-T")

    def test_DT_3(self):
        assert DT_NAME == rn.bosch_name_resolver("D+T")

    def test_DT_4(self):
        assert DT_NAME == rn.bosch_name_resolver("D+T→n+α")

    def test_DT_5(self):
        assert DT_NAME == rn.bosch_name_resolver("D+T→α+n")

    def test_DT_6(self):
        assert DT_NAME == rn.bosch_name_resolver("D+T→a+n")

    def test_DT_7(self):
        assert DT_NAME == rn.bosch_name_resolver("D+T→n+a")

    def test_DH3_0(self):
        assert DHE3_NAME == rn.bosch_name_resolver(DHE3_NAME)

    def test_DH3_1(self):
        assert DHE3_NAME == rn.bosch_name_resolver("DHe3")

    def test_DH3_2(self):
        assert DHE3_NAME == rn.bosch_name_resolver("DHe")

    def test_DH3_3(self):
        assert DHE3_NAME == rn.bosch_name_resolver("D3He")

    def test_DH3_4(self):
        assert DHE3_NAME == rn.bosch_name_resolver("D+3He")

    def test_DH3_5(self):
        assert DHE3_NAME == rn.bosch_name_resolver("D+³He")

    def test_DH3_6(self):
        assert DHE3_NAME == rn.bosch_name_resolver("D+³He→p+⁴He")

    def test_DH3_7(self):
        assert DHE3_NAME == rn.bosch_name_resolver("D+³He→p+α")

    def test_DH3_8(self):
        assert DHE3_NAME == rn.bosch_name_resolver("D+³He→α+p")

    def test_DH3_9(self):
        assert DHE3_NAME == rn.bosch_name_resolver("D+³He->a+p")

    def test_DDT_0(self):
        assert DDT_NAME == rn.bosch_name_resolver(DDT_NAME)

    def test_DDT_1(self):
        assert DDT_NAME == rn.bosch_name_resolver("D+D→p+T")

    def test_DDT_2(self):
        assert DDT_NAME == rn.bosch_name_resolver("D+D→T+p")

    def test_DDT_3(self):
        assert DDT_NAME == rn.bosch_name_resolver("²H+²H→³H+¹H")

    def test_DDT_4(self):
        assert DDT_NAME == rn.bosch_name_resolver("²H+²H→¹H+³H")

    def test_DDHe3_0(self):
        assert DDHE3_NAME == rn.bosch_name_resolver(DDHE3_NAME)

    def test_DDHe3_1(self):
        assert DDHE3_NAME == rn.bosch_name_resolver("D(d,n)3He")

    def test_DDHe3_2(self):
        assert DDHE3_NAME == rn.bosch_name_resolver("D+D→n+3He")

    def test_DDHe3_3(self):
        assert DDHE3_NAME == rn.bosch_name_resolver("²H+²H→n+3He")

    def test_DDHe3_4(self):
        assert DDHE3_NAME == rn.bosch_name_resolver("²H+²H→3He+n")

    def test_bad_name(self):
        self.assertRaises(ValueError, rn.bosch_name_resolver, "bad")


class TestProtonBoronNameResolver(unittest.TestCase):
    def test_pb_0(self):
        assert PB11_NAME == rn.proton_boron_name_resolver(PB11_NAME)

    def test_pb_1(self):
        assert PB11_NAME == rn.proton_boron_name_resolver("pB")

    def test_pb_2(self):
        assert PB11_NAME == rn.proton_boron_name_resolver("p+B")

    def test_pb_3(self):
        assert PB11_NAME == rn.proton_boron_name_resolver("p+¹¹B")


class TestProtonLithiumNameResolver(unittest.TestCase):
    def test_pli_0(self):
        assert PLI6_NAME == rn.proton_lithium_name_resolver(PLI6_NAME)

    def test_pli_1(self):
        assert PLI6_NAME == rn.proton_lithium_name_resolver("p+⁶Li")

    def test_pli_2(self):
        assert PLI6_NAME == rn.proton_lithium_name_resolver("p+⁶Li->⁴He+³He")


if __name__ == "__main__":
    unittest.main()
