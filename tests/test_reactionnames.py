import fusionrate.reactionnames as rn
from fusionrate.reactionnames import name_resolver

from fusionrate.reactionnames import DT_NAME
from fusionrate.reactionnames import DHE3_NAME
from fusionrate.reactionnames import DDT_NAME
from fusionrate.reactionnames import DDHE3_NAME

from fusionrate.reactionnames import TT_NAME
from fusionrate.reactionnames import HH_NAME
from fusionrate.reactionnames import HT_NAME
from fusionrate.reactionnames import HTD_NAME

from fusionrate.reactionnames import PLI6_NAME
from fusionrate.reactionnames import PB11_NAME

from fusionrate.reactionnames import DLI6A_NAME
from fusionrate.reactionnames import DLI6N_NAME
from fusionrate.reactionnames import DLI6P_NAME

import pytest

import unittest

ACCEPTABLE_REACTION_NAMES = {
    DT_NAME: [
        DT_NAME,
        "DT",
        "D+T",
        "D+T→n+α",
        "D+T→α+n",
        "t(d,n)a",
    ],
    DHE3_NAME: [
        DHE3_NAME,
        "DHe3",
        "DHe",
        "D3He",
        "D+3He",
        "D+³He",
        "D+³He→p+⁴He",
        "D+³He→p+α",
        "D+³He→α+p",
        "²H +³He----->a+p",
        "h(d,p)a",
    ],
    DDT_NAME: [
        DDT_NAME,
        "D+D→p+T",
        "D+D→T+p",
        "²H+²H→³H+¹H",
        "²H+²H→¹H+³H",
        "d(d,p)t",
    ],
    DDHE3_NAME: [
        DDHE3_NAME,
        "D+D→n+3He",
        "D+D→3He+n",
        "²H+²H→n+3He",
        "²H+²H→3He+n",
        "d(d,n)h",
    ],
    TT_NAME: [
        TT_NAME,
        "2T",
        "T+T",
        "T + T -> a + 2n",
        "t(t,2n)a",
    ],
    HH_NAME: [
        HH_NAME,
        "h + h -> 2 p + a",
        "h(h,2p)a",
    ],
    HT_NAME: [
        HT_NAME,
        "h + t -> p + n + a",
        "h(t,pn)a",
        "h(t,np)a",
    ],
    HTD_NAME: [
        HTD_NAME,
        "h + t -> d + a",
        "h(t,d)a",
    ],
    PB11_NAME: [
        PB11_NAME,
        "pB",
        "pB11",
        "p+B",
        "p+11B",
        "p+11B→3α",
        "p+11B→3 ⁴He",
    ],
    PLI6_NAME: [
        PLI6_NAME,
        "pLi6",
        "p + ⁶Li",
        "p + ⁶Li --> h + α",
        "6Li(p,h)a",
        "p+⁶Li->⁴He+³He"
    ],
    DLI6A_NAME: [
        DLI6A_NAME,
        "6Li(d,a)a",
    ],
    DLI6A_NAME: [
        DLI6A_NAME,
        "6Li(d,a)a",
    ],
    DLI6N_NAME: [
        DLI6N_NAME,
        "6Li(d,n)7Be",
        "6Li+d-->n+⁷Be",
        "6Li+d-->n+Be",
    ],
    DLI6P_NAME: [
        DLI6P_NAME,
        "6Li(d,p)7Li",
        "6Li+d-->p+⁷Li",
    ],
}


def generate_name_resolution_test_cases(cases):
    """Iterate over found names to expand test cases

    Parameters
    ----------
    cases: dict where each values is a list

    Returns
    -------
    list of 2-tuples, each of which is
    (canonical reaction name, raw_name)
    """
    names_to_test = []
    for canonical, raw_list in cases.items():
        for element in raw_list:
            names_to_test.append((canonical, element))
    return names_to_test


names_to_test = generate_name_resolution_test_cases(ACCEPTABLE_REACTION_NAMES)


@pytest.mark.parametrize("canonical_name, raw_name", names_to_test)
def test_name_resolver(canonical_name, raw_name):
    resolved = name_resolver(raw_name)
    assert resolved == canonical_name


class TestReactionNameParsing(unittest.TestCase):
    PROT = "H"
    DEU = "D"
    TRIT = "T"
    HE3 = "³He"
    LI6 = "⁶Li"
    B11 = "¹¹B"

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


class TestNameResolver(unittest.TestCase):
    def test_bad_name(self):
        self.assertRaises(ValueError, name_resolver, "bad")

if __name__ == "__main__":
    pytest.main()
