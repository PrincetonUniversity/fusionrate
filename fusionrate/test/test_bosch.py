import fusionrate.bosch as bosch
from fusionrate.test.utility import has_nans, no_nans

import unittest
import numpy as np


class TestBoschCrossSection(unittest.TestCase):
    r"""
    Results are from Table V of

    Bosch, H.-S.; Hale, G. M.
    Improved Formulas for Fusion Cross-Sections and
    Thermal Reactivities. Nuclear Fusion 1992, 32 (4).
    """

    def setUp(self):
        # in keV
        self.table_energies = np.array([3, 5, 10, 20, 50, 100, 200, 400])

    def compare_results(self, cs, actual):
        code_results = cs.cross_section(self.table_energies)
        compare = np.allclose(code_results, actual, rtol=2e-4, atol=1e-10)
        assert compare

    def test_cross_section_dt(self):
        cs = bosch.BoschCrossSection("DT")
        # in millibarns
        table_results = np.array(
            [
                9.808e-3,
                5.383e-1,
                2.702e1,
                4.077e2,
                4.219e3,
                3.427e3,
                1.138e3,
                4.126e2,
            ]
        )
        self.compare_results(cs, table_results)

    def test_cross_section_dhe3(self):
        cs = bosch.BoschCrossSection("D3He")
        # in millibarns
        table_results = np.array(
            [
                1.119e-11,
                5.199e-8,
                2.160e-4,
                6.568e-2,
                8.688e0,
                1.021e2,
                6.378e2,
                5.304e2,
            ]
        )
        self.compare_results(cs, table_results)

    def test_cross_section_ddpt(self):
        cs = bosch.BoschCrossSection("D(d,p)T")
        # in millibarns
        table_results = np.array(
            [
                2.513e-4,
                9.038e-3,
                2.812e-1,
                2.670e0,
                1.557e1,
                3.304e1,
                5.234e1,
                7.005e1,
            ]
        )
        self.compare_results(cs, table_results)

    def test_cross_section_ddn3he(self):
        cs = bosch.BoschCrossSection("D(d,n)³He")
        # in millibarns
        table_results = np.array(
            [
                2.445e-4,
                8.834e-3,
                2.779e-1,
                2.691e0,
                1.649e1,
                3.701e1,
                6.239e1,
                8.702e1,
            ]
        )
        self.compare_results(cs, table_results)

    def test_cross_section_nans(self):
        cs = bosch.BoschCrossSection("DT")
        val = cs.cross_section(np.array([np.nan]))
        has_nans(val)


class TestBoschRateCoeff(unittest.TestCase):
    r"""
    Results are from Table VIII of

    Bosch, H.-S.; Hale, G. M.
    Improved Formulas for Fusion Cross-Sections and
    Thermal Reactivities. Nuclear Fusion 1992, 32 (4).

    """

    def setUp(self):
        # temperatures in keV
        self.table_temperatures = np.array(
            [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        )

    def compare_results(self, r, actual):
        code_results = r.rate_coefficient(self.table_temperatures)
        compare = np.allclose(code_results, actual, rtol=5e-4, atol=1e-40)
        assert compare

    def test_rate_coefficient_dt(self):
        r = bosch.BoschRateCoeff("DT")
        # results in cm³/s
        table_results = np.array(
            [
                1.254e-26,
                5.697e-23,
                6.857e-21,
                2.977e-19,
                1.366e-17,
                1.136e-16,
                4.330e-16,
                8.649e-16,
            ]
        )
        self.compare_results(r, table_results)

    def test_rate_coefficient_dhe3(self):
        r = bosch.BoschRateCoeff("D3He")
        # results in cm³/s
        table_results = np.array(
            [
                1.414e-35,
                1.241e-29,
                3.057e-26,
                1.399e-23,
                6.377e-21,
                2.126e-19,
                3.482e-18,
                5.554e-17,
            ]
        )
        self.compare_results(r, table_results)

    def test_rate_coefficient_ddpt(self):
        r = bosch.BoschRateCoeff("D(d,p)T")
        # results in cm³/s
        table_results = np.array(
            [
                4.640e-28,
                1.204e-24,
                1.017e-22,
                3.150e-21,
                9.024e-20,
                5.781e-19,
                2.399e-18,
                9.838e-18,
            ]
        )
        self.compare_results(r, table_results)

    def test_rate_coefficient_ddn3he(self):
        r = bosch.BoschRateCoeff("D(d,n)³He")
        # results in cm³/s
        table_results = np.array(
            [
                4.482e-28,
                1.169e-24,
                9.933e-23,
                3.110e-21,
                9.128e-20,
                6.023e-19,
                2.603e-18,
                1.133e-17,
            ]
        )
        self.compare_results(r, table_results)


if __name__ == "__main__":
    unittest.main()
