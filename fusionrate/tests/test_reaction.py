from fusionrate import Reaction

import unittest
import numpy as np
from fusionrate.tests.utility import has_nans, no_nans, has_zeros, has_negs
from fusionrate.tests.utility import all_finite, all_nonneg
from fusionrate.reactionnames import ALL_REACTIONS as all_reactions
from fusionrate.reactionnames import DT_NAME

import pytest

def has_shape(x, yshape):
    assert x.shape == yshape

# all the functions herein take values in keV
okay_values = [0, 1e-3, 1, 10, 1e3, 4e3]

peta_eV = [1e12]
googol_eV = [1e97]

standard_cases = (
        ([np.nan],    has_nans),
        ([np.inf],    has_nans),
        ([-np.inf],   has_nans),
        ([-1],        has_nans),
        ([1],         no_nans),
        (1,           lambda x: has_shape(x, (1,))),
        (1,           all_nonneg),
        (1,           all_finite),
        (okay_values, no_nans),
        (okay_values, all_finite),
        (np.array(okay_values).reshape((3,2)), lambda x: has_shape(x, (3,2))),
       )

cross_section_cases = (
        ([0],         has_zeros),
        ([1],         no_nans),
        (okay_values, all_nonneg),
       )


def generate_cross_section_test_cases(cases):
    """Iterate over found reactions and schemes to expand test cases

    Parameters
    ----------
    cases: iterable of 2-tuples

    Returns
    -------
    list of 4-tuples, each of which is
    (reaction name, cross section evaluation scheme,
    test value, test that should pass)
    """
    cross_sections_to_test = []
    for reaction in all_reactions:
        rx = Reaction(reaction)
        available_cross_sections = rx.available_cross_sections()
        for scheme in available_cross_sections:
            for var, func in cases:
                cross_sections_to_test.append((rx.name, scheme, var, func))
    return cross_sections_to_test

cases = standard_cases + cross_section_cases
cross_sections_to_test = generate_cross_section_test_cases(cases)

@pytest.mark.parametrize("rx_name, scheme, var, func", cross_sections_to_test)
def test_cross_section_function(rx_name, scheme, var, func):
    rx = Reaction(rx_name)
    result = rx.cross_section(var, scheme=scheme)
    func(result)

# derivatives of cross sections
cross_section_derivative_cases = (
        ([0, 1],      no_nans),
       )

cases = standard_cases + cross_section_derivative_cases
cross_sections_to_test = generate_cross_section_test_cases(cases)

@pytest.mark.parametrize("rx_name, scheme, var, func", cross_sections_to_test)
def test_cross_section_deriv(rx_name, scheme, var, func):
    rx = Reaction(rx_name)
    result = rx.cross_section(var, scheme=scheme, derivatives=True)
    func(result)

class TestReaction(unittest.TestCase):
    def setUp(self):
        self.entemps = np.array([3, 5, 10, 20], dtype=float)  # in keV
        self.twobythree = np.array([[3, 5], [10, 20], [40, 50]], dtype=float)  # in keV
        self.array_with_zero = np.array([0, 1], dtype=float)
        self.array_with_neg = np.array([-1, 1], dtype=float)
        self.array_with_neginf = np.array([-np.inf, 1])
        self.array_with_inf = np.array([1, np.inf])
        self.array_with_nan = np.array([np.nan, 1])
        self.array_manybad = np.array([-np.inf, np.nan, np.inf])
        self.array_allnan = np.array([np.nan, np.nan, np.nan])
        self.singlefloat = 2
        self.verysmall = 1e-3
        self.rx = Reaction(DT_NAME)

    def rc_analytic_func(self, t, **kwargs):
        return self.rx.rate_coefficient(t, scheme="analytic", **kwargs)

    # tests of the analytic rate coefficient
    def test_rc_analytic_func(self):
        result = self.rc_analytic_func(self.entemps)
        all_nonneg(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_twobythree(self):
        result = self.rc_analytic_func(self.twobythree)
        all_nonneg(result)
        self.assertEqual(result.shape, self.twobythree.shape)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_single(self):
        result = self.rc_analytic_func(self.singlefloat)
        self.assertEqual(len(result), 1)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_singlenan(self):
        result = self.rc_analytic_func(np.nan)
        self.assertEqual(len(result), 1)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_zero(self):
        result = self.rc_analytic_func(self.array_with_zero)
        has_zeros(result)
        all_nonneg(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_neg(self):
        result = self.rc_analytic_func(self.array_with_neg)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_neginf(self):
        result = self.rc_analytic_func(self.array_with_neginf)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_neginf(self):
        result = self.rc_analytic_func(self.array_with_neginf)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_inf(self):
        result = self.rc_analytic_func(self.array_with_inf)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_nan(self):
        result = self.rc_analytic_func(self.array_with_nan)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_manybad(self):
        result = self.rc_analytic_func(self.array_manybad)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_allnan(self):
        result = self.rc_analytic_func(self.array_allnan)
        has_nans(result)

    @pytest.mark.skip(reason="Not ready to test rate coefficients")
    def test_rc_analytic_func_verysmall(self):
        result = self.rc_analytic_func(self.verysmall)
        all_nonneg(result)

if __name__ == "__main__":
    pytest.main()
