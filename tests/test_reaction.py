from fusionrate import Reaction

import unittest
import numpy as np
from .utility import *
from fusionrate.reactionnames import ALL_REACTIONS as all_reactions
from fusionrate.reactionnames import DT_NAME

import pytest


# all the functions herein take values in keV
okay_values = [0, 1e-3, 1, 10, 1e3, 4e3]

peta_eV = [1e12]
googol_eV = [1e97]


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


def generate_maxw_rate_coefficient_test_cases(cases):
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
    rate_coeffs_to_test = []
    for reaction in all_reactions:
        rx = Reaction(reaction)
        dist = "Maxwellian"
        for scheme in rx.available_rate_coefficient_schemes(dist):
            for var, func in cases:
                rate_coeffs_to_test.append((rx.name, dist, scheme, var, func))
    return rate_coeffs_to_test


standard_cases = (
    ([], is_empty),
    ([np.nan], has_nans),
    ([np.inf], has_nans),
    ([-np.inf], has_nans),
    ([-1], has_nans),
    ([1], no_nans),
    (1, lambda x: has_shape(x, (1,))),
    (1, all_nonneg),
    (1, all_finite),
    (okay_values, no_nans),
    (okay_values, all_finite),
    (np.array(okay_values).reshape((3, 2)), lambda x: has_shape(x, (3, 2))),
)

cross_section_cases = (
    ([0], has_zeros),
    ([1], no_nans),
    (okay_values, all_nonneg),
)


cases = standard_cases + cross_section_cases
cross_sections_to_test = generate_cross_section_test_cases(cases)


@pytest.mark.parametrize("rx_name, scheme, var, func", cross_sections_to_test)
def test_cross_section_function(rx_name, scheme, var, func):
    rx = Reaction(rx_name)
    result = rx.cross_section(var, scheme=scheme)
    func(result)


# derivatives of cross sections
cross_section_derivative_cases = (
    ([0, 1], no_nans),
)

cases = standard_cases + cross_section_derivative_cases
cross_sections_to_test = generate_cross_section_test_cases(cases)


@pytest.mark.parametrize("rx_name, scheme, var, func", cross_sections_to_test)
def test_cross_section_deriv(rx_name, scheme, var, func):
    rx = Reaction(rx_name)
    result = rx.cross_section(var, scheme=scheme, derivatives=True)
    func(result)


cases = standard_cases + cross_section_cases
rate_coefficients_to_test = generate_maxw_rate_coefficient_test_cases(cases)


@pytest.mark.parametrize("rx_name, dist, scheme, var, func", rate_coefficients_to_test)
def test_rc_max(rx_name, dist, scheme, var, func):
    rx = Reaction(rx_name)
    result = rx.rate_coefficient(var, distribution=dist, scheme=scheme)
    func(result)


if __name__ == "__main__":
    pytest.main()
