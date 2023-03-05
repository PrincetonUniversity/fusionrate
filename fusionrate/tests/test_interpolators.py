from fusionrate.interpolators import RateCoefficientInterpolator

import unittest
import numpy as np
from fusionrate.tests.utility import has_nans, no_nans
from fusionrate.reactionnames import DT_NAME


class TestRateCoefficientInterpolator1D(unittest.TestCase):
    def setUp(self):
        self.temperatures = np.array([3, 5, 10, 20], dtype=float)  # in keV
        self.array_with_zero = np.array([0, 1], dtype=float)  # in keV
        self.array_with_neg = np.array([-1, 1], dtype=float)  # in keV
        self.array_with_neginf = np.array([-np.inf, 1])  # in keV
        self.array_with_inf = np.array([1, np.inf])  # in keV
        self.array_with_nan = np.array([np.nan, 1])  # in keV
        self.dt_max = RateCoefficientInterpolator("T(d,n)a", "Maxwellian")

    def test_ratecoeff(self):
        self.dt_max.rate_coefficient(self.temperatures)

    def test_derivative(self):
        self.dt_max.derivative(self.temperatures)

    def test_parameters(self):
        self.dt_max.parameters

    def test_ratecoeff_zero_no_nans(self):
        result = self.dt_max.rate_coefficient(self.array_with_zero)
        no_nans(result)

    def test_ratecoeff_neg_no_nans(self):
        result = self.dt_max.rate_coefficient(self.array_with_neg)
        no_nans(result)

    def test_ratecoeff_neginf_no_nans(self):
        result = self.dt_max.rate_coefficient(self.array_with_neginf)
        no_nans(result)

    def test_ratecoeff_inf_no_nans(self):
        result = self.dt_max.rate_coefficient(self.array_with_neginf)
        no_nans(result)

    def test_ratecoeff_nan_has_nans(self):
        result = self.dt_max.rate_coefficient(self.array_with_nan)
        has_nans(result)

    def test_derivative_zero_no_nans(self):
        result = self.dt_max.derivative(self.array_with_zero)
        no_nans(result)

    def test_derivative_neg_no_nans(self):
        result = self.dt_max.derivative(self.array_with_neg)
        no_nans(result)

    def test_derivative_neginf_no_nans(self):
        result = self.dt_max.derivative(self.array_with_neginf)
        no_nans(result)

    def test_derivative_zero_has_nans(self):
        result = self.dt_max.derivative(self.array_with_nan)
        has_nans(result)


if __name__ == "__main__":
    unittest.main()
