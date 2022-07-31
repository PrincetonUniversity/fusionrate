from fusrate.interpolators import RateCoefficientInterpolator

import unittest
import numpy as np
from fusrate.reactionnames import DT_NAME


def no_nans(val):
    assert not np.any(np.isnan(val))


class TestRateCoefficientInterpolator1D(unittest.TestCase):

    def setUp(self):
        self.temperatures = np.array([3, 5, 10, 20])  # in keV
        self.array_with_zero = np.array([0, 5, 10, 20])  # in keV
        self.array_with_neg = np.array([-1, 5, 10, 20])  # in keV
        self.dt_max = RateCoefficientInterpolator("T(d,n)a", "Maxwellian")

    def test_ratecoeff(self):
        self.dt_max.rate_coefficient(self.temperatures)

    def test_derivative(self):
        self.dt_max.derivative(self.temperatures)

    def test_parameter_limits(self):
        self.dt_max.parameter_limits()

    def test_ratecoeff_zero(self):
        result = self.dt_max.rate_coefficient(self.array_with_zero)
        no_nans(result)

    def test_ratecoeff_neg(self):
        result = self.dt_max.rate_coefficient(self.array_with_neg)
        no_nans(result)

    def test_derivative_zero(self):
        result = self.dt_max.derivative(self.array_with_zero)
        no_nans(result)

    def test_derivative_neg(self):
        result = self.dt_max.derivative(self.array_with_neg)
        no_nans(result)


if __name__ == "__main__":
    unittest.main()
