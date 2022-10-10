import numpy as np
import scipy.interpolate

from fusionrate.constants import Distributions
from fusionrate.load_data import load_ratecoeff_hdf5

def _safe_log10(t):
    """Flushes zero or negative values to a small number"""
    very_small_exponent = -20
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.log10(t)
        res[np.isneginf(res)] = very_small_exponent
        res[np.isnan(res)] = very_small_exponent
        return res

def _ensure_lower_limit(t, lower_limit):
    return np.maximum(t, lower_limit)


class HdfRateCoefficientInterpolator:
    """Interpolate data based on an hdf5 dataset"""

    def __init__(self, dataset):
        r"""
        Parameters
        ----------
        dataset : h5py.Dataset
        """
        self._data_shape = dataset.shape
        self._num_parameters = len(self._data_shape)

        attrs = dataset.attrs
        self._canonical_reaction_name = attrs["Reaction"]
        self._data_units = attrs["Data units"]
        self._parameter_desc = attrs["Parameter descriptions"]
        self._parameter_limits = attrs["Parameter limits"]
        self._parameter_space_desc = attrs["Parameter space descriptions"]
        self._parameter_units = attrs["Parameter units"]
        self.type_of_data = attrs["Type of data"]
        self._distribution = attrs["distribution"]

        self._log_parameter_spines = [
            np.linspace(*s, self._data_shape[i])
            for i, s in enumerate(self._parameter_limits)
        ]
        self._parameter_spines = [
            10**lps for lps in self._log_parameter_spines
        ]

        self._raw_data = dataset[:]
        self._log_data = np.log10(self._raw_data)

    @property
    def parameter_limits(self):
        return 10 ** np.array(self._parameter_limits)

    @property
    def parameters(self):
        return list(
            zip(
                self._parameter_desc,
                self.parameter_limits,
                self._parameter_units,
            )
        )

    @property
    def distribution(self):
        return self._distribution

    @property
    def output_units(self):
        return self._data_units

    @property
    def canonical_reaction_name(self):
        return self._canonical_reaction_name


class OneDHdfRateCoefficientInterpolator(HdfRateCoefficientInterpolator):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._interp = scipy.interpolate.InterpolatedUnivariateSpline(
            *self._log_parameter_spines, self._log_data, ext=0
        )
        self._derivative_interp = None

    def rate_coefficient(self, temperatures):
        """Interpolate to find rate coefficients

        Parameters
        ----------
        temperatures : 1D array_like
            T in keV

        Returns
        -------
        Rate coefficients in cm³/s

        Examples
        --------
        >>> rate_coefficient([1,2,3], [2,3,4])
        [10, 20, 30]
        """
        log_temps = _safe_log10(temperatures)
        log_z = self._interp(log_temps)
        val = np.power(10, log_z)
        return val

    def _ensure_derivative(self):
        if not self._derivative_interp:
            self._derivative_interp = self._interp.derivative(n=1)

    def derivative(self, temperatures):
        self._ensure_derivative()

        # flush negatives or zeros to the lower limit
        lower_limit = self.parameter_limits[0][0]
        temperatures = _ensure_lower_limit(temperatures, lower_limit)

        log_temps = _safe_log10(temperatures)
        log_z = self._interp(log_temps)
        val = np.power(10, log_z)
        interp_prime = self._derivative_interp(log_temps)
        return val * interp_prime / temperatures


class TwoDHdfRateCoefficientInterpolator(HdfRateCoefficientInterpolator):
    def __init__(self, dataset):
        r"""
        Parameters
        ----------
        dataset : h5py.Dataset
        """
        super().__init__(dataset)
        self._interp = scipy.interpolate.RectBivariateSpline(
            *self._log_parameter_spines, self._log_data.T
        )

    def rate_coefficient(
        self,
        perp_temperatures,
        parallel_temperatures,
        grid=False,
    ):
        """Get a rate coefficient via interpolation

        Parameters
        ----------
        perp_temperatures : 1D array_like
            T_perp in keV
        parallel_temperatures : 1D array_like
            T_parallel in keV

        Returns
        -------
        Rate coefficients in cm³/s

        Examples
        --------
        >>> rate_coefficient([1,2,3], [2,3,4])
        [10, 20, 30]

        >>> rate_coefficient_grid([1,2,3], [2,3,4], grid=True)
        [[10, 20, 30],[11, 21, 31],[12, 22, 32]]
        """
        log_z = self._interp(
            np.log10(perp_temperatures),
            np.log10(parallel_temperatures),
            grid=grid,
        )
        return np.power(10, log_z.T)

    def derivative(self, perp_temperatures, parallel_temperatures, grid=False):
        pass


class RateCoeffInterpolatorFactory:
    def __init__(self):
        self._builders = {}

    def register_interpolator(self, distribution, builder):
        self._builders[distribution] = builder

    def create(self, canonical_name, distribution, **kwargs):
        builder = self._builders.get(distribution)
        if not builder:
            raise ValueError(distribution)
        dset = load_ratecoeff_hdf5(canonical_name, distribution)
        return builder(dset, **kwargs)


rate_coefficient_interpolator_factory = RateCoeffInterpolatorFactory()

rate_coefficient_interpolator_factory.register_interpolator(
    Distributions.MAXW, OneDHdfRateCoefficientInterpolator
)
rate_coefficient_interpolator_factory.register_interpolator(
    Distributions.BIMAXW, TwoDHdfRateCoefficientInterpolator
)


class RateCoefficientInterpolator:
    r"""Loads a rate coefficient interpolator"""

    def __init__(self, canonical_name, distribution, **kwargs):
        self.rci = rate_coefficient_interpolator_factory.create(
            canonical_name, distribution, **kwargs
        )
        self.rate_coefficient = self.rci.rate_coefficient
        self.derivative = self.rci.derivative
        self.parameters = self.rci.parameters
        self.output_units = self.rci.output_units
        self.canonical_reaction_name = self.rci.canonical_reaction_name
        self.distribution = self.rci.distribution


if __name__ == "__main__":
    from reactionnames import DT_NAME

    ex = RateCoefficientInterpolator("T(d,n)a", "Maxwellian")
    print(ex.parameters)
    print(ex.output_units)
    print(ex.distribution)
