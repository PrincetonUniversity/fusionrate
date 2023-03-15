import numpy as np

import fusionrate.reactionnames as rn
from fusionrate.bosch import BoschCrossSection
from fusionrate.bosch import BoschRateCoeff
from fusionrate.endf import ENDFCrossSection
from fusionrate.ion_data import ion_mass
from fusionrate.integrators import rate_coefficient_integrator_factory
from fusionrate.interpolators import RateCoefficientInterpolator
from fusionrate.constants import Distributions
from fusionrate.load_data import ratecoeff_data_exists

import functools

INTERPOLATION = "interpolation"
ANALYTIC = "analytic"
INTEGRATION = "integration"

ENDF = "ENDF"

RAW = "raw_"
FUNC = "function"
DERIV = "derivative"
PARAMS = "parameters"
OBJ = "object"


class ReactionCore:
    r"""Basic, cross-section-independent reaction data

    Beam and Target refer to the formally specified reactants in the canonical
    name, Target(Beam, product1)product2.
    This is important because ENDF cross sections are given in a frame where
    the target is in the lab frame, and we want to convert to the
    center-of-mass frame.

    I'm not sure if product1 and product2 have specific names.
    """

    def __init__(self, name):
        self._name = rn.name_resolver(name)

        self.beam, self.target = rn.reactants(self._name)

        self.m_beam = ion_mass(self.beam)
        self.m_tar = ion_mass(self.target)

        self.bt_to_com = self.m_tar / (self.m_beam + self.m_tar)

    @property
    def canonical_name(self):
        return self._name

    @property
    def reactants(self):
        return self.beam, self.target

    @property
    def reactant_masses(self):
        return self.m_beam, self.m_tar

    @property
    def beam_target_to_com_factor(self):
        return self.bt_to_com

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplementedError
        return self._name == other._name

    def __hash__(self):
        return hash((self._name,))

    def __str__(self):
        return f"{self.__class__.__name__} {self._name}"

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._name})"


def _cross_section_node(obj):
    d = {
        OBJ: obj,
        PARAMS: obj.parameters[0],
        FUNC: obj.cross_section,
        DERIV: obj.derivative,
    }
    return d


def _ratecoeff_node(obj):
    d = {
        OBJ: obj,
        PARAMS: obj.parameters,
        FUNC: obj.rate_coefficient,
        DERIV: obj.derivative,
    }
    return d


def _wrap_for_zero_when_out_of_bounds(func, bounds):
    low, high = bounds

    @functools.wraps(func)
    def wrapper(x, **kwargs):
        result = np.zeros(x.shape)
        acceptable = (x > low) & (x < high)
        result[acceptable] = func(x[acceptable], **kwargs)
        return result

    return wrapper


def _wrap_for_zero_below_lower_bound(func, bounds):
    low, high = bounds

    @functools.wraps(func)
    def wrapper(x, **kwargs):
        result = np.zeros(x.shape)
        acceptable = x > low
        result[acceptable] = func(x[acceptable], **kwargs)
        return result

    return wrapper


def _wrap_to_move_values_in_bounds(func, bounds):
    low, high = bounds

    @functools.wraps(func)
    def wrapper(x, **kwargs):
        corrected_values = np.clip(x, low, high)
        return func(corrected_values, **kwargs)

    return wrapper


BEHAVIORS = {
    "zeros": _wrap_for_zero_when_out_of_bounds,
    "zero_below": _wrap_for_zero_below_lower_bound,
    "const": _wrap_to_move_values_in_bounds,
    "raw": lambda x, y: x,
}


def _normalize_dimensions_and_copy(x):
    return np.atleast_1d(x).astype(float)


def _normalize_energy(e):
    r"""Set negative, infinite, and nan values to nan

    Parameters
    ----------
    e: float or array_like

    Returns
    -------
    A new array with the same nonnegative, finite elements as e,
    but np.nan everywhere else.

    Notes
    -----
    This returns a new copy of the array.

    Developer notes
    ---------------
    This function has not been benchmarked and may be improved, perhaps by
    altering the same array rather than returning a new one (only using
    .astype(float) if the type is not float), or by avoiding the
    logical_or and instead doing two set (=) operations.
    """
    e = _normalize_dimensions_and_copy(e)
    mask = np.logical_or(e < 0.0, ~np.isfinite(e))
    np.putmask(e, mask, np.nan)
    return e


def _operate_on_valid(func, e):
    r"""Call func(e) only for non-negative numbers

    Parameters
    ----------
    func: callable
        A function which takes one argument
    e: np.ndarray
        A numpy array representing energies or temperatures.

    """
    result = np.full(e.shape, np.nan)
    ix = e >= 0
    result[ix] = func(e[ix])
    return result


def _insert_nans(x, indices, axis=0):
    for i in indices:
        x = np.insert(x, i, np.nan, axis=axis)
    return x


def _operate_on_valid_entries_grid_mode(func, *args, **kwargs):
    """Apply a function to combinations of entries without NaN

    Parameters
    ----------
    func: callable
        A function which takes one or more arguments and returns a numpy array
        with shape equal to those of the list of its arguments in reversed
        order
    *args: one or more np.ndarrays
        These do not need to be the same length, but they all must be 1D

    Returns
    -------
    A numpy array with shape equal to the lengths of each of the input arrays,
    but in reversed order.

    """
    bad_indices = [np.where(~np.isfinite(arg))[0] for arg in args]

    some_bad_entries = any(np.array([len(a) for a in bad_indices]) > 0)

    if some_bad_entries:
        args = [
            np.delete(arg, bads, axis=0) for arg, bads in zip(args, bad_indices)
        ]

    result = func(*args, **kwargs)

    # Replace with hyperplanes of nans
    # This might end up being an n^3 process for large arrays.
    # May want to seek a better algorithm (or data structure) for masking / ignoring 'bads'.
    if some_bad_entries:
        for axis, bads in enumerate(bad_indices):
            if len(bads) > 0:
                result = _insert_nans(result, bads, axis=(len(args)-axis-1))

    return result


def _operate_on_valid_entries_of_arrays(func, *args, **kwargs):
    """Apply a function to non-NaN array entries

    Parameters
    ----------
    func: callable
        A function which takes one or more arguments
    *args: one or more np.ndarrays
        These do not need to be the same shape, but they must be broadcastable.

    Returns
    -------
    A np.ndarray with the broadcasted shape
    """
    results = np.full(np.broadcast(*args).shape, np.nan)

    status = [np.isfinite(arg) for arg in args]
    broadcasted_arrays = np.broadcast_arrays(*args)
    all_ok = np.all(np.broadcast_arrays(*status), axis=0)
    valid_inputs = [bca[all_ok] for bca in broadcasted_arrays]
    results[all_ok] = func(*valid_inputs, **kwargs)
    return results


class Reaction:
    r"""Main reaction class"""

    def __init__(self, name):
        self.rcore = ReactionCore(name)
        self._name = self.rcore.canonical_name

        # re-use functions from rcore
        self.reactants = self.rcore.reactants
        self.reactant_masses = self.rcore.reactant_masses
        self.beam_target_to_com_factor = self.rcore.beam_target_to_com_factor

        self._cross_section = dict()

        self._load_cross_section_ENDF()

        self._ratecoeff = dict()

        # see if Bosch-Hale provides reactivity
        self.has_analytic_fit = (
            self._name in BoschCrossSection.provides_reactions()
        )
        self._load_cross_section_analytic()

        initial_distributions = [Distributions.MAXW, Distributions.BIMAXW]

        for dist in initial_distributions:
            self._load_ratecoeff_analytic(dist)
            self._load_integrator(dist)
            self._load_ratecoeff_interpolator(dist)

    @property
    def name(self):
        return self._name

    def _get_function_node(self, which: str, scheme: str, distribution: str):
        if which == "cross section":
            tree = self._cross_section
        elif which == "rate coefficient":
            tree = self._ratecoeff[distribution]
        else:
            raise ValueError(
                f"""Argument "which", currently "{which}", must be either either
            "cross section" or "rate coefficient"."""
            )
        return tree[scheme]

    def _set_bounds_behavior(self, node, subfunction, bounds, behavior):
        """
        Parameters
        ----------
        bounds: list [low, high]
        behavior: string

        """
        raw_subfunction = RAW + subfunction
        # this should only ever happen once
        if not raw_subfunction in node:
            node[raw_subfunction] = node[subfunction]

        node[subfunction] = BEHAVIORS[behavior](node[raw_subfunction], bounds)

    def set_extrapolation_behavior(
        self,
        which="cross_section",
        scheme="analytic",
        distribution=Distributions.MAXW,
        derivatives=False,
        behavior="zeros",
        extrapolable=False,
    ):
        """Set the outside-bounds behavior of a function

        Parameters
        ----------
        which: str
            "cross section" or "rate coefficient"
        scheme: str
            Any of the loaded evaluation schemes
        distribution: str
            For rate coefficients, the distribution function
        derivatives: bool
            Whether to affect the main function or its derivative
        behavior: str
            * "zeros": return 0 outside the bounds
            * "zero_below": return 0 beneath the lower bound, extrapolate above
            * "const": return the boundary value
        extrapolable: bool
            If the function specifies reasonably extrapolable bounds, use those
            instead of the "safe" bounds.
        """
        node = self._get_function_node(which, scheme, distribution)
        params = node[PARAMS]
        bounds = params.extrapolable_bounds if extrapolable else params.bounds
        subfunction = DERIV if derivatives else FUNC
        self._set_bounds_behavior(node, subfunction, bounds, behavior)

    def _ensure_distribution(self, dist):
        if not self._ratecoeff.get(dist):
            self._ratecoeff[dist] = dict()

    def _load_cross_section_analytic(self):
        if self.has_analytic_fit:
            obj = BoschCrossSection(self._name)
            node = _cross_section_node(obj)
            self._cross_section[ANALYTIC] = node

            bounds = node[PARAMS].bounds

            self._set_bounds_behavior(node, FUNC, bounds, behavior="zeros")
            self._set_bounds_behavior(node, DERIV, bounds, behavior="const")

    def _load_cross_section_ENDF(self):
        obj = ENDFCrossSection(self._name)
        node = _cross_section_node(obj)
        self._cross_section[ENDF] = node

        bounds = node[PARAMS].extrapolable_bounds

        self._set_bounds_behavior(node, FUNC, bounds, behavior="zero_below")
        self._set_bounds_behavior(node, DERIV, bounds, behavior="const")

    def _load_ratecoeff_analytic(self, dist, **kwargs):
        if self.has_analytic_fit and dist == Distributions.MAXW:
            self._ensure_distribution(dist)
            obj = BoschRateCoeff(self._name, **kwargs)
            node = _ratecoeff_node(obj)
            self._ratecoeff[dist][ANALYTIC] = node

            bounds = node[PARAMS][0].bounds

            self._set_bounds_behavior(node, FUNC, bounds, behavior="zeros")
            self._set_bounds_behavior(node, DERIV, bounds, behavior="const")

    def _load_integrator(self, dist, **kwargs):
        integrator = rate_coefficient_integrator_factory.create(
            self.rcore, self._cross_section[ENDF][FUNC], dist, **kwargs
        )

        node = {OBJ: integrator, FUNC: integrator.ratecoeff}

        self._ensure_distribution(dist)

        self._ratecoeff[dist][INTEGRATION] = node

    def _load_ratecoeff_interpolator(self, dist, **kwargs):
        if ratecoeff_data_exists(self._name, dist):
            interp = RateCoefficientInterpolator(self._name, dist, **kwargs)
            node = _ratecoeff_node(interp)
            self._ensure_distribution(dist)

            self._ratecoeff[dist][INTERPOLATION] = node

    def print_available_functions(self):
        self.print_available_cross_sections()
        self.print_available_rate_coefficients()

    def available_cross_sections(self):
        return list(self._cross_section.keys())

    def print_available_cross_sections(self):
        print(f"Available cross sections for {self._name}")
        for source, method in self._cross_section.items():
            params = method[PARAMS]
            bounds = params.bounds
            unit = params.unit
            bounds_message = f"[{bounds[0]:.3f} to {bounds[1]:.3g} {unit}]"

            eb = params.extrapolable_bounds
            extra_bounds_message = ""
            if eb[0] < bounds[0] or eb[1] > bounds[1]:
                extra_bounds_message = (
                    f", extrapolable to [{eb[0]:.3f} to {eb[1]:.3g} {unit}]"
                )

            print(f"    {source}, {bounds_message}{extra_bounds_message}")

    def loaded_distributions(self):
        return list(self._ratecoeff.keys())

    def available_distributions(self):
        """List of distributions for which there are rate coefficients

        Returns
        -------
        list

        Examples
        --------
        >>> from fusionrate import Reaction
        >>> r = Reaction("D+T")
        >>> r.available_distributions()
        ["Maxwellian"]
        """
        return self.loaded_distributions()

    def available_rate_coefficient_schemes(self, distribution):
        return list(self._ratecoeff[distribution].keys())

    def print_available_rate_coefficients(self):
        print(f"Available rate coefficient methods for {self._name}")
        for distribution, schemes in self._ratecoeff.items():
            print(f"{distribution} distribution:")
            for s, method in schemes.items():
                print(f"    {s}")

    def __str__(self):
        return f"{self.__class__.__name__} {self._name}"

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._name})"

    def cross_section(self, e, scheme="ENDF", derivatives=False):
        r"""Get interpolated (or analytic) cross sections

        Parameters
        ----------
        e : array_like,
          energies in keV

        scheme : ['ENDF', 'analytic']

        derivatives : bool
            Whether or not to return derivatives

        Returns
        -------
        Cross sections in millibarns, i.e. 10^(-27) cm²

        or

        The derivative w.r.t. energy, in mb/keV
        """
        node = self._cross_section[scheme]
        key = DERIV if derivatives else FUNC
        func = node[key]

        e = _normalize_energy(e)
        return _operate_on_valid(func, e)

    def _no_cross_analytic(self, *T, **kwargs):
        r"""There is no implemented analytic formation for the cross section.

        Please do not call cross_section(e, scheme="analytic"), or it will throw
        an error.
        """
        raise NotImplementedError(
            f"There is no implemented analytic cross section"
            f" function for {self._name}."
        )

    def get_rate_coefficient_object(self, distribution: str, scheme: str):
        return self._ratecoeff[distribution][scheme][OBJ]

    def _validate_ratecoeff_opts(
        self, distribution: str, scheme: str, derivatives: bool
    ):
        issues = []
        if scheme == ANALYTIC and distribution != Distributions.MAXW:
            err = f"⋅ Analytic formulas are only for the {Distributions.MAXW} distribution."
            issues.append(err)
        if derivatives and scheme == INTEGRATION:
            err = f"""⋅ Derivatives are only available via interpolation or from
an analytic formula (for {Distributions.MAXW} distributions for the four D reactions in
the Bosch-Hale paper). Direct computation is not yet supported."""
            issues.append(err)
        if issues:
            raise ValueError(
                f"{len(issues)} issue(s) detected in call to \
rate_coefficient_x:\n"
                + "\n".join(issues)
            )

    def _validate_ratecoeff_args(self, *normalized_args, **kwargs):
        if "grid" in kwargs and kwargs["grid"]:
            dimensions = np.array([arg.ndim for arg in normalized_args])
            if not all(dimensions == 1):
                raise ValueError(
                    f"""
                    grid=True can be used only for 0D or 1D arguments.
                    Arguments' dimensions were {dimensions}.
                    """
                )

    def rate_coefficient(
        self,
        *args,
        distribution=Distributions.MAXW,
        scheme=INTERPOLATION,
        derivatives=False,
        **kwargs,
    ):
        """Interpolated, analytic, or live-integrated rate coefficients

        Parameters
        ----------
        *args : one or more array_like
            Parameters, often temperatures in keV, describing the distribution

        distribution : str
            One of the registered distribution function names.

        scheme : ['integration', 'interpolation', 'analytic']
            'analytic' is available only for certain reactions
            but can be even faster than interpolation

            'integration' performs an integration using the distribution's
            integrator function. This is the slowest method.

            'interpolation' uses pre-computed tables to look up data

        derivatives : bool
            Return derivatives of the value w.r.t. each of the parameters

        Returns
        -------
        σv in cm³/s or its derivatives w.r.t. the parameters, i.e. in cm³/s/keV
        """
        self._validate_ratecoeff_opts(distribution, scheme, derivatives)

        node = self._ratecoeff[distribution][scheme]
        key = DERIV if derivatives else FUNC
        func = node[key]

        normalized_args = [_normalize_energy(arg) for arg in args]
        self._validate_ratecoeff_args(*normalized_args, **kwargs)

        safe_operator = _operate_on_valid_entries_of_arrays
        if "grid" in kwargs and kwargs["grid"]:
            safe_operator = _operate_on_valid_entries_grid_mode

        return safe_operator(
            func, *normalized_args, **kwargs
        )


if __name__ == "__main__":
    r = Reaction("D+T")
    ts = np.array([10, 20, 30])

    cs = r.cross_section(ts, scheme="ENDF", derivatives=True)
    cs = r.cross_section(ts, scheme="analytic", derivatives=True)

    s = r.rate_coefficient(
        ts,
        scheme="interpolation",
    )
    s = r.rate_coefficient(
        ts,
        scheme="analytic",
        derivatives=False,
    )
    s = r.rate_coefficient(
        ts,
        scheme="integration",
        derivatives=False,
    )
    r.print_available_functions()
    r.set_extrapolation_behavior("cross section")
