import fusrate.reactionnames as rn
from fusrate.bosch import BoschCrossSection
from fusrate.bosch import BoschRateCoeff
from fusrate.endf import ENDFCrossSection
from fusrate.ion_data import ion_mass
from fusrate.integrators import rate_coefficient_integrator_factory
from fusrate.interpolators import RateCoefficientInterpolator
from fusrate.constants import Distributions
from fusrate.load_data import ratecoeff_data_exists

import numpy as np

INTERPOLATION = "interpolation"
ANALYTIC = "analytic"
INTEGRATION = "integration"

ENDF = "ENDF"

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
        if self.__class__ == other.__class__:
            return self._name == other._name
        return False

    def __hash__(self):
        return hash((self._name, ))

    def __str__(self):
        return f"{self.__class__.__name__} {self._name}"

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._name})"

def _cross_section_node(obj):
    d = {
        OBJ: obj,
        PARAMS: obj.parameters,
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

        # everyone gets an ENDF cross section
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

    def _ensure_distribution(self, dist):
        if not self._ratecoeff.get(dist):
            self._ratecoeff[dist] = dict()

    def _load_cross_section_analytic(self):
        if self.has_analytic_fit:
            obj = BoschCrossSection(self._name)
            d = _cross_section_node(obj)
            self._cross_section[ANALYTIC] = d

    def _load_cross_section_ENDF(self):
        obj = ENDFCrossSection(self._name)
        d = _cross_section_node(obj)
        self._cross_section[ENDF] = d

    def _load_ratecoeff_analytic(self, dist, **kwargs):
        if self.has_analytic_fit and dist == Distributions.MAXW:
            self._ensure_distribution(dist)
            obj = BoschRateCoeff(self._name, **kwargs)
            d = _ratecoeff_node(obj)
            self._ratecoeff[dist][ANALYTIC] = d

    def _load_integrator(self, dist, **kwargs):

        integrator = rate_coefficient_integrator_factory.create(
            self.rcore, self._cross_section[ENDF][FUNC], dist, **kwargs
        )

        d = {OBJ: integrator, FUNC: integrator.ratecoeff}

        self._ensure_distribution(dist)

        self._ratecoeff[dist][INTEGRATION] = d

    def _load_ratecoeff_interpolator(self, dist, **kwargs):
        if ratecoeff_data_exists(self._name, dist):
            interp = RateCoefficientInterpolator(self._name, dist, **kwargs)
            d = _ratecoeff_node(interp)
            self._ensure_distribution(dist)

            self._ratecoeff[dist][INTERPOLATION] = d

    def print_available_functions(self):
        self.print_available_cross_sections()
        self.print_available_rate_coefficients()

    def print_available_cross_sections(self):
        print(f"Available cross sections for {self._name}")
        for source, method in self._cross_section.items():
            print(f"    {source}")

    def loaded_distributions(self):
        return list(self._ratecoeff.keys())

    def available_distributions(self):
        """List of distributions for which there are rate coefficients

        Returns
        -------
        list

        Examples
        --------
        >>> from fusrate import Reaction
        >>> r = Reaction("D+T")
        >>> r.available_distributions()
        ["Maxwellian"]
        """
        return self.loaded_distributions()

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
        e = np.asarray(e)
        node = self._cross_section[scheme]
        if not derivatives:
            func = node[FUNC]
        else:
            func = node[DERIV]

        return func(e)

    def _no_cross_analytic(self, *T, **kwargs):
        r"""There is no implemented analytic formation for the cross section.

        Please do not call cross_section(e, scheme="analytic"), or it will throw
        an error.
        """
        raise NotImplementedError(
            f"There is no implemented analytic cross section"
            f" function for {self._name}."
        )

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

        args = np.asarray(args)

        node = self._ratecoeff[distribution][scheme]

        if not derivatives:
            func = node[FUNC]
        else:
            func = node[DERIV]

        return func(*args, **kwargs)


if __name__ == "__main__":
    import numpy as np

    r = Reaction("D+D->³He + n")
    # ts = np.array([10, 20, 30])

    # cs = r.cross_section(ts, scheme="ENDF", derivatives=True)
    # cs = r.cross_section(ts, scheme="analytic", derivatives=True)

    # s = r.rate_coefficient(
    #     ts,
    #     scheme="interpolation",
    # )
    # print(s)
    # s = r.rate_coefficient(
    #     ts,
    #     scheme="analytic",
    #     derivatives=False,
    # )
    # print(s)
    # s = r.rate_coefficient(
    #     ts,
    #     scheme="integration",
    #     derivatives=False,
    # )
    # print(s)
    print(r.__repr__())
