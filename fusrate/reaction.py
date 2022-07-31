import fusrate.reactionnames as rn
from fusrate.bosch import BoschCrossSection
from fusrate.bosch import BoschRateCoeff
from fusrate.endf import ENDFCrossSection
from fusrate.ion_data import ion_mass
from fusrate.integrators import RateCoefficientIntegratorBiMaxwellian
from fusrate.integrators import RateCoefficientIntegratorMaxwellian
from fusrate.interpolators import RateCoefficientInterpolator
from fusrate.constants import Distributions
from fusrate.load_data import ratecoeff_data_exists

import numpy as np

INTERPOLATION = "interpolation"
ANALYTIC = "analytic"
INTEGRATION = "integration"

ENDF = "ENDF"


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

    def reactants(self):
        return self.beam, self.target

    def reactant_masses(self):
        return self.m_beam, self.m_tar

    def beam_target_to_com_factor(self):
        return self.bt_to_com


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
        self._cross_section[ENDF] = ENDFCrossSection(self._name).cross_section
        self._cross_section[ANALYTIC] = self._no_cross_analytic

        self._ratecoeff = dict()

        # Maxwellian distribution always exists
        self._ratecoeff[Distributions.MAXW] = dict()

        # see if Bosch-Hale provides reactivity
        self.has_analytic_fit = self._name in BoschCrossSection.provides_reactions()
        if self.has_analytic_fit:
            self.bh_cross = BoschCrossSection(self._name)
            self._cross_section[ANALYTIC] = self.bh_cross.cross_section
            self.b_rcf = BoschRateCoeff(self._name)
            self._ratecoeff[Distributions.MAXW][
                ANALYTIC
            ] = self.b_rcf.rate_coefficient

        self._load_maxwellian_ratecoeff_integrator()
        self._load_maxwellian_ratecoeff_interpolator()

        self._load_bimaxwellian_ratecoeff()


    @property
    def name(self):
        return self._name

    # gross, not scalable, should be replaced
    def _load_bimaxwellian_ratecoeff(self):
        self._ratecoeff[Distributions.BIMAXW] = dict()
        self._load_bimaxwellian_ratecoeff_integrator()
        self._load_bimaxwellian_ratecoeff_interpolator()

    def _load_bimaxwellian_ratecoeff_integrator(self):
        self._ratecoeff[Distributions.BIMAXW][
            INTEGRATION
        ] = RateCoefficientIntegratorBiMaxwellian(
            self.rcore, self._cross_section[ENDF]
        ).ratecoeff

    def _load_bimaxwellian_ratecoeff_interpolator(self):
        if ratecoeff_data_exists(self._name, Distributions.BIMAXW):
            interp_bimaxw = RateCoefficientInterpolator(
                self._name, Distributions.BIMAXW
            )
            self._ratecoeff[Distributions.BIMAXW][
                INTERPOLATION
            ] = interp_bimaxw.rate_coefficient

    def _load_maxwellian_ratecoeff_integrator(self):
        self._ratecoeff[Distributions.MAXW][
            INTEGRATION
        ] = RateCoefficientIntegratorMaxwellian(
            self.rcore, self._cross_section[ENDF]
        ).ratecoeff

    def _load_maxwellian_ratecoeff_interpolator(self):
        """Adds an interpolator"""
        # Need to add logic for what to do if data does not exist
        if ratecoeff_data_exists(self._name, Distributions.MAXW):
            interp_maxw = RateCoefficientInterpolator(
                self._name, Distributions.MAXW
            )
            self._ratecoeff[Distributions.MAXW][
                INTERPOLATION
            ] = interp_maxw.rate_coefficient

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
        return f"fusrate.reaction.Reaction({self._name})"

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
        return self._cross_section[scheme](e, derivatives)

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

        return self._ratecoeff[distribution][scheme](
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    import numpy as np

    r = Reaction("D+T")
    ts = np.array([10, 20, 30])

    cs = r.cross_section(ts, scheme="ENDF", derivatives=True)
    # cs = r.cross_section(ts, scheme="analytic", derivatives=True)

    s = r.rate_coefficient(
        ts,
        scheme="interpolation",
    )
    # print(s)
    # s = r.rate_coefficient(
    #     ts,
    #     scheme="interpolation",
    #     derivatives=True,
    # )
    # print(s)
    print(r)
