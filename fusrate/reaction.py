import fusrate.reactionnames as rn
from fusrate.bosch import BoschCrossSection
from fusrate.bosch import BoschRateCoeff
from fusrate.endf import ENDFCrossSection
from fusrate.ion_data import ion_mass
from fusrate.ratecoefficient import RateCoefficientIntegratorMaxwellian
from fusrate.ratecoefficient import RateCoefficientInterpolator
from fusrate.constants import Distributions

INTERPOLATION = "interpolation"
ANALYTIC = "analytic"
INTEGRATION = "integration"


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
        self.name = rn.name_resolver(name)

        self.beam, self.target = rn.reactants(self.name)

        self.m_beam = ion_mass(self.beam)
        self.m_tar = ion_mass(self.target)

        self.bt_to_com = self.m_tar / (self.m_beam + self.m_tar)

    def canonical_name(self):
        return self.name

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
        name = self.rcore.canonical_name()
        self.name = name

        self.cross_analytic_call = self._no_cross_analytic

        self.ratecoeff = dict()
        # Maxwellian distribution always exists
        self.ratecoeff[Distributions.MAXW] = dict()

        # see if Bosch-Hale provides reactivity
        self.has_analytic_fit = name in BoschCrossSection.provides_reactions()
        if self.has_analytic_fit:
            self.bh_cross = BoschCrossSection(name)
            self.cross_analytic_call = self.bh_cross.cross_section
            b_rcf = BoschRateCoeff(name)
            self.ratecoeff[Distributions.MAXW][ANALYTIC] = b_rcf.ratecoeff

        self.cross_sec_interpolator = ENDFCrossSection(name).cross_section

        self._load_maxwellian_rate_coefficient_integrator()
        self._load_maxwellian_rate_coefficient_interpolator()

        self._rcmaxinterp = self.ratecoeff[Distributions.MAXW][
            INTERPOLATION
        ]

    def _load_maxwellian_rate_coefficient_integrator(self):
        self.ratecoeff[Distributions.MAXW][
            INTEGRATION
        ] = RateCoefficientIntegratorMaxwellian(
            self.rcore, self.cross_sec_interpolator
        ).ratecoeff

    def _load_maxwellian_rate_coefficient_interpolator(self):
        ratecoeff_interp_maxw = RateCoefficientInterpolator(
            self.name, Distributions.MAXW
        )
        self.ratecoeff[Distributions.MAXW][
            INTERPOLATION
        ] = ratecoeff_interp_maxw.rate_coefficient

    def __str__(self):
        return f"Reaction {self.canonical_name()}"

    def __repr__(self):
        return f"fusrate.reaction.Reaction({self.canonical_name()})"

    def canonical_name(self):
        return self.rcore.canonical_name()

    def reactants(self):
        return self.rcore.reactants()

    def reactant_masses(self):
        return self.rcore.reactant_masses()

    def beam_target_to_com_factor(self):
        return self.rcore.beam_target_to_com_factor()

    ### cross section functions

    def cross_section(self, e):
        r"""Look up the interpolated cross section from ENDF data

        Parameters
        ----------
        e : array_like,
          energies in keV

        Returns
        -------
        Cross sections in mb
        """
        return self.cross_sec_interpolator(e)

    def _no_cross_analytic(self, *T, **kwargs):
        r"""There is no implemented analytic formation to the reactivity.

        Please do not call the function reactivity_analytic, or it will throw
        an error.
        """
        raise NotImplementedError(
            f"There is no implemented analytic reactivity"
            f" function for {self.canonical_name()}."
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
    ):
        self._validate_ratecoeff_opts(distribution, scheme, derivatives)

        return self.ratecoeff[distribution][scheme](
            *args, derivatives=derivatives
        )


if __name__ == "__main__":
    import numpy as np

    r = Reaction("D+T")
    ts = np.array([10, 20, 30])

    s = r.rate_coefficient(
        ts,
        scheme="analytic",
        derivatives=True,
    )
    print(s)
    s = r.rate_coefficient(
        ts,
        scheme="interpolation",
        derivatives=True,
    )
    print(s)
