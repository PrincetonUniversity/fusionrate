import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import root

import fusionrate.reactionnames as rn
from fusionrate.ion_data import ion_mass
from fusionrate.load_data import cross_section_data
from fusionrate.parameter import Parameter

import functools


class LogLogExtrapolation:
    r"""Interpolate and extrapolate in log-log space

    with a straight-line right end in log-log space
    """

    def __init__(self, x, y, linear_extension=True):
        r"""
        x: array_like
        y: array_like
        """
        self.x = x
        self.y = y
        self.max_x = max(x)
        self.min_x = min(x)
        logx = np.log(x)
        logy = np.log(y)
        data = np.array([logx, logy]).T

        # linear extension in loglog space
        if linear_extension:
            last_two = data[-2:]
            last = last_two[-1]
            Δ = last_two[-1] - last_two[-2]
            subsequent_points = last + np.outer(range(1, 4), Δ)
            self.data = np.append(data, subsequent_points, axis=0)
        else:
            self.data = data

        self.logx, self.logy = self.data.T

        self.interpolator = InterpolatedUnivariateSpline(
            self.logx, self.logy, k=2, ext=0
        )

        self._derivinterp = None

    def __call__(self, newx):
        r"""Generate new values

        Parameters
        ----------
        newx: array_like
            Energies in eV.
            Whether this is beam-target energy or c.o.m. energy
            is up to the data.
        """
        log_newx = np.log(newx)
        log_newy = self.interpolator(log_newx)
        return np.exp(log_newy)

    def query_in_loglog_space(self, log_newx):
        return self.interpolator(log_newx)

    def _ensure_derivatives(self):
        if not self._derivinterp:
            self._derivinterp = self.interpolator.derivative(n=1)

    def derivatives(self, newx):
        self._ensure_derivatives()
        log_newx = np.log(newx)
        log_newy = self.interpolator(log_newx)
        val = np.exp(log_newy)

        log_newy_prime = self._derivinterp(log_newx)

        return val * log_newy_prime / newx


class ENDFCrossSection:
    VERY_LOW_CROSS_SECTION = 1e-200
    UPPER_LIMIT_MULTIPLIER = 10

    def __init__(self, r):
        r"""
        s: reaction name string
        """
        if isinstance(r, str):
            name = rn.name_resolver(r)

            beam, target = rn.reactants(name)

            m_beam = ion_mass(beam)
            m_tar = ion_mass(target)

            self.bt_to_com = m_tar / (m_beam + m_tar)
        else:
            name = r.canonical_name
            self.bt_to_com = r.bt_to_com

        self.canonical_reaction_name = name
        x_raw, y_raw = cross_section_data(name)

        # Change from lab frame to COM frame
        # and from eV to keV (to match typical scales and Bosch-Hale)
        x = x_raw * self.bt_to_com / 1e3

        # Change from b to mb
        y = y_raw * 1e3

        self.x = x
        self.interp = LogLogExtrapolation(x, y, linear_extension=True)

    def __call__(self, e):
        return self.cross_section(e)

    def cross_section(self, e):
        r"""Look up the cross section from ENDF data
        Parameters
        ----------
        e : array_like,
          energies in keV

        Returns
        -------
        Cross sections in millibarns
        """
        return self.interp(e)

    def derivative(self, e):
        return self.interp.derivatives(e)

    @functools.cached_property
    def prescribed_domain(self):
        r"""A "safe" domain in energy space
        Returns
        -------
        [min, max] of COM energy domain in keV
        """
        return [min(self.x), max(self.x)]

    def _solve_for_energy_of_low_cross_section(self, cross_section):
        r"""Find energy at which cross section is some value

        Parameters
        ----------
        cross_section: float
            Value in mb. Must be lower than the lower bound cross section.

        Returns
        -------
        float, energy in keV
        """
        y_goal = np.log(cross_section)

        f = lambda energy: self.interp.query_in_loglog_space(energy) - y_goal
        guess = min(self.x) / 10
        log_of_energy = root(f, x0=guess).x[0]
        return np.exp(log_of_energy)

    @functools.cached_property
    def extrapolable_domain(self):
        r"""A wider, but hopefully reasonable, domain in energy space

        Returns
        -------
        [min, max] of COM energy domain in keV
        """
        low_cross_section = self.VERY_LOW_CROSS_SECTION
        very_low_energy = self._solve_for_energy_of_low_cross_section(
            low_cross_section
        )

        reasonable_upper_bound = max(self.x) * self.UPPER_LIMIT_MULTIPLIER
        return [very_low_energy, reasonable_upper_bound]

    @functools.cached_property
    def parameters(self):
        return (
            Parameter(
                name="Energy",
                bounds=self.prescribed_domain,
                extrapolable_bounds=self.extrapolable_domain,
                unit="keV",
            ),
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rx = (
        "T(d,n)4He",
        "D(d,n)3He",
        "D(d,p)T",
        "3He(d,p)4He",
    )

    for r in rx:
        endf = ENDFCrossSection(r)

        safex = np.geomspace(*endf.prescribed_domain, 100)
        extendx = np.geomspace(*endf.extrapolable_domain, 100)

        plt.loglog(safex, endf(safex))
        plt.loglog(extendx, endf(extendx), ls="dashed")

    plt.show()
