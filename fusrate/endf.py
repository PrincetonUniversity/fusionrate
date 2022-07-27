import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import fusrate.reactionnames as rn
from fusrate.ion_data import ion_mass
from fusrate.load_data import cross_section_data


class LogLogExtrapolation:
    r"""Interpolate and extrapolate in log-log space

    with a straight-line right end in log-log space
    """
    SMALL = 1e-50  # to prevent errors with log of 0

    def __init__(self, x, y, linear_extension=True):
        r"""
        x: array_like
        y: array_like
        """
        self.x = x
        self.y = y
        self.max_x = max(x)
        self.min_x = min(x)
        logx = np.log(x + self.SMALL)
        logy = np.log(y)
        data = np.array([logx, logy]).T

        # linear extension in loglog space
        if linear_extension:
            last_two = data[-2:]
            last = last_two[-1]
            Δ = last_two[-1] - last_two[-2]
            subsequent_points = last + np.outer(range(1,4), Δ)
            self.data = np.append(data, subsequent_points, axis=0)
        else:
            self.data = data

        self.logx, self.logy = self.data.T

        self.interpolator = InterpolatedUnivariateSpline(
            self.logx, self.logy, k=2, ext=0
        )

    def __call__(self, newx):
        r"""Generate new values

        Parameters
        ----------
        newx: array_like
            Energies in eV.
            Whether this is beam-target energy or c.o.m. energy
            is up to the data.
        """
        log_new = np.log(newx + self.SMALL)
        log_newy = self.interpolator(log_new)
        return np.exp(log_newy)


class ENDFCrossSection:
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
            name = r.name
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

    def cross_section(self, e, derivatives=False):
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

    def prescribed_range(self):
        r"""
        Returns
        -------
        [min, max] of COM energy range in keV
        """
        return [min(self.x), max(self.x)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fusrate.load_data import load_data_file

    endf = ENDFCrossSection("D+T")

    newx = np.logspace(0, 3, 100)

    plt.loglog(newx, endf(newx))
    plt.show()
