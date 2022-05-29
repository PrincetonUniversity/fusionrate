from fusrate.load_data import load_data_file
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

import numba


class LogLogExtrapolation:
    r"""Interpolate and extrapolate in log-log space

    with a straight-line end in log-log space
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
            subsequent_points = last + np.outer([1, 2, 3], Δ)
            self.data = np.append(data, subsequent_points, axis=0)
        else:
            self.data = data

        self.logx, self.logy = self.data.T

        self.interpolator = InterpolatedUnivariateSpline(
            self.logx, self.logy, k=2, ext=0
        )

    def __call__(self, newx):
        r"""Generate new values
        newx: array_like
        """
        log_new = np.log(newx + self.SMALL)
        log_newy = self.interpolator(log_new)
        return np.exp(log_newy)


class LogLogReinterpolation:
    r"""Interpolate and extrapolate in log-log space

    with a straight-line end in log-log space
    """
    SMALL = 1e-50
    LOWBOUND = 10
    HIGHBOUND = 1e11
    NUM_REMESH = 6000
    REMESHED_LOGX = np.linspace(
        np.log(LOWBOUND), np.log(HIGHBOUND), NUM_REMESH
    )

    def __init__(self, x, y, linear_extension=True):
        r"""
        x: array_like
        y: array_like
        """
        lle = LogLogExtrapolation(x, y, linear_extension)
        self.remeshed_logy = lle.interpolator(self.REMESHED_LOGX)

    def __call__(self, newx):
        r"""Generate new values
        newx: array_like
        """
        log_new = np.log(newx)
        log_newy = np.interp(log_new, self.REMESHED_LOGX, self.remeshed_logy,
                right=self.SMALL)
        return np.exp(log_newy)

    def make_jitfunction(self):
        small = self.SMALL
        remesh_logx = self.REMESHED_LOGX
        remesh_logy = self.remeshed_logy

        @numba.njit(cache=True, fastmath=True)
        def logloginterp(newx):
            log_new = np.log(newx + small)
            log_newy = np.interp(log_new, remesh_logx, remesh_logy)
            return np.exp(log_newy)

        return logloginterp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cProfile

    data_name = "cross_section_t(d,n)a.csv"
    x, y = load_data_file(data_name)
    lle = LogLogExtrapolation(x, y, linear_extension=True)
    llr = LogLogReinterpolation(x, y, linear_extension=True)

    llr_jit = llr.make_jitfunction()

    newx = np.logspace(3, 6, 100)

    plt.loglog(newx, lle(newx))
    plt.loglog(newx, llr(newx))
    plt.loglog(newx, llr_jit(newx))
    plt.show()
