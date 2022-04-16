from fusrate.load_data import load_data_file
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

class LogLogExtrapolation():
    r"""Interpolate and extrapolate in log-log space

    with a straight-line end in log-log space
    """
    SMALL = 1e-50 # to prevent errors with log of 0

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
            Δ = (last_two[-1] - last_two[-2])
            subsequent_points = last + np.outer([1,2,3], Δ)
            self.data = np.append(data, subsequent_points, axis=0)
        else:
            self.data = data

        self.logx, self.logy = self.data.T

        #self.interpolator = interp1d(self.logx, self.logy, kind='quadratic',
        #        fill_value='extrapolate', assume_sorted=True)
        self.interpolator = InterpolatedUnivariateSpline(self.logx, self.logy, k=2, ext=0)

    def __call__(self, newx):
        r"""Generate new values
        newx: array_like
        """
        log_new = np.log(newx + self.SMALL)
        log_newy = self.interpolator(log_new)
        return np.exp(log_newy)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import cProfile

    data_name = 'cross_section_t(d,n)a.csv'
    x, y = load_data_file(data_name)
    lle = LogLogExtrapolation(x, y, linear_extension=True)

    newx = np.logspace(4, 12, 80)

    print(lle(newx))



