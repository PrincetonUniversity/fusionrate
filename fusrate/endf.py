from fusrate.load_data import load_data_file
from scipy.interpolate import interp1d
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

        self.interpolator = interp1d(self.logx, self.logy, kind='quadratic',
                fill_value='extrapolate', assume_sorted=True)

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

    default_data_dir = 'fusrate.data'
    data_name = 'cross_section_dt.csv'
    x, y = load_data_file(data_name)
    lle = LogLogExtrapolation(x, y, linear_extension=True)

    newx = np.logspace(4, 12, 80)

    print(lle(newx))


