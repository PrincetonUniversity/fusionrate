# Here I show how the cross section extrapolation works
# It is linear in log-log space

from fusrate.load_data import load_data_file
from fusrate.endf import LogLogExtrapolation

import numpy as np
import matplotlib.pyplot as plt

default_data_dir = 'fusrate.data'

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-99, 1e1])
ax.set_xlabel('Beam-target energy/eV')
ax.set_ylabel('Cross section/barns')

def plot_some_data(name):
    x, y = load_data_file(name)
    lle = LogLogExtrapolation(x, y, linear_extension=True)

    newx = np.logspace(-2, 10, 1000)
    ax.plot(x, y)
    ax.plot(newx, lle(newx), ls='dashed')

plot_some_data('cross_section_h(h,2p)a.csv')
plot_some_data('cross_section_t(t,2n)a.csv')

ax.set_ylim([1e-9, 1e1])
plt.show()
