# Here I show how the cross section extrapolation works
# It is linear in log-log space

from fusrate.load_data import load_data_file
from fusrate.endf import LogLogExtrapolation

import numpy as np
import matplotlib.pyplot as plt

default_data_dir = 'fusrate.data'
data_name = 'cross_section_dt.csv'
x, y = load_data_file(data_name)
lle = LogLogExtrapolation(x, y, linear_extension=True)

newx = np.logspace(-4, 15, 100)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-99, 1e1])
ax.plot(x, y)
ax.plot(newx, lle(newx), ls='dashed')

plt.show()
