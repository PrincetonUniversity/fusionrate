from fusrate.halebosch import HaleBoschCrossSection

import matplotlib.pyplot as plt
import numpy as np

line_styles = ['solid', '-.', 'solid','dashed',]

fig, ax = plt.subplots(1, 2)

def plot_cs(reaction):
    cs = HaleBoschCrossSection(reaction)
    e_range = cs.prescribed_range()

    e = np.logspace(*np.log10(e_range), 100)

    x_hb = e*1000
    y_hb = cs.cross_section(e)/1000
    ax[0].plot(x_hb, y_hb, label=reaction, ls=line_styles[i])
    ax[1].plot(x_hb, y_hb, label=reaction, ls=line_styles[i])

reactions = ['T(d,n)⁴He', '³He(d,p)⁴He', 'D(d,n)³He', 'D(d,p)T']
for i, reaction in enumerate(reactions):
    plot_cs(reaction)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].grid()
ax[0].set_xlim([1e3,1e7])
ax[0].set_ylim([1e-25,1e1])
ax[0].set_xlabel('COM energy/eV')
ax[0].set_ylabel('Cross section/barns')
ax[0].set_title('Wide-range plot')

ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].grid(which='both', color='lightgray')
ax[1].set_xlim([1e4,1e6])
ax[1].set_ylim([1e-3,1e1])
ax[1].set_xlabel('COM energy/eV')
ax[1].set_ylabel('Cross section/barns')
ax[1].set_title('Zoom-in on peaks')

ax[0].legend()
plt.show()
