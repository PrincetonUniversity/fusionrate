from fusrate.halebosch import HaleBoschReactivity

import matplotlib.pyplot as plt
import numpy as np

line_styles = ['solid', '-.', 'solid','dashed',]

fig, ax = plt.subplots(1, 2)

def plot_cs(reaction):
    cs = HaleBoschReactivity(reaction)
    e_range = cs.prescribed_range()

    e = np.logspace(*np.log10(e_range), 100)

    x_hb = e
    y_hb = cs.reactivity(e)/1e6/e**2
    ax[0].plot(x_hb, y_hb, label=reaction, ls=line_styles[i])
    ax[1].plot(x_hb, y_hb, label=reaction, ls=line_styles[i])

reactions = ['T(d,n)⁴He', '³He(d,p)⁴He', 'D(d,n)³He', 'D(d,p)T']
for i, reaction in enumerate(reactions):
    plot_cs(reaction)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].grid()
ax[0].set_xlim([0.1, 2e2])
ax[0].set_ylim([1e-35,1e-23])
ax[0].set_xlabel('Temperature/keV')
ax[0].set_ylabel('Reactivity / (m³/s · keV²)')

ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].grid(which='both', color='lightgray')
ax[1].set_xlim([1e0,2e2])
ax[1].set_ylim([1e-27,1e-23])
ax[1].set_xlabel('Temperature/keV')
ax[1].set_ylabel('Reactivity / (m³/s · keV²)')

ax[0].legend()
plt.show()
