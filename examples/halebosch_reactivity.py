from fusrate.halebosch import HaleBoschReactivity

import matplotlib.pyplot as plt
import numpy as np

line_styles = ['solid', '-.', 'solid','dashed',]

fig, ax = plt.subplots(1, 2)

meters_cubed_in_cm_cubed = 1e6

def plot_sv(reaction):
    sv = HaleBoschReactivity(reaction)
    t_range = sv.prescribed_range()

    t = np.logspace(*np.log10(t_range), 100)

    x_hb = t
    y_hb = sv.reactivity(t)/meters_cubed_in_cm_cubed
    ax[0].plot(x_hb, y_hb, label=reaction, ls=line_styles[i])
    ax[1].plot(x_hb, y_hb, label=reaction, ls=line_styles[i])

reactions = ['T(d,n)⁴He', '³He(d,p)⁴He', 'D(d,n)³He', 'D(d,p)T']
for i, reaction in enumerate(reactions):
    plot_sv(reaction)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].grid()
ax[0].set_xlim([0.1, 2e2])
ax[0].set_ylim([1e-38,1e-18])
ax[0].set_xlabel('Temperature/keV')
ax[0].set_ylabel('Reactivity/ (m³/s)')
ax[0].set_title('Wide-range plot')

ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].grid(which='both', color='lightgray')
ax[1].set_xlim([1e1,2e2])
ax[1].set_ylim([1e-25,1e-21])
ax[1].set_xlabel('Temperature/keV')
ax[1].set_ylabel('Reactivity (m³/s)')
ax[1].set_title('Zoom-in on peaks')

ax[0].legend()
plt.show()
