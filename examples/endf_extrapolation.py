# Here I show how the cross section extrapolation works
# It is linear in log-log space
import matplotlib.pyplot as plt
import numpy as np

from fusrate.endf import LogLogExtrapolation
from fusrate.load_data import cross_section_data

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([1e-59, 1e1])
ax.set_xlabel("Beam-target energy/eV")
ax.set_ylabel("Cross section/barns")


def plot_some_data(label):
    x, y = cross_section_data(label)
    lle = LogLogExtrapolation(x, y, linear_extension=True)

    newx = np.logspace(-1, 8, 1000)
    p = ax.plot(x, y, label=label)
    c = p[0].get_color()
    ax.plot(newx, lle(newx), ls="dashed", color=c)


plot_some_data("t(d,n)α")
plot_some_data("d(d,n)h")
plot_some_data("d(d,p)t")
plot_some_data("h(d,p)α")
plot_some_data("t(t,2n)α")
plot_some_data("h(t,pn)α")
plot_some_data("h(h,2p)α")
plot_some_data("⁶Li(d,α)α")
plot_some_data("⁶Li(d,n)⁷Be")
plot_some_data("⁶Li(d,p)⁷Li")

ax.legend()
plt.show()
