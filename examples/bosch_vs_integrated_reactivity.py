import matplotlib.pyplot as plt
import numpy as np

from fusrate.bosch import BoschRateCoeff
from fusrate.ratecoefficient import RateCoefficientInterpolator
from fusrate.reaction import Reaction


line_styles = [
    "solid",
    "-.",
    "solid",
    "dashed",
]

fig, ax = plt.subplots(1, 1)

meters_cubed_in_cm_cubed = 1e6

def plot_comparison(reaction):
    sv = BoschRateCoeff(reaction)
    t_range = sv.prescribed_range()

    r = Reaction(reaction)
    interp = RateCoefficientInterpolator(r.canonical_name(), "Maxwellian")

    t = np.logspace(*np.log10(t_range), 100)

    x = t
    y_b = sv.ratecoeff(t) / meters_cubed_in_cm_cubed
    y_i = interp.rate_coefficient(t) / meters_cubed_in_cm_cubed
    ax.plot(x, y_i/y_b, label=reaction, ls=line_styles[i])

reactions = ["T(d,n)⁴He", "³He(d,p)⁴He", "D(d,n)³He", "D(d,p)T"]
for i, reaction in enumerate(reactions):
    plot_comparison(reaction)

ax.set_xscale("log")
ax.set_yscale("linear")
ax.grid()
ax.set_ylim(0, 2)
ax.set_xlabel("Temperature/keV")
ax.set_ylabel("Rate coefficient / (m³/s)")
ax.set_title("Wide-range plot")
ax.legend()
plt.show()
