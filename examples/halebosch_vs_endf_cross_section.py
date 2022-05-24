# Here I show that the endf format cross sections
# are specified in terms of beam-on-stationary-target energies.
# We need the beam-target to COM correction factor.
# Here the 'tritium' is the 'target'.

# The first plot demonstrates that these two align when subjected to the
# transformation.
# The second plot compares the two (since they are quite close.)
# The dashed line represents the region outside the stated validity of the
# Hale-Bosch formula.
from fusrate.halebosch import HaleBoschCrossSection
from fusrate.load_data import load_data_file

import matplotlib.pyplot as plt
import numpy as np

keV_TO_eV = 1000
millibarns_TO_barns = 0.001

m_deu = 2.01355321
m_trit = 3.01550069
beam_target_to_com = m_trit / (m_deu + m_trit)

# center of mass energies
e = np.logspace(-1, np.log10(550), 100)
e2 = np.logspace(np.log10(550), 8, 100)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
ax.set_xlim([1e2, 1e8])
ax.set_ylim([1e-25, 1e1])
ax.set_xlabel("COM energy/eV")
ax.set_ylabel("Cross section")

# Hale-Bosch data
cs = HaleBoschCrossSection("DT")
x_hb = e * keV_TO_eV
y_hb = cs.cross_section(e) * millibarns_TO_barns

ax.plot(x_hb, y_hb, label="Hale-Bosch")

# Extension beyond the indicated safe range
x_hb = e2 * keV_TO_eV
y_hb = cs.cross_section(e2) * millibarns_TO_barns
ax.plot(x_hb, y_hb, color="tab:blue", ls="dashed")

# ENDF data
endf_data = load_data_file("cross_section_t(d,n)a.csv")
x_endf, y_endf = endf_data

ax.plot(x_endf * beam_target_to_com, y_endf, label="ENDF")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_yscale('linear')
ax.set_xlim([1e2, 1e8])
ax.set_ylim([0.5, 1.1])
ax.set_xlabel("COM energy/eV")
ax.set_ylabel("Relative cross section: Hale-Bosch/ENDF")

x_converted = x_endf * beam_target_to_com
x_converted_keV = x_converted / 1000

y_hb_millibarns = cs.cross_section(x_converted_keV)
y_hb_barns = y_hb_millibarns / 1000

valid = x_endf < 550e3
invalid = x_endf > 550e3
ax.plot(x_endf[valid], y_hb_barns[valid] / y_endf[valid])
ax.plot(
    x_endf[invalid],
    y_hb_barns[invalid] / y_endf[invalid],
    ls="dashed",
    color="tab:blue",
)

plt.tight_layout()
plt.show()
