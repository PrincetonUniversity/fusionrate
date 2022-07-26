import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fusrate.endf import ENDFCrossSection
from fusrate.load_data import save_ratecoeff_hdf5
from fusrate.ratecoefficient import RateCoefficientIntegratorMaxwellian
from fusrate.reaction import ReactionCore

from fusrate.reactionnames import ALL_REACTIONS

min_log10_temp = -2.0
max_log10_temp = 4.0
temperatures = np.logspace(min_log10_temp, max_log10_temp, 300)


def temperature_limits(temperatures: np.array):
    return (
        min_log10_temp,
        max_log10_temp,
    )


def ratecoeff_data_1d(rc: ReactionCore):
    cs = ENDFCrossSection(rc, "LogLogExtrapolation")
    mwrc = RateCoefficientIntegratorMaxwellian(
        rc, cs.cross_section, relerr=1e-8, maxeval=1e7, h=25
    )
    ratecoeff = mwrc.ratecoeff(temperatures)
    return ratecoeff


def plot_check_1d(t, sv):
    plt.loglog(t, sv)
    plt.show()


def generate_and_store_ratecoeff_data_1d(reaction: str):
    rc = ReactionCore(reaction)
    canonical_name = rc.canonical_name()

    ratecoeffs = ratecoeff_data_1d(rc)

    current_time = datetime.datetime.now().isoformat()

    save_ratecoeff_hdf5(
        canonical_name=canonical_name,
        distribution="Maxwellian",
        parameter_limits=(temperature_limits(temperatures),),
        parameter_units=("keV",),
        parameter_descriptions=("Temperatures",),
        parameter_space_descriptions=("Log10",),
        rate_coefficients=ratecoeffs,
        data_units="cm³/s",
        time_generated=current_time,
    )

    plot_check_1d(temperatures, ratecoeffs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    reactions = ALL_REACTIONS[:4]

    for r in reactions:
        generate_and_store_ratecoeff_data_1d(r)
