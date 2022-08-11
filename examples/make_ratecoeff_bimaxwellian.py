import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fusionrate.endf import ENDFCrossSection
from fusionrate.load_data import save_ratecoeff_hdf5
from fusionrate.ratecoefficient import RateCoefficientIntegratorBiMaxwellian
from fusionrate.reaction import ReactionCore

from fusionrate.reactionnames import ALL_REACTIONS

min_log10_temp = -2.0
max_log10_temp = 4.0
temperatures = np.logspace(min_log10_temp, max_log10_temp, 10)

t1, t2 = np.meshgrid(temperatures, temperatures)


def temperature_limits(temperatures: np.array):
    return (
        min_log10_temp,
        max_log10_temp,
    )


def ratecoeff_data_2d(rc: ReactionCore):
    cs = ENDFCrossSection(rc)

    mwrc = RateCoefficientIntegratorBiMaxwellian(
        rc, cs.cross_section, relerr=1e-6, maxeval=3e7, h=12, extramult=1e50
    )
    ratecoeff = mwrc.ratecoeff(t1, t2)
    return ratecoeff


def plot_check_2d(t, sv):
    logsv = np.log10(sv)
    plt.contourf(np.log10(t1), np.log10(t2), logsv)
    plt.show()


def generate_and_store_ratecoeff_data_2d(reaction: str):
    rc = ReactionCore(reaction)
    canonical_name = rc.canonical_name()

    ratecoeffs = ratecoeff_data_2d(rc)
    t_lims = temperature_limits(temperatures)

    current_time = datetime.datetime.now().isoformat()

    # save_ratecoeff_hdf5(
    #     canonical_name=canonical_name,
    #     distribution="BiMaxwellian",
    #     parameter_limits=(t_lims, t_lims),
    #     parameter_units=("keV","keV"),
    #     parameter_descriptions=("T_perpendicular","T_parallel"),
    #     parameter_space_descriptions=("Log10", "Log10"),
    #     rate_coefficients=ratecoeffs,
    #     data_units="cm³/s",
    #     time_generated=current_time,
    # )

    plot_check_2d(temperatures, ratecoeffs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generate_and_store_ratecoeff_data_2d("D+T")
