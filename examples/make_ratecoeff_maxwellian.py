import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fusionrate.endf import ENDFCrossSection
from fusionrate.load_data import save_ratecoeff_hdf5
from fusionrate.integrators import rate_coefficient_integrator_factory
from fusionrate.reaction import ReactionCore
from fusionrate.reaction import Reaction

from fusionrate.reactionnames import ALL_REACTIONS

min_temp = 10**-2
max_temp = 10**4
temperatures = np.geomspace(min_temp, max_temp, 300)


def temperature_limits(temperatures: np.array):
    return (
        np.log10(min_temp),
        np.log10(max_temp),
    )


def ratecoeff_data_1d(r: Reaction):
    integrator = r.get_rate_coefficient_object("Maxwellian", "integration")
    integrator.h = 25
    integrator.relerr = 1e-8
    integrator.maxeval = 1e7

    ratecoeff = r.rate_coefficient(temperatures, scheme='integration')
    return ratecoeff


def plot_check_1d(t, sv):
    plt.loglog(t, sv)
    plt.show()


def generate_and_store_ratecoeff_data_1d(reaction: str):
    r = Reaction(reaction)
    canonical_name = r.name
    ratecoeffs = ratecoeff_data_1d(r)

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
