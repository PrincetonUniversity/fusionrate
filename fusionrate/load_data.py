from importlib import resources
from pathlib import Path

import appdirs
import h5py
import numpy as np

from fusionrate.reactionnames import reaction_filename_part
from fusionrate.constants import PROJECT

DEFAULT_DATA_DIR = "fusionrate.data"
CROSS_SECTION_PREFIX = "cross_section_"
RATE_COEFF_PREFIX = "rate_coefficient_"
CROSS_SECTION_FILETYPE = ".csv"
INTERPOLATION_FILETYPE = ".hdf5"
RATE_COEFFICIENT_DSET = "rate_coefficients"

# should look in user data dir and then in the default data directory


def user_data_dir():
    return appdirs.user_data_dir(appname=PROJECT)


def file_in_user_dir(dname):
    return Path(user_data_dir(), dname)


def locate_data_file(dname: str):
    path = file_in_user_dir(dname)
    if path.is_file():
        return path
    elif resources.is_resource(DEFAULT_DATA_DIR, dname):
        with resources.path(DEFAULT_DATA_DIR, dname) as f:
            return f
    else:
        raise FileNotFoundError(dname + " not found in data.")


def load_data_file(dname: str):
    r"""Loads a 2-column csv file"""
    path = locate_data_file(dname)
    return np.loadtxt(path, delimiter=",").T


def cross_section_filename(canonical_reaction_name):
    s = reaction_filename_part(canonical_reaction_name)
    return f"{CROSS_SECTION_PREFIX}{s}{CROSS_SECTION_FILETYPE}"


def ratecoeff_data_exists(canonical_reaction_name: str, distribution: str):
    dname = ratecoeff_filename(canonical_reaction_name, distribution)
    try:
        locate_data_file(dname)
    except FileNotFoundError:
        return False

    return True


def ratecoeff_filename(canonical_reaction_name: str, distribution: str) -> str:
    r"""
    Parameters
    ----------
    distribution : str
    """
    s = reaction_filename_part(canonical_reaction_name)
    fname = f"{RATE_COEFF_PREFIX}{s}_{distribution}"
    return fname + INTERPOLATION_FILETYPE


def cross_section_data(canonical_reaction_name: str):
    """Loads data from file

    Parameters
    ----------
    canonical_reaction_name : string

    Returns
    -------
    np.array

    Examples
    --------
    >>> cross_section_data("T(d,n)⁴He")
    [[1.0000e+02, 2.0469e-56],
     [2.0000e+02, 7.4327e-39],
     ...
    ]
    """
    filename = cross_section_filename(canonical_reaction_name)
    return load_data_file(filename)


def load_ratecoeff_hdf5(canonical_name: str, distribution: str):
    reaction_filename = ratecoeff_filename(canonical_name, distribution)
    with locate_data_file(reaction_filename) as f:
        hdf = h5py.File(f, "r")
        dset = hdf[RATE_COEFFICIENT_DSET]
        return dset


def save_ratecoeff_hdf5(
    canonical_name: str,
    distribution: str,
    parameter_limits,
    parameter_units,
    parameter_descriptions,
    parameter_space_descriptions,
    rate_coefficients,
    data_units,
    time_generated,
):
    reaction_filename = ratecoeff_filename(canonical_name, distribution)
    full_name = file_in_user_dir(reaction_filename)

    with h5py.File(full_name, "w") as f:
        dset = f.create_dataset(RATE_COEFFICIENT_DSET, data=rate_coefficients)
        dset.attrs["Reaction"] = canonical_name
        dset.attrs["Type of data"] = reaction_filename
        dset.attrs["Data units"] = data_units
        dset.attrs["distribution"] = distribution
        dset.attrs["Parameter limits"] = parameter_limits
        dset.attrs["Parameter units"] = parameter_units
        dset.attrs["Parameter descriptions"] = parameter_descriptions
        dset.attrs[
            "Parameter space descriptions"
        ] = parameter_space_descriptions
        dset.attrs["Time generated"] = time_generated


if __name__ == "__main__":
    from reactionnames import DT_NAME

    csf = cross_section_filename(DT_NAME)
    print(type(locate_data_file(csf)))
