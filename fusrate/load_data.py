from fusrate.reactionnames import reaction_name_simplify
from fusrate.reactionnames import reaction_name_to_endf

from importlib import resources

import numpy as np

DEFAULT_DATA_DIR = "fusrate.data"
CROSS_SECTION_PREFIX = "cross_section_"
CROSS_SECTION_FILETYPE = ".csv"


def locate_data_file(dname):
    if resources.is_resource(DEFAULT_DATA_DIR, dname):
        with resources.path(DEFAULT_DATA_DIR, dname) as f:
            return f
    else:
        raise FileNotFoundError(dname + " not found in data directory.")


def load_data_file(dname):
    r"""Loads a 2-column csv file"""
    path = locate_data_file(dname)
    return np.loadtxt(path, delimiter=",").T


def cross_section_filename(canonical_reaction_name):
    s = reaction_name_simplify(canonical_reaction_name)
    s = reaction_name_to_endf(s)
    return CROSS_SECTION_PREFIX + s + CROSS_SECTION_FILETYPE


def cross_section_data(canonical_reaction_name):
    filename = cross_section_filename(canonical_reaction_name)
    return load_data_file(filename)


if __name__ == "__main__":
    from reactionnames import DT_NAME

    print(cross_section_data(DDHE3_NAME).T)
