from importlib import resources

import numpy as np

DEFAULT_DATA_DIR = 'fusrate.data'

def locate_data_file(dname):
    if resources.is_resource(DEFAULT_DATA_DIR, dname):
        with resources.path(DEFAULT_DATA_DIR, dname) as f:
            return f
    else:
        raise FileNotFoundError(dname + " not found in data directory.")

def load_data_file(dname):
    r"""Loads a 2-column csv file
    """
    path = locate_data_file(dname)
    return np.loadtxt(path, delimiter=',').T


if __name__=='__main__':
    data_name = 'cross_section_t(d,n)a.csv'
    print(load_data_file(data_name).T)

