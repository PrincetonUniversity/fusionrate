import json
from importlib import resources

from fusrate.load_data import DEFAULT_DATA_DIR

__all__ = ["ion_mass"]

with resources.path(DEFAULT_DATA_DIR, "ions.json") as f:
    with open(f, "rb") as s:
        ion_data = json.load(s)


def ion_mass(s):
    r"""
    Parameters
    ----------
    s: string
       One of the canonical ion names

    Returns
    -------
    Mass in amu
    """
    return ion_data[s]["mass"]


if __name__ == "__main__":
    print(ion_data)
