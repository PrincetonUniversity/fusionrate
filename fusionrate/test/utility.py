import numpy as np

def no_nans(val):
    assert not np.any(np.isnan(val))

def has_nans(val):
    assert np.any(np.isnan(val))
