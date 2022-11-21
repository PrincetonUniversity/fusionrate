import numpy as np

def no_nans(val):
    assert not np.any(np.isnan(val))

def has_nans(val):
    assert np.any(np.isnan(val))

def has_zeros(val):
    assert np.any(val == 0.0)

def has_negs(val):
    assert np.any(val < 0.0)

def has_infs(val):
    assert np.any(np.isinf(val))

def all_finite(val):
    assert np.all(np.isfinite(val))

def all_nonneg(val):
    assert np.all(val >= 0.0)
