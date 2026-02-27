import numpy as np
from simulator.sampling import sample_inputs

def test_sampling_within_bounds():
    bounds = {"a": (0.0, 1.0), "b": (-2.0, 2.0)}
    out = sample_inputs(bounds, n=128, strategy="lhs", seed=1)
    assert out["a"].shape == (128,)
    assert out["b"].shape == (128,)
    assert np.all(out["a"] >= 0.0) and np.all(out["a"] <= 1.0)
    assert np.all(out["b"] >= -2.0) and np.all(out["b"] <= 2.0)
