import numpy as np
from simulator import Box, Detector, Hotspot, simulate_measured_activity

def test_simulator_shapes_and_nonnegative():
    box = Box(Lx=0.1, Ly=0.05, Lz=0.1)
    det = Detector(area_m2=1e-4, efficiency=0.2, background_cps=1.0, dwell_s=1.0)
    hs = Hotspot(width_x_m=0.05, depth_y_m=0.01, height_z_m=0.05, mean_activity_bq=1e5, size_sigma_m=0.002)
    d = np.linspace(0.01, 0.2, 15)
    expected, measured = simulate_measured_activity(d, box=box, detector=det, hotspot=hs, noise="none")
    assert expected.shape == (15,)
    assert measured.shape == (15,)
    assert np.all(expected >= 0)
    assert np.all(measured >= 0)
