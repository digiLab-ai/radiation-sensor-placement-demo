import numpy as np

from simulator import Box, Detector
from simulator.batch import run_design


def test_run_design_can_use_activity_as_the_only_qoi():
    box = Box(Lx=0.05, Ly=0.05, Lz=0.12)
    detector = Detector(area_m2=1e-4, efficiency=0.2, background_cps=1.0, dwell_s=1.0)
    sensor_positions = np.array(
        [
            [box.Lx / 2.0, -0.10, box.Lz / 2.0],
            [box.Lx / 2.0, -0.02, box.Lz / 2.0],
        ]
    )

    inputs_df, measurements_df = run_design(
        sensor_positions_m=sensor_positions,
        sensor_names=["S1", "S2"],
        box=box,
        detector=detector,
        bounds={"mean_activity_bq": (1e4, 2e4), "height_z_m": (0.02, 0.08)},
        fixed_params={
            "width_x_m": box.Lx / 2.0,
            "depth_y_m": box.Ly / 2.0,
            "size_sigma_m": 0.003,
        },
        n_samples=8,
        strategy="lhs",
        seed=3,
        noise="none",
    )

    assert list(inputs_df.columns) == [
        "mean_activity_bq",
        "height_z_m",
        "width_x_m",
        "depth_y_m",
        "size_sigma_m",
    ]
    assert np.all(inputs_df["width_x_m"] == box.Lx / 2.0)
    assert np.all(inputs_df["depth_y_m"] == box.Ly / 2.0)
    assert np.all(inputs_df["height_z_m"].between(0.02, 0.08))
    assert np.all(inputs_df["mean_activity_bq"].between(1e4, 2e4))
    assert list(measurements_df.columns) == ["S1", "S2"]
    assert measurements_df.shape == (8, 2)
