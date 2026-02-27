from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .core import Box, Detector, Hotspot, simulate_measured_activity
from .sampling import sample_inputs


def run_design(
    *,
    distances_m: np.ndarray,
    box: Box,
    detector: Detector,
    bounds: Dict[str, Tuple[float, float]],
    n_samples: int,
    strategy: str = "lhs",
    seed: int = 0,
    mu_material_m_inv: float = 0.0,
    fov_half_angle_deg: Optional[float] = None,
    distance_offset_m: float = 0.0,
    noise: str = "poisson",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce two DataFrames:
      1) inputs_df: sampled hotspot parameters
      2) measurements_df: measured cps at each candidate distance
    """
    distances_m = np.asarray(distances_m, dtype=float).ravel()
    if distances_m.size == 0:
        raise ValueError("distances_m must be non-empty")

    sampled = sample_inputs(bounds, n=n_samples, strategy=strategy, seed=seed)
    inputs_df = pd.DataFrame(sampled)

    meas_cols = [f"cps_d_{d:.6g}m" for d in distances_m]
    meas = np.empty((len(inputs_df), len(distances_m)), dtype=float)

    rng = np.random.default_rng(seed + 12345)
    for i in range(len(inputs_df)):
        hs = Hotspot(
            width_x_m=float(inputs_df.loc[i, "width_x_m"]),
            depth_y_m=float(inputs_df.loc[i, "depth_y_m"]),
            height_z_m=float(inputs_df.loc[i, "height_z_m"]),
            mean_activity_bq=float(inputs_df.loc[i, "mean_activity_bq"]),
            size_sigma_m=float(inputs_df.loc[i, "size_sigma_m"]),
        )
        _, measured = simulate_measured_activity(
            distances_m,
            box=box,
            detector=detector,
            hotspot=hs,
            mu_material_m_inv=mu_material_m_inv,
            fov_half_angle_deg=fov_half_angle_deg,
            distance_offset_m=distance_offset_m,
            noise=noise,  # type: ignore[arg-type]
            rng=rng,
        )
        meas[i, :] = measured

    measurements_df = pd.DataFrame(meas, columns=meas_cols)
    return inputs_df, measurements_df
