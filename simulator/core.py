from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned box (sample) with fixed size, metres."""
    Lx: float  # width (x)
    Ly: float  # depth (y)  (front face is at y=0)
    Lz: float  # height (z)


@dataclass(frozen=True)
class Detector:
    """
    Simple planar detector facing the sample's front face (y=0).
    The detector plane is parallel to the x–z plane, centered on the box center in x,z.
    """
    area_m2: float = 1e-4
    efficiency: float = 0.25
    background_cps: float = 2.0
    dwell_s: float = 1.0


@dataclass(frozen=True)
class Hotspot:
    """
    Hotspot inside the box.

    Coordinates are metres, in the box frame:
      x in [0, Lx], y in [0, Ly], z in [0, Lz]
    with the detector located in front of the y=0 face (negative y direction).
    """
    width_x_m: float
    depth_y_m: float
    height_z_m: float
    mean_activity_bq: float
    size_sigma_m: float


def _validate_params(box: Box, det: Detector, hs: Hotspot) -> None:
    for name, v in [("Lx", box.Lx), ("Ly", box.Ly), ("Lz", box.Lz)]:
        if v <= 0:
            raise ValueError(f"Box.{name} must be > 0, got {v}")

    if not (0 <= det.efficiency <= 1):
        raise ValueError(f"Detector.efficiency must be in [0,1], got {det.efficiency}")
    if det.area_m2 <= 0:
        raise ValueError(f"Detector.area_m2 must be > 0, got {det.area_m2}")
    if det.dwell_s <= 0:
        raise ValueError(f"Detector.dwell_s must be > 0, got {det.dwell_s}")
    if det.background_cps < 0:
        raise ValueError(f"Detector.background_cps must be >= 0, got {det.background_cps}")
    if hs.mean_activity_bq < 0:
        raise ValueError(f"Hotspot.mean_activity_bq must be >= 0, got {hs.mean_activity_bq}")
    if hs.size_sigma_m <= 0:
        raise ValueError(f"Hotspot.size_sigma_m must be > 0, got {hs.size_sigma_m}")

    if not (0 <= hs.width_x_m <= box.Lx and 0 <= hs.depth_y_m <= box.Ly and 0 <= hs.height_z_m <= box.Lz):
        raise ValueError(
            "Hotspot must lie within the box: "
            f"x={hs.width_x_m} in [0,{box.Lx}], "
            f"y={hs.depth_y_m} in [0,{box.Ly}], "
            f"z={hs.height_z_m} in [0,{box.Lz}]"
        )


def simulate_measured_activity(
    distances_m: Optional[np.ndarray],
    *,
    box: Box,
    detector: Detector,
    hotspot: Hotspot,
    sensor_positions_m: Optional[np.ndarray] = None,
    mu_material_m_inv: float = 0.0,
    fov_half_angle_deg: Optional[float] = None,
    distance_offset_m: float = 0.0,
    noise: Literal["none", "poisson"] = "poisson",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate activity measurements at a set of detector locations.

    Model (small-detector approximation):
      expected_cps = background_cps + signal_cps

      signal_cps ≈ (A_total * efficiency) * (Omega / (4π)) * exp(-mu * path_in_material)

    where:
      Omega ≈ detector.area * cos(theta) / r^2
      r = distance from hotspot to detector center
      theta = angle between detector normal (+y) and direction to hotspot

    Inputs:
      - distances_m: legacy front-of-box detector standoff distances
      - sensor_positions_m: explicit detector centers as an (N, 3) array in metres

    Options:
      - FOV half-angle: if provided, reject signal outside cone
      - Hotspot size: near-field smoothing by r_eff^2 = r^2 + sigma^2
      - Poisson noise: counts ~ Poisson(expected_cps * dwell_s)

    Returns
    -------
    expected_cps : (N,) float
    measured_cps : (N,) float
    """
    _validate_params(box, detector, hotspot)

    if rng is None:
        rng = np.random.default_rng()

    if mu_material_m_inv < 0:
        raise ValueError(f"mu_material_m_inv must be >= 0, got {mu_material_m_inv}")

    if sensor_positions_m is not None:
        sensor_positions = np.asarray(sensor_positions_m, dtype=float)
        if sensor_positions.ndim != 2 or sensor_positions.shape[1] != 3:
            raise ValueError("sensor_positions_m must have shape (N, 3)")
        det_x = sensor_positions[:, 0]
        det_y = sensor_positions[:, 1] - float(distance_offset_m)
        det_z = sensor_positions[:, 2]
    else:
        if distances_m is None:
            raise ValueError("Either distances_m or sensor_positions_m must be provided")
        d = np.asarray(distances_m, dtype=float).ravel()
        if d.size == 0:
            raise ValueError("distances_m must be non-empty")
        if np.any(d < 0):
            raise ValueError("distances_m must be >= 0")
        det_x = np.full_like(d, box.Lx / 2.0, dtype=float)
        det_z = np.full_like(d, box.Lz / 2.0, dtype=float)
        det_y = -(d + float(distance_offset_m))

    hx, hy, hz = hotspot.width_x_m, hotspot.depth_y_m, hotspot.height_z_m

    dx = hx - det_x
    dy = hy - det_y
    dz = hz - det_z

    r2 = dx * dx + dy * dy + dz * dz

    sigma2 = float(hotspot.size_sigma_m) ** 2
    r2_eff = r2 + sigma2
    r_eff = np.sqrt(r2_eff)

    cos_theta = dy / np.maximum(r_eff, 1e-12)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    if fov_half_angle_deg is not None:
        half_angle = np.deg2rad(float(fov_half_angle_deg))
        cos_max = np.cos(half_angle)
        in_fov = cos_theta >= cos_max
    else:
        in_fov = np.ones_like(cos_theta, dtype=bool)

    omega = detector.area_m2 * cos_theta / np.maximum(r2_eff, 1e-18)
    omega = np.clip(omega, 0.0, 4.0 * np.pi)

    path_in_material = hotspot.depth_y_m / np.maximum(cos_theta, 1e-12)
    attenuation = np.exp(-mu_material_m_inv * np.clip(path_in_material, 0.0, 1e6))

    signal_cps = (hotspot.mean_activity_bq * detector.efficiency) * (omega / (4.0 * np.pi)) * attenuation
    signal_cps = np.where(in_fov, signal_cps, 0.0)

    expected_cps = detector.background_cps + signal_cps

    if noise == "none":
        measured_cps = expected_cps.copy()
    elif noise == "poisson":
        lam = np.maximum(expected_cps * detector.dwell_s, 0.0)
        counts = rng.poisson(lam=lam)
        measured_cps = counts / detector.dwell_s
    else:
        raise ValueError(f"Unknown noise='{noise}'")

    return expected_cps, measured_cps
