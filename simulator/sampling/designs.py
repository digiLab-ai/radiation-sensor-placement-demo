from __future__ import annotations

from typing import Dict, Literal, Tuple
import numpy as np

Strategy = Literal["random", "lhs", "sobol"]


def _scale_unit_to_bounds(u: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = list(bounds.keys())
    for j, k in enumerate(keys):
        lo, hi = bounds[k]
        if hi <= lo:
            raise ValueError(f"Invalid bounds for {k}: ({lo}, {hi})")
        out[k] = lo + u[:, j] * (hi - lo)
    return out


def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    u = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


def _sobol_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    try:
        from scipy.stats import qmc  # type: ignore
        eng = qmc.Sobol(d=d, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
        m = int(np.ceil(np.log2(max(n, 1))))
        u = eng.random_base2(m=m)
        if u.shape[0] >= n:
            return u[:n]
        u2 = eng.random(n - u.shape[0])
        return np.vstack([u, u2])
    except Exception:
        return rng.random((n, d))


def sample_inputs(
    bounds: Dict[str, Tuple[float, float]],
    *,
    n: int,
    strategy: Strategy = "lhs",
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Sample parameters within bounds using random / LHS / Sobol."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if not bounds:
        raise ValueError("bounds must be non-empty")

    rng = np.random.default_rng(seed)
    d = len(bounds)

    if strategy == "random":
        u = rng.random((n, d))
    elif strategy == "lhs":
        u = _lhs_unit(n=n, d=d, rng=rng)
    elif strategy == "sobol":
        u = _sobol_unit(n=n, d=d, rng=rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return _scale_unit_to_bounds(u, bounds)
