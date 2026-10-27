"""Run SeAn-CP versus numerical full-plane scanning on a non-proportional ellipse."""

import numpy as np
from seancp import HarmonicHistory, full_plane_scan, seancp_fs_ext

# Stress [MPa], tensorial small strain, not engineering shear strain.
# Orthogonal harmonic shear components generate a circular shear path on z.
z = np.zeros((3, 3))
E_sin = z.copy(); E_sin[0, 2] = E_sin[2, 0] = 0.0006
E_cos = z.copy(); E_cos[1, 2] = E_cos[2, 1] = 0.0006
history = HarmonicHistory(np.diag([10.0, 20.0, 140.0]), z, z,
                          z, E_sin, E_cos)

for name in ("longest_chord", "mcc", "mce"):
    restricted = seancp_fs_ext(history, algorithm=name)
    full, normal = full_plane_scan(history, algorithm=name)
    error = 100 * (restricted.value / full - 1) if full else 0
    print(f"{name:14s} SeAn-CP={restricted.value:.8g} full={full:.8g}"
          f" signed error={error:+.3f}%")
    print(f"  full normal={normal}; three families="
          f"{[(p.family, round(p.best_value, 8)) for p in restricted.paths]}")
