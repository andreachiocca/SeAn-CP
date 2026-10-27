"""Run SeAn-CP versus numerical full-plane scanning on a harmonic history."""

import numpy as np
from seancp import HarmonicHistory, full_plane_scan, seancp_fs_ext

# Stress [MPa], tensorial small strain, not engineering shear strain.
Smean = np.diag([5., 35., 18.])
Ssin = np.diag([0., 100., -30.])
Scos = np.array([[0., 0., 0.], [0., 0., 45.], [0., 45., 0.]])
Emean = np.zeros((3, 3))
Esin = np.diag([-.0001, .0005, -.0002])
Ecos = Scos * 1.3e-5
history = HarmonicHistory(Smean, Ssin, Scos, Emean, Esin, Ecos)

for name in ("longest_chord", "mcc", "mce"):
    restricted = seancp_fs_ext(history, algorithm=name)
    full, normal = full_plane_scan(history, algorithm=name)
    error = 100*(restricted.value/full - 1) if full else 0
    print(f"{name:14s} SeAn-CP={restricted.value:.8g} full={full:.8g}"
          f" signed error={error:+.3f}%")
    print(f"  full normal={normal}; three families="
          f"{[(p.family, round(p.best_value, 8)) for p in restricted.paths]}")
