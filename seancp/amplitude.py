"""Engineering shear-strain amplitudes on a plane.

For a complete harmonic ellipse, longest_chord and MCC are both the major
semiaxis. MCE is the Euclidean norm of the two ellipse semiaxes, consistent
with multifat's harmonic-shear definition. For general sampled paths,
longest_chord is half the path diameter, while MCC is the minimum enclosing
circle radius. MCE is unavailable for general sampled paths.
"""

from __future__ import annotations

import numpy as np

ALGORITHMS = ("longest_chord", "mcc", "mce")


def harmonic_amplitude(sine: np.ndarray, cosine: np.ndarray, algorithm: str) -> np.ndarray:
    """Return engineering shear amplitudes for coefficient arrays (...,2|3)."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown amplitude algorithm {algorithm!r}; choose {ALGORITHMS}")
    a, b = np.asarray(sine, dtype=float), np.asarray(cosine, dtype=float)
    if a.shape != b.shape or a.ndim == 0 or a.shape[-1] not in (2, 3):
        raise ValueError("Sine and cosine must have matching shapes (..., 2|3)")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Nonfinite shear coefficients")
    scale = np.maximum(np.max(np.abs(a), axis=-1), np.max(np.abs(b), axis=-1))
    factor = np.where(scale == 0, 1.0, scale)
    a, b = a / factor[..., None], b / factor[..., None]
    aa, bb, ab = np.sum(a*a, axis=-1), np.sum(b*b, axis=-1), np.sum(a*b, axis=-1)
    if algorithm == "mce":
        radius = np.sqrt(aa + bb)
    else:
        radius = np.sqrt(0.5 * (aa + bb + np.hypot(aa - bb, 2 * ab)))
    with np.errstate(over="raise", invalid="raise"):
        return scale * radius


def _smallest_enclosing_circle(points: np.ndarray) -> float:
    """Deterministic incremental minimum enclosing disk for 2D points.

    Intended for tens/hundreds of time points, not raw high-frequency sensor
    records. A fixed permutation avoids pathological chronological ordering.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0:
        raise ValueError("Expected nonempty (T, 2) path")
    p = p[np.random.default_rng(103).permutation(len(p))]
    center, radius = p[0].copy(), 0.0

    def outside(q, c, r):
        return np.linalg.norm(q - c) > r + 2e-12 * max(1.0, r)

    for i in range(1, len(p)):
        if not outside(p[i], center, radius):
            continue
        center, radius = p[i].copy(), 0.0
        for j in range(i):
            if not outside(p[j], center, radius):
                continue
            center = 0.5 * (p[i] + p[j])
            radius = np.linalg.norm(p[i] - center)
            for k in range(j):
                if not outside(p[k], center, radius):
                    continue
                a, b, c = p[i], p[j], p[k]
                u, v = b - a, c - a
                determinant = 2.0 * (u[0]*v[1] - u[1]*v[0])
                if abs(determinant) < 1e-14 * max(1.0, np.linalg.norm(u)*np.linalg.norm(v)):
                    pairs = ((a, b), (a, c), (b, c))
                    x, y = max(pairs, key=lambda pair: np.linalg.norm(pair[0] - pair[1]))
                    center = (x + y) / 2.0
                    radius = np.linalg.norm(x - center)
                else:
                    u2, v2 = u @ u, v @ v
                    center = a + np.array([(u2*v[1] - v2*u[1]) / determinant,
                                           (v2*u[0] - u2*v[0]) / determinant])
                    radius = np.linalg.norm(a - center)
    return float(radius)


def sampled_amplitudes(shear_paths: np.ndarray, algorithm: str) -> np.ndarray:
    """Return sampled path amplitudes from engineering shear vectors (T,P,3).

    MCE requires a complete harmonic representation and cannot be assigned to
    arbitrary sampled paths without introducing another constitutive rule.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown amplitude algorithm {algorithm!r}")
    if algorithm == "mce":
        raise ValueError("MCE requires a complete single-harmonic history")
    q = np.asarray(shear_paths, dtype=float)
    if q.ndim != 3 or q.shape[2] != 3 or q.shape[0] < 2 or not np.isfinite(q).all():
        raise ValueError("Shear paths must have finite shape (T>=2, P, 3)")
    if algorithm == "longest_chord":
        diam2 = np.zeros(q.shape[1])
        for i in range(q.shape[0]-1):
            differences = q[i+1:] - q[i]
            diam2 = np.maximum(diam2, np.max(np.sum(differences*differences, axis=-1), axis=0))
        return 0.5 * np.sqrt(diam2)
    result = np.empty(q.shape[1])
    for i in range(q.shape[1]):
        centered = q[:, i, :] - q[0, i, :]
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        result[i] = _smallest_enclosing_circle(q[:, i, :] @ vt[:2].T)
    return result
