"""Corrected three-family semi-analytical Fatemi-Socie extended search.

All three families are derived from the extreme eigenvectors of:
(1) a strain range, (2) stress at the first anchor, and
(3) stress at the opposite anchor.
FS_ext maximises the FULL damage parameter, never a shear-only metric.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .amplitude import ALGORITHMS, harmonic_amplitude, sampled_amplitudes


def _tensor(x: np.ndarray, label: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (3, 3) or not np.isfinite(x).all() or not np.allclose(x, x.T, rtol=0, atol=1e-12):
        raise ValueError(f"{label} must be a finite symmetric (3,3) tensor")
    return x.copy()


@dataclass(frozen=True)
class HarmonicHistory:
    """Stress [MPa] and tensorial small strains, all 3x3.

    S(t) = S_mean + S_sin*sin(t) + S_cos*cos(t), similarly for E(t).
    Sine and cosine are the exact coefficients of one complete cycle.
    """
    S_mean: np.ndarray
    S_sin: np.ndarray
    S_cos: np.ndarray
    E_mean: np.ndarray
    E_sin: np.ndarray
    E_cos: np.ndarray

    def __post_init__(self) -> None:
        for label in ("S_mean", "S_sin", "S_cos", "E_mean", "E_sin", "E_cos"):
            object.__setattr__(self, label, _tensor(getattr(self, label), label))

    @classmethod
    def from_endpoints(cls, S0, S1, E0, E1) -> "HarmonicHistory":
        """Convert two opposed proportional extrema to a harmonic history."""
        s0, s1 = _tensor(S0, "S0"), _tensor(S1, "S1")
        e0, e1 = _tensor(E0, "E0"), _tensor(E1, "E1")
        zero = np.zeros((3, 3))
        return cls((s0+s1)/2, (s0-s1)/2, zero,
                   (e0+e1)/2, (e0-e1)/2, zero)

    def at(self, phase: float) -> tuple[np.ndarray, np.ndarray]:
        return (self.S_mean + self.S_sin*np.sin(phase) + self.S_cos*np.cos(phase),
                self.E_mean + self.E_sin*np.sin(phase) + self.E_cos*np.cos(phase))


@dataclass(frozen=True)
class SampledHistory:
    """Synchronous stress and strain snapshots; each has shape (T>=2,3,3)."""
    stresses: np.ndarray
    strains: np.ndarray

    def __post_init__(self) -> None:
        for label in ("stresses", "strains"):
            value = np.asarray(getattr(self, label), float)
            if (value.ndim != 3 or value.shape[1:] != (3,3) or len(value) < 2
                or not np.isfinite(value).all()
                or not np.allclose(value, value.transpose(0,2,1), atol=1e-12, rtol=0)):
                raise ValueError(f"{label} must contain >=2 finite symmetric (3,3) tensors")
            object.__setattr__(self, label, value.copy())
        if self.stresses.shape != self.strains.shape:
            raise ValueError("Stress and strain histories need identical time sampling")


History = HarmonicHistory | SampledHistory


@dataclass(frozen=True)
class PlanePath:
    family: str
    angles_rad: np.ndarray
    normals: np.ndarray
    values: np.ndarray
    best_normal: np.ndarray
    best_value: float


@dataclass(frozen=True)
class SearchResult:
    value: float
    normal: np.ndarray
    paths: tuple[PlanePath, PlanePath, PlanePath]
    anchor_phase_rad: float | None


def _normals(normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3 or not np.isfinite(n).all():
        raise ValueError("Plane normals must have finite shape (P, 3)")
    r = np.linalg.norm(n, axis=1)
    if np.any(r <= 0):
        raise ValueError("Zero plane normal")
    return n/r[:,None]


def _projection(T: np.ndarray, normals: np.ndarray) -> np.ndarray:
    traction = normals @ T.T
    return traction - np.sum(traction*normals, axis=1)[:,None]*normals


def evaluate_fs_ext(
    history: History,
    normals: np.ndarray,
    *,
    k_fs: float = 0.4,
    sigma_y: float = 300.0,
    algorithm: str = "longest_chord",
) -> np.ndarray:
    """Return gamma_a*(1+k_fs*max(0,sigma_nn,max)/sigma_y) for every normal.

    gamma(t)=2*[E(t)n-(n.E(t)n)n] is an ENGINEERING shear-strain vector.
    Both factors are evaluated on the SAME plane. For harmonic loading
    the normal-stress extrema are evaluated analytically over the cycle.
    """
    if not (np.isfinite(k_fs) and k_fs >= 0 and np.isfinite(sigma_y) and sigma_y > 0):
        raise ValueError("Need finite k_fs>=0 and sigma_y>0")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown amplitude algorithm {algorithm!r}; choose {ALGORITHMS}")
    n = _normals(normals)
    if isinstance(history, HarmonicHistory):
        qsin = 2 * _projection(history.E_sin, n)
        qcos = 2 * _projection(history.E_cos, n)
        amplitude = harmonic_amplitude(qsin, qcos, algorithm)
        mean = np.einsum("pi,ij,pj->p", n, history.S_mean, n)
        sn = np.einsum("pi,ij,pj->p", n, history.S_sin, n)
        cs = np.einsum("pi,ij,pj->p", n, history.S_cos, n)
        normal_max = mean + np.hypot(sn, cs)
    elif isinstance(history, SampledHistory):
        nSn = np.einsum("pi,tij,pj->tp", n, history.stresses, n)
        normal_max = np.max(nSn, axis=0)
        En = np.einsum("tij,pj->tpi", history.strains, n)
        shear_paths = 2 * (En - np.sum(En*n[None,:,:], axis=-1)[...,None]*n[None,:,:])
        amplitude = sampled_amplitudes(shear_paths, algorithm)
    else:
        raise TypeError("History must be HarmonicHistory or SampledHistory")
    return amplitude * (1 + (k_fs/sigma_y)*np.maximum(0.0, normal_max))


def _anchors(history: History, anchor_phase_rad: float | None):
    if isinstance(history, SampledHistory):
        if anchor_phase_rad is not None:
            raise ValueError("Sampled histories do not have a harmonic phase")
        # Choose the sample pair with maximum Frobenius strain-tensor range.
        e = history.strains
        sizes = np.sum((e[:, None] - e[None, :])**2, axis=(2,3))
        a, b = np.unravel_index(np.argmax(sizes), sizes.shape)
        if a == b:
            a, b = 0, len(e)-1
        return (e[a]-e[b], history.stresses[a], history.stresses[b], None)
    if anchor_phase_rad is None:
        # For a harmonic tensor path, the phase of maximal antipodal
        # Frobenius strain range is an eigenvector of the 2x2 Gram matrix.
        a, b = history.E_sin, history.E_cos
        aa, ab, bb = np.sum(a*a), np.sum(a*b), np.sum(b*b)
        if abs(aa-bb) <= 1e-12*max(aa, bb, 1e-30) and abs(ab) <= 1e-12*max(aa, bb, 1e-30):
            phase = 0.0
        else:
            _, v = np.linalg.eigh(np.array([[aa, ab], [ab, bb]]))
            phase = float(np.arctan2(v[0,-1], v[1,-1]))
    else:
        if not np.isfinite(anchor_phase_rad):
            raise ValueError("anchor_phase_rad must be finite")
        phase = float(anchor_phase_rad)
    s0, e0 = history.at(phase)
    s1, e1 = history.at(phase + np.pi)
    return (e0-e1, s0, s1, phase)


def _family_normals(tensor: np.ndarray, angles: np.ndarray) -> np.ndarray:
    _, e = np.linalg.eigh(tensor)  # ascending eigenvalue order
    n = np.cos(angles)[:,None]*e[:,2] + np.sin(angles)[:,None]*e[:,0]
    return _normals(n)


def seancp_fs_ext(
    history: History, *, k_fs: float = 0.4, sigma_y: float = 300.0,
    algorithm: str = "longest_chord", samples_per_path: int = 181,
    anchor_phase_rad: float | None = None, refine: bool = True,
) -> SearchResult:
    """Return FS_ext maximum restricted to the three SeAn-CP plane families.

    Each family is a full Mohr great circle (0<=angle<pi), including the
    conjugate planes omitted/inconsistently handled in parts of MAIN.m.
    The original eigenvector family definitions are preserved.
    """
    if samples_per_path < 8:
        raise ValueError("samples_per_path must be >= 8")
    diff, s0, s1, phase = _anchors(history, anchor_phase_rad)
    families = (("strain_range", diff), ("stress_anchor_0", s0), ("stress_anchor_1", s1))
    angles = np.linspace(0.0, np.pi, samples_per_path, endpoint=False)
    step = np.pi/samples_per_path
    paths = []
    for name, tensor in families:
        n = _family_normals(tensor, angles)
        scores = evaluate_fs_ext(history, n, k_fs=k_fs, sigma_y=sigma_y, algorithm=algorithm)
        idx = int(np.argmax(scores))
        best_value, best_normal = float(scores[idx]), n[idx].copy()
        if refine:
            def objective(a):
                vec = _family_normals(tensor, np.array([a % np.pi]))
                return -float(evaluate_fs_ext(history, vec, k_fs=k_fs,
                                              sigma_y=sigma_y, algorithm=algorithm)[0])
            # Refine each local maximum, including multiple distinct basins.
            peaks = [i for i in range(samples_per_path) if
                     scores[i] >= scores[(i-1)%samples_per_path] and
                     scores[i] >= scores[(i+1)%samples_per_path]]
            for i in peaks:
                fit = minimize_scalar(objective, bounds=(angles[i]-step, angles[i]+step),
                                      method="bounded", options={"xatol":1e-12})
                if -fit.fun > best_value:
                    best_value = float(-fit.fun)
                    best_normal = _family_normals(tensor, np.array([fit.x % np.pi]))[0]
        paths.append(PlanePath(name, angles.copy(), n, scores, best_normal, best_value))
    chosen = max(paths, key=lambda path: path.best_value)
    return SearchResult(chosen.best_value, chosen.best_normal.copy(), tuple(paths), phase)


def full_plane_scan(
    history: History, *, k_fs: float = 0.4, sigma_y: float = 300.0,
    algorithm: str = "longest_chord", n_theta: int = 91,
    n_phi: int = 181, refine: bool = True,
) -> tuple[float, np.ndarray]:
    """Independent full-sphere numerical scanning and local refinement.

    This is a numerical *reference*, not an analytical global-optimum proof.
    """
    if n_theta < 4 or n_phi < 8:
        raise ValueError("Grid dimensions are too small")
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    n = np.stack([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)], axis=-1).reshape(-1,3)
    damage = evaluate_fs_ext(history, n, k_fs=k_fs, sigma_y=sigma_y, algorithm=algorithm)
    idx = int(np.argmax(damage))
    result, normal = float(damage[idx]), n[idx]
    if refine:
        top = np.argsort(damage)[-24:][::-1]
        unique = []
        for j in top:
            if all(abs(np.dot(n[j], q)) < np.cos(np.deg2rad(8)) for q in unique):
                unique.append(n[j])
            if len(unique) == 8:
                break
        for seed in unique:
            def objective(x):
                return -float(evaluate_fs_ext(history, np.asarray(x).reshape(1,3),
                             k_fs=k_fs, sigma_y=sigma_y, algorithm=algorithm)[0])
            sol = minimize(objective, seed, method="BFGS", options={"maxiter":200, "gtol":1e-11})
            if np.linalg.norm(sol.x) > 0 and -sol.fun > result:
                result, normal = float(-sol.fun), sol.x/np.linalg.norm(sol.x)
    return result, normal.copy()
