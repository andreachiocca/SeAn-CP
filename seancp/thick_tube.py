"""Elastic, axisymmetric thick-walled tube under harmonic pressure, force and torque.

All lengths are mm, forces N, moments N mm, stresses and E in MPa.
The orthonormal local basis follows the provided MAIN_v3.py: (r, theta, z).
The resulting complete harmonic stress/strain tensors can be passed directly
into the independent FS_ext scanning and SeAn-CP routines.

Pressure end conditions:
  closed: pressure thrust is carried by the gauge tube: axial pressure stress A.
  pressure_compensated: pressure thrust is entirely reacted outside the gauge
     tube: no axial stress from pressure, despite the Lamé wall stresses.
The prescribed axial force is always the *additional* force applied to the
section, not an uncorrected actuator/load-cell readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, hypot, isfinite, pi, sin
from typing import Literal

import numpy as np

from .core import HarmonicHistory

EndCondition = Literal["closed", "pressure_compensated"]


@dataclass(frozen=True)
class TubeGeometry:
    D_int: float  # mm
    thickness: float  # mm

    def __post_init__(self):
        if not (isfinite(self.D_int) and isfinite(self.thickness)
                and self.D_int > 0 and self.thickness > 0):
            raise ValueError("D_int and thickness must be positive finite lengths [mm]")

    @property
    def r_i(self) -> float:
        return self.D_int / 2.0

    @property
    def r_o(self) -> float:
        return self.r_i + self.thickness

    @property
    def area(self) -> float:
        """Net tube cross-sectional area, mm^2."""
        return pi * (self.r_o**2 - self.r_i**2)

    @property
    def J(self) -> float:
        """Exact polar second moment of area, mm^4."""
        return pi * (self.r_o**4 - self.r_i**4) / 2.0

    def radius(self, position: float | str) -> float:
        if isinstance(position, str):
            if position == "inner":
                return self.r_i
            if position == "outer":
                return self.r_o
            if position == "mean":
                return (self.r_i + self.r_o) / 2.0
            raise ValueError("position must be 'inner', 'outer', 'mean', or a radius [mm]")
        r = float(position)
        if not isfinite(r) or not self.r_i <= r <= self.r_o:
            raise ValueError(f"radius must lie within [{self.r_i}, {self.r_o}] mm")
        return r


@dataclass(frozen=True)
class ElasticMaterial:
    E: float  # MPa
    nu: float

    def __post_init__(self):
        if not isfinite(self.E) or self.E <= 0 or not isfinite(self.nu) or not -1 < self.nu < 0.5:
            raise ValueError("Require E>0 MPa and -1<nu<0.5")

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    def strain(self, stress: np.ndarray) -> np.ndarray:
        """Linear isotropic Hooke law, tensorial strains (including shear)."""
        S = np.asarray(stress, dtype=float)
        if S.shape[-2:] != (3, 3) or not np.isfinite(S).all():
            raise ValueError("stress must contain finite (...,3,3) tensors")
        trace = np.trace(S, axis1=-2, axis2=-1)
        return ((1.0 + self.nu) * S - self.nu * trace[..., None, None] * np.eye(3)) / self.E


@dataclass(frozen=True)
class HarmonicLoad:
    """Scalar L(t)=mean+sine*sin(omega*t)+cosine*cos(omega*t)."""
    mean: float = 0.0
    sine: float = 0.0
    cosine: float = 0.0

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.mean, self.sine, self.cosine)):
            raise ValueError("All harmonic coefficients must be finite")

    @classmethod
    def from_amplitude_phase(cls, mean: float, amplitude: float, phase_rad: float = 0.0):
        """L(t)=mean+amplitude*sin(omega*t+phase_rad)."""
        if not isfinite(amplitude) or amplitude < 0 or not isfinite(phase_rad):
            raise ValueError("Amplitude must be nonnegative; phase must be finite")
        return cls(mean, amplitude * cos(phase_rad), amplitude * sin(phase_rad))

    @property
    def amplitude(self) -> float:
        return hypot(self.sine, self.cosine)

    @property
    def extrema(self) -> tuple[float, float]:
        return self.mean - self.amplitude, self.mean + self.amplitude

    def at(self, phase_rad: float) -> float:
        if not isfinite(phase_rad):
            raise ValueError("phase must be finite")
        return self.mean + self.sine * sin(phase_rad) + self.cosine * cos(phase_rad)


@dataclass(frozen=True)
class TubeLoads:
    """Independent common-frequency load coefficients.

    p_i/p_o: gauge pressure [MPa]; force: additional axial force [N];
    torque: torque [N mm]. A negative pressure is algebraically permitted,
    but physical feasibility must be assessed independently.
    """
    p_i: HarmonicLoad = field(default_factory=HarmonicLoad)
    p_o: HarmonicLoad = field(default_factory=HarmonicLoad)
    force: HarmonicLoad = field(default_factory=HarmonicLoad)
    torque: HarmonicLoad = field(default_factory=HarmonicLoad)

    def __post_init__(self):
        for name in ("p_i", "p_o", "force", "torque"):
            if not isinstance(getattr(self, name), HarmonicLoad):
                raise TypeError(f"{name} must be a HarmonicLoad")

    def at(self, phase_rad: float) -> dict[str, float]:
        return {name: getattr(self, name).at(phase_rad)
                for name in ("p_i", "p_o", "force", "torque")}


def _check_end_condition(end_condition: str) -> None:
    if end_condition not in ("closed", "pressure_compensated"):
        raise ValueError("end_condition must be 'closed' or 'pressure_compensated'")


def stress_tensor(
    geometry: TubeGeometry,
    position: float | str,
    *, p_i: float = 0.0, p_o: float = 0.0, force: float = 0.0,
    torque: float = 0.0, end_condition: EndCondition = "closed",
) -> np.ndarray:
    """Lamé + axial membrane stress + Saint-Venant torsion at radius r.

    For a=r_i, b=r_o, A=(p_i*a^2-p_o*b^2)/(b^2-a^2),
    B=a^2*b^2*(p_i-p_o)/(b^2-a^2), the stresses are
      sigma_rr=A-B/r^2; sigma_tt=A+B/r^2;
      sigma_zz=(A if closed else 0)+force/area;
      tau_tz=torque*r/J.
    No radial shear or bending stress is introduced.
    """
    if not isinstance(geometry, TubeGeometry):
        raise TypeError("geometry must be a TubeGeometry")
    _check_end_condition(end_condition)
    r = geometry.radius(position)
    if not all(isfinite(q) for q in (p_i, p_o, force, torque)):
        raise ValueError("Loads must be finite")
    a, b = geometry.r_i, geometry.r_o
    diff = b*b - a*a
    A = (p_i*a*a - p_o*b*b) / diff
    B = a*a*b*b*(p_i-p_o) / diff
    sigma_r = A - B/r**2
    sigma_theta = A + B/r**2
    sigma_z = (A if end_condition == "closed" else 0.0) + force/geometry.area
    tau = torque*r/geometry.J
    return np.array([[sigma_r, 0., 0.], [0., sigma_theta, tau], [0., tau, sigma_z]])


def harmonic_history(
    geometry: TubeGeometry,
    material: ElasticMaterial,
    loads: TubeLoads,
    position: float | str = "outer",
    *, end_condition: EndCondition = "closed",
) -> HarmonicHistory:
    """Exact single-frequency local stress/strain history for the CP solver.

    The wall solution is linear in each independent load. Build the mean,
    sine and cosine tensors separately, retaining actuator phase offsets.
    """
    if not isinstance(material, ElasticMaterial):
        raise TypeError("material must be an ElasticMaterial")
    if not isinstance(loads, TubeLoads):
        raise TypeError("loads must be a TubeLoads")
    tensors = []
    for attribute in ("mean", "sine", "cosine"):
        T = stress_tensor(
            geometry, position, end_condition=end_condition,
            **{key: getattr(getattr(loads, key), attribute)
               for key in ("p_i", "p_o", "force", "torque")},
        )
        tensors.append(T)
    smean, ssin, scos = tensors
    return HarmonicHistory(
        smean, ssin, scos,
        material.strain(smean), material.strain(ssin), material.strain(scos),
    )


@dataclass(frozen=True)
class RadialProfile:
    """Instantaneous through-wall fields, all in (r,theta,z) basis."""
    radius_mm: np.ndarray
    stress_MPa: np.ndarray
    strain: np.ndarray
    von_mises_MPa: np.ndarray


def von_mises(stress: np.ndarray) -> np.ndarray:
    """J2 equivalent stress for arbitrary symmetric (...,3,3) tensors."""
    S = np.asarray(stress, dtype=float)
    if S.shape[-2:] != (3, 3) or not np.isfinite(S).all():
        raise ValueError("Expected finite (...,3,3) tensors")
    dev = S - np.trace(S, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)/3.0
    return np.sqrt(1.5 * np.sum(dev*dev, axis=(-2, -1)))


def radial_profile(
    geometry: TubeGeometry,
    material: ElasticMaterial,
    loads: TubeLoads,
    *, phase_rad: float = 0.0, n_points: int = 41,
    end_condition: EndCondition = "closed",
) -> RadialProfile:
    """Stress, strain and von Mises fields across the wall at one phase."""
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    instant = loads.at(phase_rad)
    radii = np.linspace(geometry.r_i, geometry.r_o, n_points)
    stresses = np.stack([stress_tensor(geometry, float(r), **instant,
                                       end_condition=end_condition) for r in radii])
    return RadialProfile(radii, stresses, material.strain(stresses), von_mises(stresses))
