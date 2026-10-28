"""Phase 5: ideal boundary-traction feasibility and reduced experimental load spaces.

The exact boundary traction t = sigma @ n on the faces of a homogeneous
elastic block is *a mathematical elasticity construction*. It does not prove
that a finite fixture reproduces uniform stress, nor that it sustains fatigue.

Reduced models assume a fixed specimen orientation throughout the cycle and
fit the three harmonic tensor coefficients jointly. Their errors are
coefficient-space Euclidean distances, not experimental errors.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from .core import HarmonicHistory, full_plane_scan, seancp_fs_ext
from .thick_tube import TubeGeometry, HarmonicLoad, TubeLoads, ElasticMaterial, harmonic_history


@dataclass(frozen=True)
class FeasibilityConfig:
    E: float = 210000.0
    nu: float = 0.3
    sigma_y: float = 600.0
    k_fs: float = 0.4
    cube_side_mm: float = 5.0
    tube_D_int_mm: float = 12.0
    tube_t_mm: float = 1.5
    seed: int = 51620


def read_selected_cases(path: str | Path) -> list[dict]:
    """Read the exact Phase 4 reference, not regenerated random cases."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("selected_cases"), list):
        raise ValueError("Expected the Phase 4 selected_cases JSON")
    return data["selected_cases"]


def coefficients(case: dict) -> np.ndarray:
    data = np.asarray([case[k] for k in
                       ("stress_mean_mpa", "stress_sin_mpa", "stress_cos_mpa")], dtype=float)
    if data.shape != (3, 3, 3) or not np.isfinite(data).all() or not np.allclose(data, data.transpose(0, 2, 1), atol=1e-10):
        raise ValueError("Expected three finite, symmetric 3x3 stress tensors")
    return data


def harmonic_from_stress(coeff: np.ndarray, config: FeasibilityConfig) -> HarmonicHistory:
    mat = ElasticMaterial(config.E, config.nu)
    return HarmonicHistory(*coeff, *(mat.strain(c) for c in coeff))


def phase_tensor(coeff: np.ndarray, phases: np.ndarray) -> np.ndarray:
    phases = np.asarray(phases, dtype=float)
    return (coeff[0][None] + np.sin(phases)[:, None, None]*coeff[1] +
            np.cos(phases)[:, None, None]*coeff[2])


def max_traction_components(coeff: np.ndarray, side_mm: float) -> dict:
    """Peak ideal tractions (and face-resultant loads) on a cubic specimen."""
    if side_mm <= 0:
        raise ValueError("Side must be positive")
    a = coeff[0];sin=coeff[1];cos=coeff[2]
    comp_high = a + np.hypot(sin, cos)
    comp_low = a - np.hypot(sin, cos)
    area = side_mm**2
    return {
        "normal_peak_tensile_mpa": float(np.max(np.diag(comp_high))),
        "normal_peak_compressive_mpa": float(np.min(np.diag(comp_low))),
        "max_abs_component_mpa": float(np.max(np.maximum(np.abs(comp_high), np.abs(comp_low)))),
        "max_face_component_load_kn": float(np.max(np.maximum(np.abs(comp_high), np.abs(comp_low))) * area / 1000.),
        "max_face_shear_component_load_kn": float(np.max(
            [max(abs(comp_high[i,j]),abs(comp_low[i,j])) for i in range(3) for j in range(i+1,3)]) * area/1000.),
        "normal_axes_change_sign": [bool(lo < -1e-9 and hi > 1e-9) for lo, hi in zip(np.diag(comp_low), np.diag(comp_high))]
    }


def face_traction_program(coeff: np.ndarray, side_mm: float) -> list[dict]:
    """Independent harmonic traction components on positive and negative faces.

    The *ideal* resultant vector on a face is A*sigma(t)*normal. A 5mm cube
    is only a chosen force conversion scale, not a validated specimen design.
    """
    result = []
    area=side_mm**2
    for axis in range(3):
        for sign in (-1, 1):
            normal = np.eye(3)[axis]*sign
            traction = coeff @ normal
            for direction in range(3):
                m,s,c=traction[:,direction]
                result.append(dict(face="-" if sign<0 else "+", face_axis="xyz"[axis],
                    force_axis="xyz"[direction], traction_mean_mpa=float(m),
                    traction_sin_mpa=float(s), traction_cos_mpa=float(c),
                    force_mean_kn=float(m*area/1000), force_sin_kn=float(s*area/1000),
                    force_cos_kn=float(c*area/1000), force_min_kn=float((m-np.hypot(s,c))*area/1000),
                    force_max_kn=float((m+np.hypot(s,c))*area/1000)))
    return result


def tensor_relative_distance(reference: np.ndarray, approx: np.ndarray) -> float:
    return float(100*np.linalg.norm(approx-reference)/np.linalg.norm(reference))


def _normal(theta_phi: np.ndarray) -> np.ndarray:
    theta,phi=theta_phi
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])


def _optimize_normal(coeff: np.ndarray, *, model: str, seed: int) -> np.ndarray:
    norm=np.linalg.norm(coeff)
    def fn(params):
        n=_normal(params)
        if model=="plane_stress":
            P=np.eye(3)-np.outer(n,n)
            projected=P@coeff@P
        elif model=="tube_signed":
            v=coeff@n
            radial_shear=v-(v@n)[:,None]*n[None,:]
            projected=coeff-n[None,:,None]*radial_shear[:,None,:]-radial_shear[:,:,None]*n[None,None,:]
        else:raise ValueError(model)
        return float(np.linalg.norm(projected-coeff)**2/norm**2)
    fits=[differential_evolution(fn, [(0,np.pi),(0,2*np.pi)], seed=q,
                               maxiter=190,popsize=12,tol=1e-11,polish=True)
          for q in (seed,19,101)]
    fit=min(fits,key=lambda z:z.fun)
    return _normal(fit.x)


def projection_plane_stress(coeff: np.ndarray, seed: int=19):
    n=_optimize_normal(coeff,model="plane_stress",seed=seed)
    P=np.eye(3)-np.outer(n,n)
    return P@coeff@P, {"free_surface_normal":n.tolist()}


def projection_tube_signed(coeff: np.ndarray, seed: int=19):
    """Best-fit, fixed-axis tube with TWO algebraically signed pressures.

    At a selected wall location, p_i, p_o, F, M span the 4-dimensional
    subspace of symmetric tensors having zero radial shear components.
    The result is an upper bound on practical tube matching, since negative
    gauge pressures are permitted and actuator capacities are unconstrained.
    """
    n=_optimize_normal(coeff,model="tube_signed",seed=seed)
    v=coeff@n
    shear=v-(v@n)[:,None]*n[None,:]
    projected=coeff-n[None,:,None]*shear[:,None,:]-shear[:,:,None]*n[None,None,:]
    return projected, {"radial_direction":n.tolist(),"signed_gauge_pressure_warning":True}


def projection_triaxial_normal(coeff: np.ndarray, seed: int=19):
    """Find one *fixed* orthonormal basis minimising off-diagonal stress modes."""
    nrm=np.linalg.norm(coeff)
    def fn(rotvec):
        Q=Rotation.from_rotvec(rotvec).as_matrix()
        local=Q.T@coeff@Q
        d=local-np.array([np.diag(np.diag(x)) for x in local])
        return float(np.sum(d*d)/(nrm*nrm))
    fits=[differential_evolution(fn,[(-np.pi,np.pi)]*3,seed=q,maxiter=200,
                               popsize=12,tol=1e-11,polish=True)
          for q in (seed,19,101)]
    fit=min(fits,key=lambda z:z.fun)
    Q=Rotation.from_rotvec(fit.x).as_matrix()
    local=Q.T@coeff@Q
    projected=Q@np.array([np.diag(np.diag(x)) for x in local])@Q.T
    return projected,{"fixed_axes_columns":Q.tolist()}


def project_lorentz(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of (mean,sine,cosine) on mean>=hypot(sine,cosine).

    This is the exact second-order-cone condition for non-negative pressure
    throughout one complete sinusoidal cycle.
    """
    t=float(v[0]); x=np.asarray(v[1:],float); r=float(np.linalg.norm(x))
    if r<=t:return np.asarray(v,float).copy()
    if r<=-t:return np.zeros(3)
    p=0.5*(r+t)
    return np.r_[p,p*x/r]


def projection_tube_positive_internal(coeff: np.ndarray,geometry: TubeGeometry,
                                      seed: int=19):
    """Numerical fit to the *outer wall* of a tube with p_o=0, p_i(t)>=0.

    Other controls F(t) and M(t) are freely signed. The rotation is fixed
    for the whole cycle. The objective is a Frobenius stress-coefficient fit.
    No load capacity, material failure, or through-wall FS check is imposed.
    """
    def construction(rotvec):
        Q=Rotation.from_rotvec(rotvec).as_matrix()
        local=Q.T@coeff@Q
        out=np.zeros_like(local)
        out[:,1,1]=project_lorentz(local[:,1,1])
        out[:,2,2]=local[:,2,2]
        out[:,1,2]=out[:,2,1]=local[:,1,2]
        return Q@out@Q.T,Q,out
    nrm=np.linalg.norm(coeff)
    def fn(v):return float(np.linalg.norm(construction(v)[0]-coeff)**2/nrm**2)
    fits=[differential_evolution(fn,[(-np.pi,np.pi)]*3,seed=q,
                               maxiter=240,popsize=12,tol=1e-10,polish=True)
          for q in (seed,19,101)]
    fit=min(fits,key=lambda z:z.fun)
    projected,Q,local=construction(fit.x)
    a,b=geometry.r_i,geometry.r_o
    factor=2*a*a/(b*b-a*a)
    pressure=local[:,1,1]/factor
    area,J=geometry.area,geometry.J
    # sigma_z = A + F/area with A = p_i*a^2/(b^2-a^2) at closed ends.
    axial_pressure=pressure*a*a/(b*b-a*a)
    force=(local[:,2,2]-axial_pressure)*area
    torque=local[:,1,2]*J/b
    loads=TubeLoads(p_i=HarmonicLoad(*pressure),force=HarmonicLoad(*force),
                    torque=HarmonicLoad(*torque))
    # independently reconstruct and rotate: exact equality is required.
    mat=ElasticMaterial(210000.,.3)
    s=harmonic_history(geometry,mat,loads,position="outer")
    recovered=Q@np.stack([s.S_mean,s.S_sin,s.S_cos])@Q.T
    if not np.allclose(recovered,projected,atol=1e-8,rtol=1e-10):
        raise RuntimeError("Tube inverse reconstruction mismatch")
    return projected,{"fixed_basis_columns":Q.tolist(),
        "pressure_internal_mpa":[float(x) for x in pressure],
        "pressure_range_mpa":[float(x) for x in loads.p_i.extrema],
        "additional_axial_force_n":[float(x) for x in force],
        "additional_force_range_n":[float(x) for x in loads.force.extrema],
        "torque_nmm":[float(x) for x in torque],
        "torque_range_nmm":[float(x) for x in loads.torque.extrema]}


def score_history(coeff: np.ndarray,config: FeasibilityConfig,algorithm="mce",
                  *,high_precision=False) -> dict:
    hist=harmonic_from_stress(coeff,config)
    kw=dict(k_fs=config.k_fs,sigma_y=config.sigma_y,algorithm=algorithm)
    semi=seancp_fs_ext(hist,samples_per_path=361 if high_precision else 91,**kw)
    maximum,normal=full_plane_scan(hist,n_theta=121 if high_precision else 51,
                                   n_phi=241 if high_precision else 101,**kw)
    return {"seancp_fs_ext":float(semi.value),"full_fs_ext":float(maximum),
            "error_percent":float(100*(semi.value/maximum-1)) if maximum>0 else 0.,
            "full_normal":[float(x) for x in normal],
            "seancp_normal":[float(x) for x in semi.normal]}


def independent_maximum(coeff: np.ndarray,config: FeasibilityConfig,algorithm="mce",seed=22):
    """Use a separate global algorithm, not the full_plane_scan optimizer."""
    from scipy.optimize import differential_evolution
    from .core import evaluate_fs_ext
    h=harmonic_from_stress(coeff,config)
    kw=dict(k_fs=config.k_fs,sigma_y=config.sigma_y,algorithm=algorithm)
    def f(x):
        n=_normal(x)
        return -float(evaluate_fs_ext(h,n.reshape(1,3),**kw)[0])
    fit=differential_evolution(f,[(0,np.pi),(0,2*np.pi)],seed=seed,
                                maxiter=150,popsize=12,tol=1e-11,polish=True)
    return -float(fit.fun)


def best_plane_stress_hoop_minimum(coeff: np.ndarray, free_normal: np.ndarray) -> dict:
    """Most favourable minimum hoop stress across fixed tangent directions.

    At a free outer surface of a tube, p_o=0. An exact plane-stress target
    with radial direction free_normal needs hoop stress proportional to p_i.
    If even the best tangent direction has a negative cyclic minimum, an
    internal-only positive-pressure tube cannot reproduce that history.
    """
    n=np.asarray(free_normal,float);n=n/np.linalg.norm(n)
    U,_,_=np.linalg.svd(np.eye(3)-np.outer(n,n))
    u,v=U[:,:2].T
    def minimum(angle):
        m=np.cos(angle)*u+np.sin(angle)*v
        a=np.einsum('i,kij,j->k',m,coeff,m)
        return float(a[0]-np.hypot(a[1],a[2]))
    result=differential_evolution(lambda x:-minimum(x[0]),[(0,np.pi)],
                                  seed=615,tol=1e-13,popsize=10,polish=True)
    return {"most_favourable_hoop_minimum_mpa":float(-result.fun),
            "angle_from_tangent_basis_deg":float(np.rad2deg(result.x[0])),
            "tangent_basis_vectors":[u.tolist(),v.tolist()]}
