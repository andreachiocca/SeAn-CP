"""FE influence-superposition and constrained harmonic FS_ext search for a tube with one side notch.

No stress-concentration factors or notch FE results are fabricated. The basis
input MUST come from a converged three-load MAPDL run of notched_tube.mac with
BASIS_MODE=1. Results are valid only for the actual mesh/boundary conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import csv
import numpy as np

from .core import HarmonicHistory, seancp_fs_ext, full_plane_scan

# Rows: harmonic mean/sine/cosine; columns: internal pressure [MPa],
# additional axial force [N], torque [N mm]. Basis file uses these magnitudes.
UNIT_CONTROLS = np.array([1.0, 1000.0, 1000.0])


@dataclass(frozen=True)
class GaugeElement:
    element_id: int
    stress_response: np.ndarray  # (3 controls, 3, 3), MPa for UNIT_CONTROLS
    strain_response: np.ndarray  # tensorial (3 controls, 3, 3)
    coordinates: np.ndarray | None = None


def tensor_from_six(x: np.ndarray, *, engineering: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 6 or not np.isfinite(x).all():
        raise ValueError("Expected finite (...,6) components ordered xx yy zz xy yz xz")
    out = np.zeros(x.shape[:-1] + (3, 3))
    for idx, (a, b) in enumerate(((0,0),(1,1),(2,2),(0,1),(1,2),(0,2))):
        value = x[...,idx] * (0.5 if engineering and a != b else 1.0)
        out[...,a,b] = value
        out[...,b,a] = value
    return out


def _rows(path: Path, ncols: int) -> np.ndarray:
    from numpy import array
    output=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith(("#","!")):
            continue
        fields=[z.strip() for z in line.split(",") if z.strip()]
        if len(fields) != ncols:
            raise ValueError(f"Expected {ncols} comma-separated numeric columns in {path}")
        vals=[float(z.replace('D','E').replace('d','e')) for z in fields]
        if not np.isfinite(vals).all():
            raise ValueError("Nonfinite FE values")
        output.append(vals)
    if not output:
        raise ValueError(f"No FE basis rows in {path}")
    return array(output, dtype=float)


def read_influence_basis(folder: str | Path) -> dict[int, GaugeElement]:
    """Read matching 3 basis coefficients for all gauge elements.

    The three MAPDL solves are independent UNIT LOADS, not time instants.
    """
    folder=Path(folder)
    s=_rows(folder/'fe_stress.csv',8)
    e=_rows(folder/'fe_strain.csv',8)
    if s.shape!=e.shape or not np.array_equal(s[:,:2],e[:,:2]):
        raise ValueError("Stress/strain indices must match exactly")
    # Enforce integer basis and element IDs; each element must have all three.
    if not np.allclose(s[:,:2],np.rint(s[:,:2])):
        raise ValueError("FE basis and element IDs must be integers")
    keys=[tuple(map(int,row)) for row in s[:,:2]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate FE basis record")
    coord={}
    cp=folder/'fe_coordinates.csv'
    if cp.exists():
        x=_rows(cp,5)
        if not np.array_equal(x[:,:2],s[:,:2]):
            raise ValueError("Coordinates do not match FE tensor indices")
        for key,pt in zip(keys,x[:,2:]):coord[key]=pt
    out={}
    for eid in sorted(set(int(row[1]) for row in s)):
        mask=s[:,1]==eid
        order=np.argsort(s[mask,0])
        rowidx=s[mask,:2][order].astype(int)
        if rowidx[:,0].tolist() != [1,2,3]:
            raise ValueError(f"Missing unit loading basis for element {eid}")
        ss=tensor_from_six(s[mask,2:][order])
        ee=tensor_from_six(e[mask,2:][order],engineering=True)
        out[eid]=GaugeElement(eid,ss,ee,coord.get((1,eid)))
    return out


def validate_controls(controls: np.ndarray, *, pmax: float | None = None,
                      force_limit: float | None = None, torque_limit: float | None = None):
    """Validate physically nonnegative internal pressure and optional limits.

    Valid pressure waveform is p_m+ps*sin(t)+pc*cos(t), with pmin>=0.
    """
    c=np.asarray(controls,float)
    if c.shape!=(3,3) or not np.isfinite(c).all():
        raise ValueError("Controls must have finite (3,3) shape")
    amplitude=np.hypot(c[1],c[2])
    if c[0,0]+1e-10 < amplitude[0]:
        raise ValueError("Internal pressure must remain nonnegative over complete cycle")
    for j,upper in enumerate((pmax,force_limit,torque_limit)):
        if upper is not None and (not np.isfinite(upper) or upper<=0):
            raise ValueError("Control limits must be positive")
        if upper is not None and abs(c[0,j])+amplitude[j]>upper*(1+1e-10):
            raise ValueError(f"Harmonic control {j} exceeds supplied amplitude limit")
    return c


def history_from_response(element: GaugeElement, controls: np.ndarray) -> HarmonicHistory:
    c=validate_controls(controls) / UNIT_CONTROLS[None,:]
    stress=np.einsum('aj,jkl->akl',c,element.stress_response)
    strain=np.einsum('aj,jkl->akl',c,element.strain_response)
    return HarmonicHistory(stress[0],stress[1],stress[2],strain[0],strain[1],strain[2])


def generate_controls(rng: np.random.Generator, pmax: float, fmax: float,
                      tmax: float) -> np.ndarray:
    """Draw bounded mean/phase/amplitude without negative internal pressure."""
    if min(pmax,fmax,tmax)<=0 or not np.isfinite([pmax,fmax,tmax]).all():
        raise ValueError("Finite positive illustrative machine ranges required")
    c=np.zeros((3,3))
    pmean=rng.uniform(.25*pmax,.75*pmax)
    pamp=rng.uniform(.10,.92)*min(pmean,pmax-pmean)
    c[0,0]=pmean
    phase=rng.uniform(-np.pi,np.pi)
    c[1,0]=pamp*np.cos(phase)
    c[2,0]=pamp*np.sin(phase)
    for k,limit in ((1,fmax),(2,tmax)):
        mean=rng.uniform(-.15,.15)*limit
        amplitude=rng.uniform(.25,.95)*(limit-abs(mean))
        phase=rng.uniform(-np.pi,np.pi)
        c[:,k]=[mean,amplitude*np.cos(phase),amplitude*np.sin(phase)]
    return validate_controls(c,pmax=pmax,force_limit=fmax,torque_limit=tmax)


def max_von_mises(history: HarmonicHistory, phase_count: int = 32) -> float:
    s=history.S_mean[None]+np.sin(np.linspace(0,2*np.pi,phase_count,endpoint=False))[:,None,None]*history.S_sin + np.cos(np.linspace(0,2*np.pi,phase_count,endpoint=False))[:,None,None]*history.S_cos
    dev=s-np.trace(s,axis1=-2,axis2=-1)[:,None,None]*np.eye(3)/3
    return float(np.max(np.sqrt(1.5*np.einsum('tij,tij->t',dev,dev))))


def region_comparison(elements: Mapping[int,GaugeElement], controls: np.ndarray,
                      *, algorithm: str='mce', coarse: bool=True,
                      k_fs: float=0.4, sigma_y: float=600.0):
    """Compare maxima over the SAME selected notch-region elements.

    This is a per-subset FE comparison, not a whole-specimen maximum proof.
    """
    if not elements: raise ValueError("Empty gauge-element selection")
    best_ref=(-1.0,None,None)
    best_semi=(-1.0,None,None)
    for eid,element in elements.items():
        h=history_from_response(element,controls)
        semi=seancp_fs_ext(h,algorithm=algorithm,k_fs=k_fs,sigma_y=sigma_y,
                           samples_per_path=37 if coarse else 181,refine=not coarse)
        ref,n=full_plane_scan(h,algorithm=algorithm,k_fs=k_fs,sigma_y=sigma_y,
                              n_theta=31 if coarse else 91,
                              n_phi=61 if coarse else 181,refine=not coarse)
        if ref>best_ref[0]:best_ref=(float(ref),eid,n)
        if semi.value>best_semi[0]:best_semi=(float(semi.value),eid,semi.normal)
    reference,refid,normal=best_ref
    estimate,semiid,semiplane=best_semi
    return dict(reference_fs_ext=reference,seancp_fs_ext=estimate,
                error_percent=float(100*(estimate/reference-1)) if reference>1e-14 else 0.0,
                reference_element=refid,seancp_element=semiid,
                reference_normal=normal.tolist(),seancp_normal=semiplane.tolist())


def screen_notch_controls(elements: Mapping[int,GaugeElement], *, seed: int = 31027,
                          trials:int=40, pmax:float=40.0, fmax:float=10000.0,
                          tmax:float=40000.0, algorithm:str='mce',
                          max_vm:float|None=600.0,
                          screen_elements:int=12, shortlisted:int=4,
                          k_fs:float=0.4, sigma_y:float=600.0):
    """Random screening and independent refined verification (once FE exists).

    Machine limits here are *user-configurable demonstration ranges*, not rig
    specs. Only tested local stress levels are included in the yield screen.
    """
    if trials<1 or shortlisted<1:raise ValueError("Require positive trials/shortlist")
    if algorithm not in ('longest_chord','mcc','mce'):raise ValueError("Unknown method")
    if not elements:raise ValueError("No extracted FE elements")
    if max_vm is not None and (max_vm<=0 or not np.isfinite(max_vm)):
        raise ValueError("max_vm must be finite positive when specified")
    rng=np.random.default_rng(seed)
    # Screening focuses on largest basis stress magnitudes, not necessarily
    # the true critical point; --screen-elements 0 evaluates ALL extracted.
    candidates=sorted(elements.values(),key=lambda e:np.linalg.norm(e.stress_response[0])*pmax+
                      np.linalg.norm(e.stress_response[1])*fmax/1000+
                      np.linalg.norm(e.stress_response[2])*tmax/1000,reverse=True)
    if screen_elements>0:candidates=candidates[:screen_elements]
    chosen={el.element_id:el for el in candidates}
    rows=[]
    for i in range(trials):
        c=generate_controls(rng,pmax,fmax,tmax)
        if max_vm is not None:
            peak=max(max_von_mises(history_from_response(e,c)) for e in candidates)
            if peak>max_vm:continue
        result=region_comparison(chosen,c,algorithm=algorithm,coarse=True,
                                 k_fs=k_fs,sigma_y=sigma_y)
        rows.append(dict(trial=i,controls=c,screen=result))
    rows.sort(key=lambda x:x['screen']['error_percent'])
    verified=[]
    for x in rows[:shortlisted]:
        check=region_comparison(chosen,x['controls'],algorithm=algorithm,coarse=False,
                                k_fs=k_fs,sigma_y=sigma_y)
        verified.append(dict(trial=x['trial'],controls=x['controls'],screen=x['screen'],refined=check))
    verified.sort(key=lambda x:x['refined']['error_percent'])
    return dict(tested=len(rows),requested=trials,screened_element_ids=list(chosen),
                method=algorithm,limits={'pmax_mpa':pmax,'force_abs_max_n':fmax,
                                         'torque_abs_max_nmm':tmax,'vm_screen_max_mpa':max_vm},
                shortlist=verified)


def write_controls_macro(path: str|Path, controls: np.ndarray) -> None:
    c=validate_controls(controls)
    lines=["! Generated physical harmonic pressure/force/torque controls.",
           "! CTRL(1/2/3,1/2/3) are mean/sine/cosine x p[MPa]/F[N]/T[Nmm]."]
    for i in range(3):
        for j in range(3):lines.append(f"CTRL({i+1},{j+1})={c[i,j]:.15g}")
    Path(path).write_text('\n'.join(lines)+'\n')


def fit_target_to_notch(element: GaugeElement, target_stress: np.ndarray, *,
                        pmax:float=40.0, fmax:float=10000., tmax:float=40000.,
                        maxiter:int=250):
    """Fit a target harmonic tensor to three realistic controls at one FE point.

    Uses SLSQP with p>=0 and |mean|+harmonic amplitude <= supplied caps.
    The closest 3-control history need not retain the original FS_ext error.
    """
    from scipy.optimize import minimize
    target=np.asarray(target_stress,float)
    if target.shape!=(3,3,3) or not np.isfinite(target).all():
        raise ValueError("Expected finite target mean/sine/cosine (3,3,3) tensors")
    if min(pmax,fmax,tmax)<=0:raise ValueError('Invalid actuator limits')
    mat=element.stress_response.reshape(3,9)
    targetflat=target.reshape(3,9)
    limits=np.array([pmax,fmax,tmax])/UNIT_CONTROLS
    initial=targetflat@np.linalg.pinv(mat)
    # Strictly feasible start: no cyclic loads, positive mean pressure.
    x0=np.zeros((3,3))
    x0[0,0]=pmax/(4*UNIT_CONTROLS[0])
    x0[0,1:]=np.clip(initial[0,1:],-.1*limits[1:],.1*limits[1:])
    def objective(v):
        x=v.reshape(3,3)
        delta=x@mat-targetflat
        return float(np.sum(delta*delta)/max(np.sum(targetflat*targetflat),1e-12))
    def constraints(v):
        x=v.reshape(3,3)
        amp=np.hypot(x[1],x[2])
        return np.array([x[0,0]-amp[0],limits[0]-x[0,0]-amp[0],
                         limits[1]-abs(x[0,1])-amp[1],
                         limits[2]-abs(x[0,2])-amp[2]])
    bounds=[(-limits[j] if j>0 else 0,limits[j]) for i in range(3) for j in range(3)]
    out=minimize(objective,x0.ravel(),method='SLSQP',bounds=bounds,
                 constraints=[{'type':'ineq','fun':constraints}],
                 options={'maxiter':maxiter,'ftol':1e-12})
    x=out.x.reshape(3,3)
    if np.min(constraints(out.x)) < -1e-5:
        raise RuntimeError('No physically valid constrained target fit found')
    controls=x*UNIT_CONTROLS
    validate_controls(controls,pmax=pmax,force_limit=fmax,torque_limit=tmax)
    stress=np.einsum('ij,jkl->ikl',x,element.stress_response)
    d=100*np.linalg.norm(stress-target)/max(1e-15,np.linalg.norm(target))
    return dict(controls=controls,stress_coefficient_distance_percent=float(d),
                converged=bool(out.success),objective=out.fun,iterations=int(out.nit))
