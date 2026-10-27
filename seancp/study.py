"""Single-frequency harmonic pressure-force-torque study of SeAn-CP FS_ext.

The full-plane numerical reference and three-path SeAn-CP evaluation are
computed independently, using the same strain-amplitude definition per case.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
import csv
import json
import hashlib
import math

import numpy as np
from scipy.optimize import differential_evolution

from .core import evaluate_fs_ext, full_plane_scan, seancp_fs_ext
from .thick_tube import (TubeGeometry, ElasticMaterial, HarmonicLoad, TubeLoads,
                         harmonic_history, stress_tensor, von_mises)

FAMILIES = ("proportional", "torque_90", "pressure_90", "mixed_90",
            "general", "mean_shifted")
ALGORITHMS = ("longest_chord", "mcc", "mce")


@dataclass(frozen=True)
class StudyConfig:
    geometry: TubeGeometry = TubeGeometry(12.0, 1.5)
    material: ElasticMaterial = ElasticMaterial(210000.0, .30)
    sigma_y_mpa: float = 900.0  # Illustrative, not a measured material property.
    k_fs: float = .4
    end_condition: str = "closed"
    max_pressure_mpa: float = 40.0
    max_axial_amplitude_n: float = 18000.0
    max_torque_amplitude_nmm: float = 150000.0
    screening_fraction: float = .9
    min_reference_fs: float = 1e-7
    seed: int = 23109

    def __post_init__(self):
        if not (np.isfinite(self.sigma_y_mpa) and self.sigma_y_mpa > 0 and
                np.isfinite(self.k_fs) and self.k_fs >= 0 and
                0 < self.screening_fraction <= 1 and
                self.end_condition in ("closed", "pressure_compensated") and
                self.max_pressure_mpa > 0 and self.max_axial_amplitude_n > 0 and
                self.max_torque_amplitude_nmm > 0 and isinstance(self.seed, int) and self.seed >= 0):
            raise ValueError("Invalid study configuration")


@dataclass(frozen=True)
class LoadCase:
    case_id: str
    family: str
    loads: TubeLoads


def harmonic(mean: float, amplitude: float, phase: float = 0.) -> HarmonicLoad:
    return HarmonicLoad.from_amplitude_phase(float(mean), float(amplitude), float(phase))


def fixed_cases() -> list[LoadCase]:
    """Ten canonical controls including proportional and out-of-phase loading."""
    # id, pressure mean/amplitude/phase, force amplitude/mean/phase,
    # torque phase/amplitude. All phase arguments are in radians.
    vals = [
        ("axial_only", 0, 0, 0, 12000, 0, 0, 0, 0),
        ("torsion_only", 0, 0, 0, 0, 0, 0, 0, 110000),
        ("pressure_only", 20, 10, 0, 0, 0, 0, 0, 0),
        ("axial_torsion_proportional", 0, 0, 0, 12000, 0, 0, 0, 110000),
        ("pressure_axial_torque_proportional", 20, 10, 0, 12000, 0, 0, 0, 110000),
        ("pressure_torsion_90", 20, 10, 0, 12000, 0, 0, np.pi/2, 110000),
        ("pressure_90", 20, 10, np.pi/2, 12000, 0, 0, 0, 110000),
        ("two_phase_90", 20, 10, np.pi/2, 12000, 0, 0, -np.pi/2, 110000),
        ("phase_general", 20, 10, .9, 12000, 0, 0, 2.1, 110000),
        ("mean_tension_torsion", 20, 8, 1.4, 9000, 11000, 0, 1.8, 90000),
    ]
    return [LoadCase("control_"+name,"control",TubeLoads(
        p_i=harmonic(pm,pa,pp),force=harmonic(fm,fa,pf),
        torque=harmonic(0,ta,pt)))
        for name,pm,pa,pp,fa,fm,pf,pt,ta in vals]


def random_cases(config: StudyConfig, n_per_family: int) -> list[LoadCase]:
    """Deterministic stratified sampling; pressure remains nonnegative."""
    if not isinstance(n_per_family,int) or n_per_family < 0:
        raise ValueError("n_per_family must be nonnegative")
    rng=np.random.default_rng(config.seed)
    result=[]
    for family in FAMILIES:
        for i in range(n_per_family):
            pm=rng.uniform(.12,.88)*config.max_pressure_mpa
            pa=rng.uniform(.04,1)*min(pm, config.max_pressure_mpa-pm)
            fa=rng.uniform(.15,1)*config.max_axial_amplitude_n
            ta=rng.uniform(.15,1)*config.max_torque_amplitude_nmm
            fm=rng.uniform(-.65,.65)*config.max_axial_amplitude_n
            tm=rng.uniform(-.25,.25)*config.max_torque_amplitude_nmm
            if family=="proportional": pp,tp=0.,0.
            elif family=="torque_90": pp,tp=0.,rng.choice([-1,1])*np.pi/2
            elif family=="pressure_90": pp,tp=rng.choice([-1,1])*np.pi/2,0.
            elif family=="mixed_90": pp,tp=np.pi/2,-np.pi/2
            else: pp,tp=rng.uniform(-np.pi,np.pi,2)
            if family=="mean_shifted":
                fm=rng.uniform(-1.1,1.1)*config.max_axial_amplitude_n
                tm=rng.uniform(-.7,.7)*config.max_torque_amplitude_nmm
            loads=TubeLoads(p_i=harmonic(pm,pa,pp),force=harmonic(fm,fa),
                            torque=harmonic(tm,ta,tp))
            result.append(LoadCase(f"{family}_{i:04d}",family,loads))
    return result


def peak_vm_sampled(case: LoadCase, config: StudyConfig, n_phase: int=48) -> float:
    """Sampled through-cycle VM screen, not a certified continuous bound."""
    peak=0.
    for phase in np.linspace(0,2*np.pi,n_phase,endpoint=False):
        loads=case.loads.at(float(phase))
        for r in ("inner","outer"):
            S=stress_tensor(config.geometry,r,end_condition=config.end_condition,**loads)
            peak=max(peak,float(von_mises(S)))
    return peak


def load_fields(case: LoadCase) -> dict:
    result={"case_id":case.case_id,"family":case.family}
    for name,unit in (("p_i","mpa"),("p_o","mpa"),("force","n"),("torque","nmm")):
        l=getattr(case.loads,name)
        for part in ("mean","sine","cosine"):
            result[f"{name}_{part}_{unit}"]=float(getattr(l,part))
        result[f"{name}_amplitude_{unit}"]=l.amplitude
        result[f"{name}_minimum_{unit}"]=l.extrema[0]
        result[f"{name}_maximum_{unit}"]=l.extrema[1]
        result[f"{name}_phase_rad"]=math.atan2(l.cosine,l.sine)
    return result


def evaluate_case(case: LoadCase, config: StudyConfig, algorithm: str,
                  radii: Sequence[float], quality: str="screen") -> tuple[dict,list[dict]]:
    """Use wall-wide maxima to avoid misleading best-local-plane comparisons."""
    if algorithm not in ALGORITHMS: raise ValueError("Unsupported amplitude")
    if quality not in ("screen","fine"): raise ValueError("Unknown resolution")
    nr,nt,np_ = (61,31,61) if quality=="screen" else (361,121,241)
    points=[]
    for radius in radii:
        h=harmonic_history(config.geometry,config.material,case.loads,float(radius),
                           end_condition=config.end_condition)
        restricted=seancp_fs_ext(h,algorithm=algorithm,k_fs=config.k_fs,
                                sigma_y=config.sigma_y_mpa,samples_per_path=nr)
        reference,nref=full_plane_scan(h,algorithm=algorithm,k_fs=config.k_fs,
                                       sigma_y=config.sigma_y_mpa,n_theta=nt,n_phi=np_)
        if restricted.value > reference*(1+2e-6)+1e-12:
            raise RuntimeError(f"Full-plane reference not sufficiently converged {case.case_id} r={radius}")
        points.append({"case_id":case.case_id,"family":case.family,"algorithm":algorithm,
                       "quality":quality,"radius_mm":float(radius),
                       "fs_ext_reference":reference,"fs_ext_seancp":restricted.value,
                       "error_local_percent":100*(restricted.value/reference-1)
                           if reference>config.min_reference_fs else float("nan")})
    jref=int(np.argmax([p["fs_ext_reference"] for p in points]))
    jsemi=int(np.argmax([p["fs_ext_seancp"] for p in points]))
    ref=points[jref]["fs_ext_reference"]
    semi=points[jsemi]["fs_ext_seancp"]
    row={**load_fields(case),"algorithm":algorithm,"quality":quality,
         "fs_ext_reference":ref,"fs_ext_seancp":semi,
         "error_wall_percent":100*(semi/ref-1) if ref>config.min_reference_fs else float("nan"),
         "governing_ref_radius_mm":points[jref]["radius_mm"],
         "governing_seancp_radius_mm":points[jsemi]["radius_mm"],"n_radii":len(points)}
    return row,points


def rank_distinct(rows: list[dict], n: int) -> list[dict]:
    ordered=sorted((x for x in rows if np.isfinite(x["error_wall_percent"])),
                   key=lambda x:(x["error_wall_percent"],x["case_id"],x["algorithm"]))
    out=[];seen=set()
    for row in ordered:
        if row["case_id"] not in seen:
            seen.add(row["case_id"]);out.append(row)
        if len(out)>=n: break
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore")
        w.writeheader();w.writerows(rows)


def independent_global_check(history, config: StudyConfig, algorithm: str,seed:int=19):
    """Independent differential evolution on hemisphere; not a global proof."""
    def f(x):
        theta,phi=x
        normal=np.array([[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]])
        return -float(evaluate_fs_ext(history,normal,algorithm=algorithm,k_fs=config.k_fs,
                                      sigma_y=config.sigma_y_mpa)[0])
    result=differential_evolution(f,((0,np.pi/2),(0,2*np.pi)),seed=seed,popsize=12,
                                  maxiter=100,tol=1e-10,polish=True)
    theta,phi=result.x
    return -float(result.fun),np.array([np.sin(theta)*np.cos(phi),
                                        np.sin(theta)*np.sin(phi),np.cos(theta)])


def run_study(outdir: str | Path,*,n_per_family:int=18, config:StudyConfig=StudyConfig(),
              top_n:int=4,plot:bool=True,resume:bool=False) -> dict:
    if top_n<1 or not isinstance(top_n,int): raise ValueError("top_n must be positive")
    output=Path(outdir);output.mkdir(parents=True,exist_ok=True)
    cases=fixed_cases()+random_cases(config,n_per_family)
    sig_args={"config":asdict(config),"n_per_family":n_per_family,
              "version":"harmonic-fs-ext-phase3-v1","screen_grid":[31,61]}
    signature=hashlib.sha256(json.dumps(sig_args,sort_keys=True).encode()).hexdigest()
    sigfile=output/"screening_signature.json"
    accepted=[];rejected=[];peaks={}
    for case in cases:
        pk=peak_vm_sampled(case,config)
        peaks[case.case_id]=pk
        (accepted if pk <= config.sigma_y_mpa*config.screening_fraction else rejected).append(case)
    screen_radii=[config.geometry.r_i,(config.geometry.r_i+config.geometry.r_o)/2,config.geometry.r_o]
    cached=False
    if resume and sigfile.is_file() and (output/"screening_cases.csv").is_file():
        sig=json.loads(sigfile.read_text())
        if sig.get("sha256")!=signature:
            raise ValueError("Cached screening configuration differs")
        with (output/"screening_cases.csv").open(newline="",encoding="utf-8") as f:
            screen_rows=list(csv.DictReader(f))
        for r in screen_rows:
            for k in ("fs_ext_reference","fs_ext_seancp","error_wall_percent",
                      "governing_ref_radius_mm","governing_seancp_radius_mm"):
                r[k]=float(r[k])
        if len(screen_rows)!=3*len(accepted):
            raise ValueError("Cached screening rows do not match accepted cases")
        cached=True
    else:
        screen_rows=[];screen_points=[]
        for case in accepted:
            for algorithm in ("longest_chord","mce"):
                row,pts=evaluate_case(case,config,algorithm,screen_radii,"screen")
                row["von_mises_sampled_max_mpa"]=peaks[case.case_id]
                screen_rows.append(row);screen_points+=pts
                if algorithm=="longest_chord":
                    screen_rows.append({**row,"algorithm":"mcc"})
                    screen_points +=[{**p,"algorithm":"mcc"} for p in pts]
        write_csv(output/"screening_cases.csv",screen_rows)
        write_csv(output/"screening_radial.csv",screen_points)
        sigfile.write_text(json.dumps({**sig_args,"sha256":signature},indent=2),encoding="utf-8")
    candidates={c.case_id:c for c in accepted}
    initial=rank_distinct(screen_rows,top_n)
    shortlist=rank_distinct(screen_rows,min(len(accepted),max(3*top_n,top_n)))
    fine_radii=np.linspace(config.geometry.r_i,config.geometry.r_o,11)
    refined=[];radial=[]
    for r in shortlist:
        case=candidates[r["case_id"]]
        for algorithm in ("longest_chord","mce"):
            row,points=evaluate_case(case,config,algorithm,fine_radii,"fine")
            row["von_mises_sampled_max_mpa"]=peaks[case.case_id]
            refined.append(row);radial+=points
            if algorithm=="longest_chord":
                refined.append({**row,"algorithm":"mcc"})
                radial +=[{**p,"algorithm":"mcc"} for p in points]
    selected=rank_distinct(refined,top_n)
    write_csv(output/"refined_candidates.csv",refined)
    write_csv(output/"refined_radial.csv",radial)
    write_csv(output/"worst_four.csv",selected)
    checks=[]
    for row in selected:
        case=candidates[row["case_id"]];alg=row["algorithm"]
        radius=row["governing_ref_radius_mm"]
        h=harmonic_history(config.geometry,config.material,case.loads,radius,
                           end_condition=config.end_condition)
        kw={"k_fs":config.k_fs,"sigma_y":config.sigma_y_mpa,"algorithm":alg}
        coarse,_=full_plane_scan(h,**kw,n_theta=41,n_phi=81)
        fine,_=full_plane_scan(h,**kw,n_theta=181,n_phi=361)
        de,n_de=independent_global_check(h,config,alg)
        checks.append({"case_id":row["case_id"],"algorithm":alg,"radius_mm":radius,
                       "coarse_fs_ext":coarse,"dense_fs_ext":fine,"independent_de_fs_ext":de,
                       "relative_change_percent":100*(fine/coarse-1) if coarse else float("nan"),
                       "independent_difference_percent":100*(de/fine-1) if fine else float("nan")})
    write_csv(output/"convergence_checks.csv",checks)
    manifest={"description":"Single harmonic FS_ext SeAn-CP load-space search",
              "input":{**sig_args,"top_n":top_n},
              "screening":{"candidates":len(cases),"accepted":len(accepted),
                           "rejected_vm_sampled":len(rejected),"cache_reused":cached,
                           "screen_radii_mm":screen_radii,
                           "fine_radii_mm":list(map(float,fine_radii)),
                           "fine_grid":[121,241]},
              "initial_worst_ids":[r["case_id"] for r in initial],
              "final_worst_ids":[r["case_id"] for r in selected],
              "cautions":["Illustrative yield strength; no physical machine feasibility asserted",
                          "Full-plane comparator is numerical, not a certified global bound",
                          "MCC and longest chord coincide for single harmonic ellipses"]}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if plot:
        from .study_plots import plot_damage_landscape,plot_screening_summary,plot_radial_comparison
        plot_screening_summary(screen_rows,output/"error_distribution.png")
        for s in selected:
            case=candidates[s["case_id"]]
            for alg in ("longest_chord","mce"):
                row=next(x for x in refined if x["case_id"]==case.case_id and x["algorithm"]==alg)
                h=harmonic_history(config.geometry,config.material,case.loads,
                                   row["governing_ref_radius_mm"],end_condition=config.end_condition)
                for three_d in (False,True):
                    plot_damage_landscape(h,config,
                                          output/f"{case.case_id}_{alg}_{'surface3d' if three_d else 'heatmap'}.png",
                                          title=f"{case.case_id} | {alg} | r={row['governing_ref_radius_mm']:.2f} mm",
                                          algorithm=alg,three_d=three_d)
            pts=[r for r in radial if r["case_id"]==case.case_id and r["algorithm"] in ("longest_chord","mce")]
            plot_radial_comparison(pts,output/f"{case.case_id}_through_wall.png",title=case.case_id)
    return {"worst":selected,"screening":screen_rows,"manifest":manifest}
