"""Assess Phase 4 harmonic stress tensors against reduced test configurations.

Run: python examples/phase5_feasibility.py --out results/phase5
Physical feasibility is NOT certified by local tensor projection or cube
traction equilibrium. Experimental reproduction needs FE and rig validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from seancp.feasibility import (
    FeasibilityConfig, coefficients, face_traction_program,
    harmonic_from_stress, independent_maximum, max_traction_components,
    projection_plane_stress, projection_triaxial_normal, projection_tube_signed,
    projection_tube_positive_internal, read_selected_cases, score_history,
    tensor_relative_distance, best_plane_stress_hoop_minimum,
)
from seancp.thick_tube import TubeGeometry, TubeLoads, HarmonicLoad, ElasticMaterial, harmonic_history, von_mises


def write_csv(path: Path, records):
    if not records:return
    with path.open("w", newline="", encoding="utf8") as h:
        writer=csv.DictWriter(h, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader();writer.writerows(records)


def _flatten_for_output(record):
    return {k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in record.items()}


def calculate(out: Path, source: Path, config: FeasibilityConfig, *, plots: bool=True):
    out.mkdir(parents=True,exist_ok=True)
    source_data=json.loads(source.read_text(encoding="utf8"))
    cases=read_selected_cases(source)
    geometry=TubeGeometry(config.tube_D_int_mm,config.tube_t_mm)
    material=ElasticMaterial(config.E,config.nu)
    cases_out=[]; models_out=[]; tractions=[]; model_data=[]; waveform_rows=[]
    for item in cases:
        tensor=coefficients(item)
        cid=item["case_id"]
        metrics=max_traction_components(tensor,config.cube_side_mm)
        reference=score_history(tensor,config,high_precision=True)
        validation=independent_maximum(tensor,config,seed=cid)
        if abs(validation-reference["full_fs_ext"]) > 1e-6*reference["full_fs_ext"]:
            raise RuntimeError(f"Case {cid} full-plane reference not reproduced independently")
        first=dict(case_id=cid,family=item["family"],**metrics,
                   original_fs_ext=reference["full_fs_ext"],
                   original_error_percent=reference["error_percent"],
                   check_difference_percent=100*(validation/reference["full_fs_ext"]-1))
        if cid==1739:
            exact,ninfo=projection_plane_stress(tensor,seed=cid)
            first.update(best_plane_stress_hoop_minimum(tensor,ninfo["free_surface_normal"]))
        cases_out.append(_flatten_for_output(first))
        for row in face_traction_program(tensor,config.cube_side_mm):
            tractions.append({"case_id":cid,**row})
        models=[("original",tensor,{"original_stress_tensor":True}),
                ("three_normal",*projection_triaxial_normal(tensor,seed=cid)),
                ("plane_stress",*projection_plane_stress(tensor,seed=cid)),
                ("tube_signed",*projection_tube_signed(tensor,seed=cid)),
                ("tube_positive_inner",*projection_tube_positive_internal(tensor,geometry,seed=cid))]
        for name,projected,meta in models:
            s=reference if name=="original" else score_history(projected,config,high_precision=True)
            diff=tensor_relative_distance(tensor,projected)
            score={"case_id":cid,"family":item["family"],"model":name,
                   "tensor_distance_percent":diff,"fs_ext_full":s["full_fs_ext"],
                   "fs_ext_seancp":s["seancp_fs_ext"],"error_percent":s["error_percent"],
                   "error_retained_percent_points":s["error_percent"]-reference["error_percent"]}
            if name=="tube_positive_inner":
                loads=TubeLoads(p_i=HarmonicLoad(*meta["pressure_internal_mpa"]),
                                force=HarmonicLoad(*meta["additional_axial_force_n"]),
                                torque=HarmonicLoad(*meta["torque_nmm"]))
                wall=[];vms=[]
                for r in np.linspace(geometry.r_i,geometry.r_o,11):
                    hist=harmonic_history(geometry,material,loads,float(r))
                    sc=score_history(np.stack([hist.S_mean,hist.S_sin,hist.S_cos]),config)
                    wall.append((float(r),sc["full_fs_ext"],sc["seancp_fs_ext"]))
                    for phase in np.linspace(0,2*np.pi,48,endpoint=False):
                        st,_=hist.at(float(phase));vms.append(float(von_mises(st)))
                top_ref=max(wall,key=lambda x:x[1]);top_sean=max(wall,key=lambda x:x[2])
                score.update({"wall_wide_error_percent":100*(top_sean[2]/top_ref[1]-1),
                              "governing_reference_radius_mm":top_ref[0],
                              "governing_seancp_radius_mm":top_sean[0],
                              "peak_sampled_wall_von_mises_mpa":max(vms)})
            models_out.append(score)
            if (cid==2213 and name=="original") or (cid==1739 and name=="original") or (cid==6920 and name=="three_normal"):
                if cid==6920:
                    Q=np.asarray(meta["fixed_axes_columns"])
                    local=Q.T@projected@Q
                    comps=[("sigma_xx",local[:,0,0]),("sigma_yy",local[:,1,1]),("sigma_zz",local[:,2,2])]
                elif cid==1739:
                    comps=[("sigma_xx",tensor[:,0,0]),("sigma_yy",tensor[:,1,1]),("tau_xy",tensor[:,0,1])]
                else:
                    comps=[("sigma_xx",tensor[:,0,0]),("sigma_yy",tensor[:,1,1]),("sigma_zz",tensor[:,2,2])]
                for component,vector in comps:
                    m,sn,co=vector; amplitude=float(np.hypot(sn,co))
                    phase=float(np.rad2deg(np.arctan2(co,sn)))
                    waveform_rows.append({"case_id":cid,"model":name,"stress_component":component,
                        "mean_mpa":float(m),"sine_mpa":float(sn),"cosine_mpa":float(co),
                        "amplitude_mpa":amplitude,"phase_deg":phase,
                        "minimum_mpa":float(m-amplitude),"maximum_mpa":float(m+amplitude),
                        "minimum_5mm_face_kn":float((m-amplitude)*config.cube_side_mm**2/1000),
                        "maximum_5mm_face_kn":float((m+amplitude)*config.cube_side_mm**2/1000)})
            model_data.append({"case_id":cid,"family":item["family"],"model":name,
                               "stress_coefficients_mpa":projected.tolist(),
                               "mapping":meta})
    write_csv(out/"feasibility_cases.csv",cases_out)
    write_csv(out/"projection_results.csv",models_out)
    write_csv(out/"ideal_cube_face_tractions.csv",tractions)
    write_csv(out/"reduced_control_waveforms.csv",waveform_rows)
    (out/"projected_histories.json").write_text(json.dumps({
        "definitions":"Local, idealised elastic stress tensors; no FEA or rig validation",
        "reference_configuration":source_data["config"],
        "config":{"cube_side_mm":config.cube_side_mm,"tube_D_int_mm":config.tube_D_int_mm,
                  "tube_thickness_mm":config.tube_t_mm},
        "records":model_data},indent=2)+"\n",encoding="utf8")
    if plots:
        from seancp.feasibility_plots import plot_reduction, plot_cube_loads
        plot_reduction(models_out,out/"error_retention.png")
        plot_cube_loads(cases_out,out/"illustrative_cube_forces.png")
        from seancp.unconstrained_plots import plot_comparison
        from seancp.unconstrained import TensorStudyConfig
        fsconfig=TensorStudyConfig(young_mpa=config.E,nu=config.nu,
                                   sigma_y_mpa=config.sigma_y,k_fs=config.k_fs,n_cases=1)
        for cid,mode in ((2213,"original"),(1739,"original"),(6920,"three_normal"),(7348,"original")):
            chosen=next(x for x in model_data if x["case_id"]==cid and x["model"]==mode)
            history=harmonic_from_stress(np.asarray(chosen["stress_coefficients_mpa"]),config)
            plot_comparison(history,fsconfig,out/f"case_{cid}_{mode}_3d.png",
                            label=f"Case {cid} ({mode.replace('_',' ')})",algorithm="mce")
    return cases_out,models_out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,default=Path(__file__).resolve().parents[1]/
                   "results/phase4_reference/selected_cases.json")
    p.add_argument("--out",type=Path,default=Path("results/phase5"))
    p.add_argument("--no-plots",action="store_true")
    args=p.parse_args()
    rows,models=calculate(args.out,args.reference,FeasibilityConfig(),plots=not args.no_plots)
    for row in rows:
        cid=row["case_id"]
        print(f"Case {cid}, original FS_ext error {row['original_error_percent']:+.3f}%, "
              f"5mm-face max force {row['max_face_component_load_kn']:.2f} kN")
        for m in (x for x in models if x["case_id"]==cid and x["model"]!="original"):
            print(f"  {m['model']:20} tensor change {m['tensor_distance_percent']:6.1f}% "
                  f"FS_ext error {m['error_percent']:7.2f}%")


if __name__=="__main__":
    main()
