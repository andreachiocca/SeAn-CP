"""Read MAPDL critical-region Fourier tensors and compare FS_ext searches.

MAPDL must have solved THREE independent linear static cases: mean, sine,
cosine, with exactly the same mesh. This is not a load-step time series.

Each result row is [basis_id (1..3), element_id, xx, yy, zz, xy, yz, xz].
For strain, MAPDL EPEL shears are engineering shear, divided by 2 here.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from seancp.core import HarmonicHistory, evaluate_fs_ext, full_plane_scan, seancp_fs_ext
from seancp.unconstrained import TensorStudyConfig
from seancp.unconstrained_plots import plot_comparison

ORDER=((0,0),(1,1),(2,2),(0,1),(1,2),(0,2))

def tensor_from_six(components, *, engineering_strain=False):
    c=np.asarray(components,dtype=float)
    if c.shape[-1]!=6:raise ValueError("Expected six tensor components")
    t=np.zeros(c.shape[:-1]+(3,3),dtype=float)
    for j,(r,k) in enumerate(ORDER):
        val=c[...,j]*(0.5 if engineering_strain and r!=k else 1.0)
        t[...,r,k]=val
        t[...,k,r]=val
    return t


def load_rows(path:Path):
    """Read MAPDL whitespace/numeric CSV, with optional trailing comma."""
    rows=[]
    for line in Path(path).read_text().splitlines():
        parts=[x.strip() for x in line.split(",") if x.strip()]
        if not parts or parts[0].startswith(("!","#")): continue
        if len(parts)<8:raise ValueError(f"Expected 8 numeric columns in {path}: {line!r}")
        vals=np.array([float(x.replace("D","E")) for x in parts[:8]])
        if not np.isfinite(vals).all():raise ValueError("Nonfinite FE values")
        rows.append(vals)
    if not rows:raise ValueError(f"Missing data in {path}")
    arr=np.array(rows)
    indices=np.rint(arr[:,:2]).astype(int)
    if not np.allclose(indices,arr[:,:2]) or not set(indices[:,0]).issubset({1,2,3}):
        raise ValueError("Invalid basis index (expect 1=mean 2=sin 3=cos)")
    return indices,arr[:,2:8]


def histories_from_files(directory:Path):
    """Return element_id->HarmonicHistory and FE-vs-Hooke diagnostic."""
    si,s=load_rows(directory/"fe_stress.csv")
    ei,e=load_rows(directory/"fe_strain.csv")
    if len(si)!=len(ei) or not np.array_equal(si,ei):
        raise ValueError("Stress and strain rows must have identical element/basis indices")
    keys=[tuple(x) for x in si.tolist()]
    if len(keys)!=len(set(keys)):
        raise ValueError("Duplicate stress/strain sample")
    elements=sorted(set(si[:,1]))
    config=TensorStudyConfig()
    histories={}
    errors=[]
    for elem in elements:
        sel=si[:,1]==elem
        steps=si[sel,0]
        if sorted(steps.tolist())!=[1,2,3]:
            raise ValueError(f"Element {elem} missing a Fourier coefficient")
        st=tensor_from_six(s[sel])[np.argsort(steps)]
        et=tensor_from_six(e[sel],engineering_strain=True)[np.argsort(steps)]
        h=HarmonicHistory(st[0],st[1],st[2],et[0],et[1],et[2])
        histories[elem]=h
        hooke=((1+config.nu)*st-config.nu*np.trace(st,axis1=1,axis2=2)[:,None,None]*np.eye(3))/config.young_mpa
        errors.append(np.linalg.norm(et-hooke)/max(1e-14,np.linalg.norm(hooke)))
    return histories,dict(element_count=len(histories),max_hooke_relative_error=float(max(errors)),
                          median_hooke_relative_error=float(np.median(errors)))


def run_postprocess(directory:Path,output:Path,*,max_elements=16,algorithm="mce",case_id=2213):
    if algorithm not in ("longest_chord","mcc","mce"):
        raise ValueError("Invalid amplitude algorithm")
    histories,diagnostics=histories_from_files(directory)
    config=TensorStudyConfig()
    output.mkdir(parents=True,exist_ok=True)
    # Sort by actual harmonic stress magnitude, retain the centre (if present)
    # and highest-amplitude critical-region points. This is a sampling policy,
    # not a proof that the omitted elements cannot govern.
    entries=sorted(histories.items(),key=lambda pair: np.linalg.norm(pair[1].S_sin)**2+
                   np.linalg.norm(pair[1].S_cos)**2,reverse=True)
    if max_elements>0: entries=entries[:max_elements]
    rows=[]
    for eid,h in entries:
        semi=seancp_fs_ext(h,algorithm=algorithm,k_fs=config.k_fs,sigma_y=config.sigma_y_mpa,samples_per_path=181)
        maximum,n=full_plane_scan(h,algorithm=algorithm,k_fs=config.k_fs,sigma_y=config.sigma_y_mpa,
                                  n_theta=91,n_phi=181,refine=True)
        rows.append(dict(element_id=int(eid),seancp_fs_ext=float(semi.value),
                         reference_fs_ext=float(maximum),error_percent=float(100*(semi.value/maximum-1))
                         if maximum>0 else 0,reference_normal=n.tolist(),
                         seancp_normal=semi.normal.tolist()))
    if not rows:raise ValueError("No selected gauge elements")
    winner=max(rows,key=lambda x:x['reference_fs_ext'])
    plot_comparison(histories[winner['element_id']],config,output/f"case_{case_id}_FE_{algorithm}_3d.png",
                    label=f"FE gauge element {winner['element_id']} (case {case_id})",algorithm=algorithm)
    plot_comparison(histories[winner['element_id']],config,output/f"case_{case_id}_FE_{algorithm}_2d.png",
                    label=f"FE gauge element {winner['element_id']} (case {case_id})",
                    algorithm=algorithm,three_d=False)
    with (output/"element_comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=['element_id','seancp_fs_ext','reference_fs_ext','error_percent',
                                            'reference_normal','seancp_normal'])
        writer.writeheader();writer.writerows(rows)
    case_data=json.loads((ROOT/"results/phase4_reference/selected_cases.json").read_text())
    orig=next((x for x in case_data['selected_cases'] if x['case_id']==case_id),None)
    target_deviation=None
    if orig is not None:
        origS=np.array([orig[f'stress_{k}_mpa'] for k in ('mean','sin','cos')])
        h=histories[winner['element_id']]
        actual=np.array([h.S_mean,h.S_sin,h.S_cos])
        target_deviation=float(100*np.linalg.norm(actual-origS)/np.linalg.norm(origS))
    report={'status':'FE_IMPORTED_NOT_EXPERIMENTAL','algorithm':algorithm,'case_id':case_id,
            'model_directory':str(directory),'studied_gauge_elements':len(rows),
            'total_exported_gauge_elements':len(histories),
            'selected_by':'largest_reference_fs_ext_in_processed_subset',
            'governing_element':winner,'stress_coefficient_distance_to_target_percent':target_deviation,
            'diagnostics':diagnostics,
            'note':'FE cannot be called experimentally validated. Check mesh convergence and gauge-field uniformity.'}
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fe-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case-id',type=int,default=2213)
    p.add_argument('--algorithm',choices=['longest_chord','mcc','mce'],default='mce')
    p.add_argument('--max-elements',type=int,default=16,help='0=all exported gauge elements')
    a=p.parse_args()
    print(json.dumps(run_postprocess(a.fe_dir,a.output,max_elements=a.max_elements,
                                     algorithm=a.algorithm,case_id=a.case_id),indent=2))
