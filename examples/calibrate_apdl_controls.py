"""Invert FE control-to-gauge linear response for triaxial/planar setups.

Run unit-control FE solves first with six_arm_calibration.mac or
planar_calibration.mac. The 3 resulting load steps are basis columns;
the resulting stress coefficients represent the actual central-region FE
response, not the forces assumed by a nominal stress-times-area rule.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from generate_apdl_loads import make_macro
from postprocess_apdl_fs_ext import load_rows, tensor_from_six

CONTROL = {"six_arm":(0,1,2), "planar":(0,1,3)}

def fit(basis_file:Path,case:dict,geometry:str):
    ids, arrays = load_rows(basis_file)
    index=CONTROL[geometry]
    columns=[]
    for loadstep in (1,2,3):
        selected=arrays[ids[:,0] == loadstep]
        if len(selected) < 1: raise ValueError(f"No basis results for load step {loadstep}")
        columns.append(np.mean(selected[:,index],axis=0))
    M=np.column_stack(columns)
    if np.linalg.cond(M)>1e7:
        raise ValueError(f"FE response matrix ill-conditioned {np.linalg.cond(M):.3g}")
    T=np.array([case[f"stress_{k}_mpa"] for k in ("mean","sin","cos")])
    targets=np.array([np.diag(x) if geometry=="six_arm" else [x[0,0],x[1,1],x[0,1]] for x in T])
    amplitudes=np.linalg.solve(M,targets.T).T
    output=np.zeros((3,3,3))
    for k in range(3):
        if geometry=="six_arm":
            np.fill_diagonal(output[k],amplitudes[k])
        else:
            output[k,0,0],output[k,1,1],output[k,0,1]=amplitudes[k]
            output[k,1,0]=output[k,0,1]
    return output, M, amplitudes


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--geometry",required=True,choices=CONTROL)
    p.add_argument("--basis",type=Path,required=True,help="fe_stress.csv from calibration MAPDL run")
    p.add_argument("--case-id",type=int,default=None)
    p.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1]/"results/phase4_reference/selected_cases.json")
    p.add_argument("--output",type=Path,required=True,help="APDL calibrated .mac filename")
    args=p.parse_args()
    case_id=args.case_id or (2213 if args.geometry=="six_arm" else 1739)
    cases=json.loads(args.source.read_text())["selected_cases"]
    case=next((x for x in cases if x["case_id"]==case_id),None)
    if case is None:raise ValueError(f"Unknown case {case_id}")
    if (args.geometry=="six_arm" and case_id!=2213) or (args.geometry=="planar" and case_id!=1739):
        raise ValueError("These fixed-axis fixtures were designed for cases 2213 and 1739 only")
    out,M,amp=fit(args.basis,case,args.geometry)
    args.output.write_text(make_macro(out))
    print("Central stress response matrix (MPa per unit nominal control):\n",M)
    print("Corrected control coefficients [mean,sin,cos] x 3:\n",amp)
    print("Wrote",args.output,"(rerun actual specimen with this macro)")

if __name__=="__main__": main()
