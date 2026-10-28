"""Freeze exact Phase 4 stress coefficients as MAPDL macros, in MPa.

Run from anywhere. Individual MAPDL drivers use /INPUT,case_XXXX,mac.
The coefficients remain targets, not fabricated FE results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ORDER = [(0,0),(1,1),(2,2),(0,1),(1,2),(0,2)]

def make_macro(coefficients):
    matrices = np.asarray(coefficients, dtype=float)
    if matrices.shape != (3, 3, 3):
        raise ValueError("Need 3 symmetric Fourier stress tensors")
    if not np.allclose(matrices, np.swapaxes(matrices, 1, 2), atol=1e-9):
        raise ValueError("Stress tensors must be symmetric")
    lines = ["! Fourier stress coefficients MPa; rows: mean, sin, cos",
             "! cols: xx yy zz xy yz xz"]
    for j,(r,c) in enumerate(ORDER, 1):
        for k in range(3):
            lines.append(f"ST({k+1},{j})={matrices[k,r,c]:.15g}")
    return "\n".join(lines)+"\n"


def generate(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    data=json.loads(source.read_text())
    for case in data["selected_cases"]:
        mats=[case[f"stress_{k}_mpa"] for k in ("mean","sin","cos")]
        (output/f"case_{case['case_id']}.mac").write_text(make_macro(mats))
    return len(data["selected_cases"])

if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=root/"results/phase4_reference/selected_cases.json")
    p.add_argument("--output",type=Path,default=root/"apdl")
    args=p.parse_args()
    print(f"Generated {generate(args.source,args.output)} stress macros in {args.output}")
