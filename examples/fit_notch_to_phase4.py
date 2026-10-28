"""Fit four Phase 4 stress histories to a MAPDL-computed notch influence basis.

No FE files? No result. It never substitutes an axisymmetric or artificial
notch stress field for missing MAPDL outputs.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from seancp.notched_tube_search import (read_influence_basis,fit_target_to_notch,
                                       history_from_response,write_controls_macro,
                                       region_comparison)

def run(folder:Path,output:Path,*,case:int=2213,pmax=40,fmax=10000,tmax=40000,
        max_elements=20,method='mce'):
    source=json.loads((ROOT/'results/phase4_reference/selected_cases.json').read_text())
    item=next((v for v in source['selected_cases'] if v['case_id']==case),None)
    if item is None:raise ValueError('Case not found')
    target=np.array([item[f'stress_{key}_mpa'] for key in ('mean','sin','cos')])
    allbasis=read_influence_basis(folder)
    els=sorted(allbasis.values(),key=lambda e:np.linalg.norm(e.stress_response[0])*pmax+
               np.linalg.norm(e.stress_response[1])*fmax/1000+
               np.linalg.norm(e.stress_response[2])*tmax/1000,reverse=True)
    if max_elements>0:els=els[:max_elements]
    records=[]
    for el in els:
        result=fit_target_to_notch(el,target,pmax=pmax,fmax=fmax,tmax=tmax)
        check=region_comparison({el.element_id:el},result['controls'],algorithm=method,coarse=False)
        records.append(dict(element_id=el.element_id,
                     distance_percent=result['stress_coefficient_distance_percent'],
                     fs_error_percent=check['error_percent'],
                     fs_full=check['reference_fs_ext'],
                     fit_converged=result['converged'],
                     controls=result['controls'].tolist()))
    records.sort(key=lambda x:x['distance_percent'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'target_fits.json').write_text(json.dumps(records,indent=2)+'\n')
    if records:
        write_controls_macro(output/'notch_loads_best_fit.mac',np.array(records[0]['controls']))
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--basis-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--case',type=int,default=2213)
    p.add_argument('--pmax',type=float,default=40)
    p.add_argument('--fmax',type=float,default=10000)
    p.add_argument('--tmax',type=float,default=40000)
    p.add_argument('--max-elements',type=int,default=20)
    p.add_argument('--method',choices=['longest_chord','mcc','mce'],default='mce')
    a=p.parse_args()
    print(run(a.basis_dir,a.out,case=a.case,pmax=a.pmax,fmax=a.fmax,
              tmax=a.tmax,max_elements=a.max_elements,method=a.method)[:3])
