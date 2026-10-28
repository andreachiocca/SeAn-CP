"""Search local FS_ext limitations on actual MAPDL three-control FE responses.

Requires a genuine run of apdl/notched_tube.mac with BASIS_MODE=1.
Use --trials for a reproducible screening; no FE results are invented.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from seancp.notched_tube_search import (read_influence_basis,screen_notch_controls,
                                       write_controls_macro,history_from_response)
from seancp.unconstrained import TensorStudyConfig
from seancp.unconstrained_plots import plot_comparison


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--basis-dir',type=Path,required=True,help='MAPDL UNIT basis FE CSV directory')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trials',type=int,default=60)
    p.add_argument('--seed',type=int,default=31027)
    p.add_argument('--algorithm',choices=['longest_chord','mcc','mce'],default='mce')
    p.add_argument('--pressure-max',type=float,default=40)
    p.add_argument('--force-max',type=float,default=10000)
    p.add_argument('--torque-max',type=float,default=40000)
    p.add_argument('--vm-limit',type=float,default=600)
    p.add_argument('--screen-elements',type=int,default=12,help='0=all exported notch elements')
    a=p.parse_args()
    elements=read_influence_basis(a.basis_dir)
    result=screen_notch_controls(elements,seed=a.seed,trials=a.trials,pmax=a.pressure_max,
            fmax=a.force_max,tmax=a.torque_max,algorithm=a.algorithm,
            max_vm=a.vm_limit,screen_elements=a.screen_elements)
    a.out.mkdir(parents=True,exist_ok=True)
    # Built-in JSON encoder cannot handle numpy arrays.
    def convert(z):return z.tolist() if isinstance(z,np.ndarray) else (int(z) if isinstance(z,np.integer) else float(z))
    (a.out/'screening.json').write_text(json.dumps(result,default=convert,indent=2)+'\n')
    for rank,v in enumerate(result['shortlist'],1):
        write_controls_macro(a.out/f'notch_loads_rank{rank}.mac',v['controls'])
        eid=v['refined']['reference_element']
        h=history_from_response(elements[eid],v['controls'])
        plot_comparison(h,TensorStudyConfig(),a.out/f'notch_rank{rank}_FE_{a.algorithm}_3d.png',
                        label=f'Notched tube, rank {rank}, FE element {eid}',
                        algorithm=a.algorithm,three_d=True)
        print(rank,v['refined']['error_percent'],v['refined']['reference_fs_ext'])
    print('Screened',result['tested'],'of',result['requested'],'candidate histories')

if __name__=='__main__':main()
