"""Prepare independent MAPDL folders for a notch-radius/depth sensitivity study.

This script generates decks, it does NOT run MAPDL or claim FE solutions.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
APDL=ROOT/'apdl'


def _values(text):
    return [float(v) for v in text.split(',') if v.strip()]


def prepare(out:Path, depths, radii, *, mesh_h=.75, inner=6.0,
            outer=7.5, length=48., mode=1, with_smooth=True):
    if not (0<inner<outer and mesh_h>0 and length>0 and mode in (0,1)):
        raise ValueError('Invalid tube dimensions or mode')
    template=(APDL/'notched_tube.mac').read_text()
    files=('export_notch.mac','notch_loads.mac','notch_unit_basis.mac')
    rows=[]
    for rad in radii:
        for dep in depths:
            if dep<=0 or rad<=dep or dep>=outer-inner:
                continue
            name=f'R{rad:.2f}_D{dep:.2f}'.replace('.','p')
            folder=out/name
            folder.mkdir(parents=True,exist_ok=True)
            content=template
            change={'RI':inner,'RO':outer,'LENGTH':length,'NOTCH_R':rad,
                    'NOTCH_D':dep,'MESH_H':mesh_h,'BASIS_MODE':mode}
            for key,value in change.items():
                import re
                content,n=re.subn(rf'(?m)^{key}=[^\n]+$',f'{key}={value:g}',content)
                if n!=1:raise ValueError('Cannot set '+key)
            (folder/'notched_tube.mac').write_text(content)
            for item in files:
                (folder/item).write_bytes((APDL/item).read_bytes())
            rows.append({'folder':name,'notch_radius_mm':rad,'notch_depth_mm':dep,
                         'residual_ligament_mm':outer-inner-dep,
                         'mesh_h_mm':mesh_h,'basis_mode':mode})
    if with_smooth:
        folder=out/'smooth_reference'
        folder.mkdir(parents=True,exist_ok=True)
        content=template.replace('NOTCH_ON=1','NOTCH_ON=0')
        content=content.replace('BASIS_MODE=0',f'BASIS_MODE={mode}')
        for key,value in {'RI':inner,'RO':outer,'LENGTH':length,'MESH_H':mesh_h}.items():
            import re
            content=re.sub(rf'(?m)^{key}=[^\n]+$',f'{key}={value:g}',content)
        (folder/'notched_tube.mac').write_text(content)
        for item in files:(folder/item).write_bytes((APDL/item).read_bytes())
        rows.append({'folder':'smooth_reference','notch_radius_mm':0.,'notch_depth_mm':0.,
                     'residual_ligament_mm':outer-inner,'mesh_h_mm':mesh_h,'basis_mode':mode})
    if not rows:raise ValueError('No valid dimple designs')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'geometry_sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('results/notch_sweep_designs'))
    p.add_argument('--depths',default='.25,.45,.65')
    p.add_argument('--radii',default='1,1.5,2')
    p.add_argument('--mesh-h',type=float,default=.75)
    p.add_argument('--mode',type=int,choices=[0,1],default=1)
    p.add_argument('--no-smooth',action='store_true')
    a=p.parse_args()
    r=prepare(a.out,_values(a.depths),_values(a.radii),mesh_h=a.mesh_h,mode=a.mode,with_smooth=not a.no_smooth)
    print('Prepared',len(r),'MAPDL input folders, no solver runs performed.')
