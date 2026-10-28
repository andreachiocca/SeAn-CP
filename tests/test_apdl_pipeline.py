"""Unit tests for APDL Fourier files; no solver is represented as executed."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from generate_apdl_loads import make_macro, generate
from calibrate_apdl_controls import fit
from postprocess_apdl_fs_ext import load_rows,tensor_from_six,histories_from_files
from seancp.unconstrained import elastic_strain,TensorStudyConfig


def case():
    import json
    return next(x for x in json.loads((ROOT/'results/phase4_reference/selected_cases.json').read_text())['selected_cases'] if x['case_id']==2213)


def test_macro_contains_exact_coefficients():
    c=case();m=np.array([c[f'stress_{k}_mpa'] for k in ('mean','sin','cos')])
    text=make_macro(m)
    assert 'ST(1,1)=-40.6328096659451' in text
    assert 'ST(2,1)=255.002765664997' in text
    assert len(text.splitlines())==20


def test_six_component_roundtrip():
    data=np.array([[1,2,3,4,5,6]],float)
    t=tensor_from_six(data)[0]
    np.testing.assert_array_equal(t,[[1,4,6],[4,2,5],[6,5,3]])
    eng=tensor_from_six(data,engineering_strain=True)[0]
    assert eng[0,1]==2.0 and eng[2,0]==3.0


def test_analytic_fourier_fixture(tmp_path):
    c=case();S=np.array([c[f'stress_{k}_mpa'] for k in ('mean','sin','cos')])
    E=np.array([elastic_strain(x,TensorStudyConfig()) for x in S])
    def six(x):return [x[0,0],x[1,1],x[2,2],x[0,1],x[1,2],x[0,2]]
    for name,vals,mult in [('fe_stress.csv',S,1),('fe_strain.csv',E,2)]:
        with (tmp_path/name).open('w') as f:
            for k,x in enumerate(vals,1):
                comp=six(x);comp[3:]=[z*mult for z in comp[3:]]
                f.write(','.join(str(z) for z in [k,47,*comp])+',\n')
    d,diag=histories_from_files(tmp_path)
    assert list(d)==[47]
    assert diag['max_hooke_relative_error']<1e-12
    np.testing.assert_allclose(d[47].S_cos,S[2])


def test_missing_basis_rejected(tmp_path):
    for name in ('fe_stress.csv','fe_strain.csv'):
        (tmp_path/name).write_text('1,12,1,1,1,0,0,0\n2,12,1,1,1,0,0,0\n')
    with pytest.raises(ValueError,match='missing a Fourier coefficient'):
        histories_from_files(tmp_path)


def test_nonfinite_input_rejected(tmp_path):
    p=tmp_path/'fe_stress.csv'
    p.write_text('1,1,nan,1,1,0,0,0\n')
    with pytest.raises(ValueError,match='Nonfinite'):
        load_rows(p)


def test_calibration_inverts_cross_talk(tmp_path):
    mat=np.array([[.90,.13,.07],[.11,.92,.06],[.04,.09,.88]])
    p=tmp_path/'fe_stress.csv'
    with p.open('w') as f:
        for k in range(3):
            out=mat[:,k]
            f.write(','.join(map(str,[k+1,19,*out,0,0,0]))+'\n')
    _,m,a=fit(p,case(),'six_arm')
    np.testing.assert_allclose(m,mat)
    c=case();target=np.array([np.diag(np.array(c[f'stress_{k}_mpa'])) for k in ('mean','sin','cos')])
    np.testing.assert_allclose((m@a.T).T,target)


def test_static_macro_includes_all_paths():
    root=ROOT/'apdl'
    for p in ['ideal_cube.mac','six_arm.mac','planar_traction_cruciform.mac']:
        s=(root/p).read_text()
        assert 'SOLVE' in s and 'export_critical' in s and 'OUTRES,ALL,ALL' in s
    assert 'VGLUE,ALL' in (root/'six_arm.mac').read_text()
