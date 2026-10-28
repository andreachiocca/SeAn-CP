"""Tests for the single-sided notch FE interface. MAPDL is NOT executed here."""
from pathlib import Path
import numpy as np
import pytest
from seancp.notched_tube_search import (
    GaugeElement, generate_controls, validate_controls, history_from_response,
    read_influence_basis, region_comparison, screen_notch_controls,
    write_controls_macro, UNIT_CONTROLS, tensor_from_six,
)
from seancp.thick_tube import TubeGeometry, ElasticMaterial, stress_tensor

ROOT=Path(__file__).resolve().parents[1]


def smooth_reference_basis():
    """Analytical smooth tube for testing transport only, not a notch FE result."""
    g=TubeGeometry(12,1.5)
    mat=ElasticMaterial(210000,0.3)
    S=np.array([stress_tensor(g,'outer',p_i=1),
                stress_tensor(g,'outer',force=1000),
                stress_tensor(g,'outer',torque=1000)])
    E=mat.strain(S)
    return GaugeElement(87,S,E,np.array([7.5,0,0]))


def test_geometry_is_3d_and_asymmetric_one_side():
    s=(ROOT/'apdl/notched_tube.mac').read_text()
    assert 'VSBV,TUBEID,TOOLID' in s
    assert 'NOTCH_ON=1' in s
    assert 'SPHERE,0,NOTCH_R' in s
    assert 'NOTCH_C=RO+NOTCH_R-NOTCH_D' in s
    assert 'NOTCH_D,GE,RO-RI' in s
    assert 'ET,1,187' in s and 'VMESH,ALL' in s
    assert 'CSYS,1' in s and 'NSEL,S,LOC,X,RI-0.01,RI+0.01' in s
    assert 'SF,ALL,PRES,PI' in s
    assert 'FATOT=FA+PI*3.141592653589793*RI*RI' in s
    assert 'NOTCH_D=0.45' in s
    assert 'BASIS_MODE=0' in s
    assert 'SOLVE' in s and '*USE,export_notch' in s
    assert 'ESEL,S,CENT,X' in (ROOT/'apdl/export_notch.mac').read_text()


def test_pressure_nonnegative_over_every_phase():
    rng=np.random.default_rng(11)
    for _ in range(35):
        c=generate_controls(rng,40,10000,40000)
        validate_controls(c,pmax=40,force_limit=10000,torque_limit=40000)
        phi=np.linspace(0,2*np.pi,301)
        actual=c[0,0]+c[1,0]*np.sin(phi)+c[2,0]*np.cos(phi)
        assert min(actual)>=-1e-12 and max(actual)<=40+1e-12
    bad=c.copy();bad[0,0]=2;bad[1,0]=5;bad[2,0]=0
    with pytest.raises(ValueError,match='pressure'):
        validate_controls(bad)


def test_response_superposition_matches_thick_wall_analytic_solution():
    g=TubeGeometry(12,1.5)
    e=smooth_reference_basis()
    c=np.array([[20,2000,10000],[10,5000,11000],[5,1000,-3000]],float)
    h=history_from_response(e,c)
    for i,T in enumerate((h.S_mean,h.S_sin,h.S_cos)):
        ref=stress_tensor(g,'outer',p_i=c[i,0],force=c[i,1],torque=c[i,2])
        np.testing.assert_allclose(T,ref,rtol=1e-12,atol=1e-12)
    assert UNIT_CONTROLS.tolist()==[1,1000,1000]
    np.testing.assert_allclose(h.E_sin,ElasticMaterial(210000,0.3).strain(h.S_sin))


def test_fe_files_roundtrip_uses_engineering_shear_and_noncontiguous_ids(tmp_path):
    e=smooth_reference_basis()
    def six(x):
        return [x[0,0],x[1,1],x[2,2],x[0,1],x[1,2],x[0,2]]
    for name,dat,factor in [('fe_stress.csv',e.stress_response,1),
                            ('fe_strain.csv',e.strain_response,2)]:
        with (tmp_path/name).open('w') as f:
            for basis,tensor in enumerate(dat,1):
                values=six(tensor);values[3:]=np.array(values[3:])*factor
                f.write(','.join(str(k) for k in (basis,87,*values))+',\n')
    with (tmp_path/'fe_coordinates.csv').open('w') as f:
        for basis in range(1,4):f.write(f'{basis},87,7.5,0,0,\n')
    v=read_influence_basis(tmp_path)
    assert list(v)==[87]
    np.testing.assert_allclose(v[87].stress_response,e.stress_response)
    np.testing.assert_allclose(v[87].strain_response,e.strain_response)
    np.testing.assert_allclose(v[87].coordinates,[7.5,0,0])
    np.testing.assert_allclose(tensor_from_six([1,2,3,4,5,6],engineering=True)[0,1],2.)


def test_missing_basis_and_misordered_files_rejected(tmp_path):
    (tmp_path/'fe_stress.csv').write_text('1,87,1,1,1,0,0,0\n2,87,1,1,1,0,0,0\n')
    (tmp_path/'fe_strain.csv').write_text('1,87,1,1,1,0,0,0\n2,87,1,1,1,0,0,0\n')
    with pytest.raises(ValueError,match='Missing'):
        read_influence_basis(tmp_path)
    (tmp_path/'fe_strain.csv').write_text('2,87,1,1,1,0,0,0\n1,87,1,1,1,0,0,0\n')
    with pytest.raises(ValueError,match='indices'):
        read_influence_basis(tmp_path)


def test_region_comparison_and_random_search_on_smooth_synthetic_basis():
    e=smooth_reference_basis()
    controls=np.array([[22,0,0],[15,8000,-12000],[0,4500,28000]],float)
    scores=region_comparison({87:e},controls,algorithm='mce',coarse=True)
    assert scores['reference_element']==scores['seancp_element']==87
    assert np.isfinite(scores['error_percent'])
    assert scores['reference_fs_ext']>=0
    search=screen_notch_controls({87:e},seed=4,trials=2,algorithm='longest_chord',
                                 pmax=40,fmax=10000,tmax=40000,
                                 screen_elements=0,shortlisted=1,max_vm=600)
    assert search['requested']==2 and search['tested']<=2
    if search['shortlist']:
        assert search['shortlist'][0]['refined']['reference_element']==87


def test_apdl_load_macro_emits_all_coefficients(tmp_path):
    c=np.array([[21,0,0],[12,6000,-3000],[4,-2000,25000]],float)
    dest=tmp_path/'notch_loads.mac'
    write_controls_macro(dest,c)
    text=dest.read_text()
    assert text.count('CTRL(')==10  # one comment + nine APDL assignments
    assert 'CTRL(1,1)=21' in text
    assert 'CTRL(3,3)=25000' in text
    with pytest.raises(ValueError):
        write_controls_macro(dest,np.array([[0,0,0],[1,0,0],[0,0,0]]))


def test_geometry_design_sweep_generates_real_decks(tmp_path):
    import sys
    sys.path.insert(0,str(ROOT/'examples'))
    from generate_notch_geometry_sweep import prepare
    out=prepare(tmp_path,[.25,.45,.9],[1.0,1.5],mesh_h=.65)
    assert len(out)==7
    assert (tmp_path/'geometry_sweep.csv').exists()
    assert 'NOTCH_ON=0' in (tmp_path/'smooth_reference'/'notched_tube.mac').read_text()
    model=(tmp_path/'R1p50_D0p45'/'notched_tube.mac').read_text()
    assert 'NOTCH_R=1.5' in model and 'NOTCH_D=0.45' in model
    assert 'MESH_H=0.65' in model
    assert 'BASIS_MODE=1' in model
    assert (tmp_path/'R1p50_D0p45'/'export_notch.mac').exists()


def test_inverse_target_fit_recovers_valid_nonnegative_controls():
    from seancp.notched_tube_search import fit_target_to_notch
    e=smooth_reference_basis()
    c=np.array([[20,1000,0],[6,5000,15000],[3,2000,10000]],float)
    h=history_from_response(e,c)
    target=np.array([h.S_mean,h.S_sin,h.S_cos])
    result=fit_target_to_notch(e,target,pmax=40,fmax=10000,tmax=40000)
    assert result['stress_coefficient_distance_percent']<1.0
    fitted=result['controls']
    validate_controls(fitted,pmax=40,force_limit=10000,torque_limit=40000)
    h2=history_from_response(e,fitted)
    assert np.linalg.norm(h2.S_sin-h.S_sin)<2.0
