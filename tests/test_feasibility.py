"""Mechanical and numerical checks for Phase 5 model reductions."""
import numpy as np
import pytest
from pathlib import Path

from seancp.feasibility import (
    FeasibilityConfig, coefficients, face_traction_program, max_traction_components,
    projection_plane_stress, projection_triaxial_normal, projection_tube_signed,
    projection_tube_positive_internal, project_lorentz, best_plane_stress_hoop_minimum, read_selected_cases,
    score_history, tensor_relative_distance
)
from seancp.thick_tube import TubeGeometry

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"results/phase4_reference/selected_cases.json"
CASES={x["case_id"]:coefficients(x) for x in read_selected_cases(SRC)}
CFG=FeasibilityConfig()


def test_ideal_uniform_cube_boundary_is_in_static_equilibrium():
    """Opposite face resultants and torques cancel if stress is symmetric."""
    side=5.0
    for C in CASES.values():
        force=np.zeros((3,3))
        moment=np.zeros((3,3))
        for axis in range(3):
            for sign in (-1.,1.):
                n=np.eye(3)[axis]*sign
                pos=(side/2)*n
                f=(C@n)*(side**2)
                force+=f
                moment+=np.cross(np.broadcast_to(pos,(3,3)),f)
        np.testing.assert_allclose(force,0,atol=1e-11)
        np.testing.assert_allclose(moment,0,atol=1e-10)


def test_case2213_is_exact_three_normal_axis_loading():
    c=CASES[2213]
    fitted,_=projection_triaxial_normal(c,seed=2213)
    assert tensor_relative_distance(c,fitted)<1e-8
    values=max_traction_components(c,5)
    assert values["max_face_shear_component_load_kn"]==0
    assert values["normal_axes_change_sign"]==[True,True,True]
    assert values["max_face_component_load_kn"]==pytest.approx(10.3875305,rel=1e-7)
    s=score_history(fitted,CFG)
    assert s["error_percent"]==pytest.approx(-37.386,abs=.08)


def test_case1739_has_exact_plane_stress_state():
    c=CASES[1739]
    fitted,meta=projection_plane_stress(c,seed=1739)
    assert tensor_relative_distance(c,fitted)<1e-8
    n=meta["free_surface_normal"]
    np.testing.assert_allclose(c@np.asarray(n),0,atol=1e-8)
    s=score_history(fitted,CFG)
    assert s["error_percent"]==pytest.approx(-12.330,abs=.1)


def test_case6920_has_large_error_after_three_normal_projection():
    c=CASES[6920]
    fitted,meta=projection_triaxial_normal(c,seed=6920)
    assert tensor_relative_distance(c,fitted)==pytest.approx(16.3092,rel=.002)
    assert len(meta["fixed_axes_columns"])==3
    assert score_history(fitted,CFG)["error_percent"]==pytest.approx(-7.573,abs=.15)


def test_cases6920_7348_cannot_be_exactly_plane_stress_or_tube_signed():
    for cid in (6920,7348):
        c=CASES[cid]
        a,_=projection_plane_stress(c,seed=cid)
        b,_=projection_tube_signed(c,seed=cid)
        assert tensor_relative_distance(c,a)>30
        assert tensor_relative_distance(c,b)>10


def test_nonnegative_harmonic_pressure_cone_is_exact():
    signals=[np.array([10.,2.,3.]),np.array([0.,5.,0.]),
             np.array([-10.,5.,0.]),np.array([2.,3.,4.])]
    expected=[np.array([10.,2.,3.]),np.array([2.5,2.5,0.]),
              np.zeros(3),np.array([3.5,2.1,2.8])]
    for inp,ans in zip(signals,expected):
        result=project_lorentz(inp)
        np.testing.assert_allclose(result,ans,atol=1e-12)
        assert result[0] + 1e-12 >= np.linalg.norm(result[1:])


def test_positive_pressure_tube_is_reconstructed_and_changes_case2213():
    c=CASES[2213]
    geom=TubeGeometry(12.0,1.5)
    fitted,meta=projection_tube_positive_internal(c,geom,seed=2213)
    m,s,co=meta["pressure_internal_mpa"]
    assert m+1e-10>=np.hypot(s,co)
    assert meta["pressure_range_mpa"][0]>=-1e-10
    assert tensor_relative_distance(c,fitted)>50
    assert score_history(fitted,CFG)["error_percent"]>-1


def test_face_traction_program_contains_symmetric_harmonic_opposing_forces():
    for c in CASES.values():
        rows=face_traction_program(c,5.)
        assert len(rows)==18
        for axis in 'xyz':
            plus=[r for r in rows if r['face_axis']==axis and r['face']=='+']
            minus=[r for r in rows if r['face_axis']==axis and r['face']=='-']
            for p,m in zip(plus,minus):
                for component in ('force_mean_kn','force_sin_kn','force_cos_kn'):
                    assert p[component]==pytest.approx(-m[component])



def test_case1739_can_not_have_nonnegative_internal_pressure_for_exact_outer_wall_fit():
    tensor=CASES[1739]
    _,meta=projection_plane_stress(tensor,seed=1739)
    b=best_plane_stress_hoop_minimum(tensor,meta["free_surface_normal"])
    assert b["most_favourable_hoop_minimum_mpa"]==pytest.approx(-143.77017,abs=0.001)
