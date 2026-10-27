"""Phase 3 reproducibility, amplitude and method-comparison regression tests."""
import csv
from pathlib import Path
import numpy as np
import pytest
from seancp import evaluate_fs_ext
from seancp.thick_tube import harmonic_history
from seancp.study import (StudyConfig, FAMILIES, fixed_cases, random_cases,
                          evaluate_case, peak_vm_sampled, independent_global_check,
                          run_study)


def test_reproducible_harmonic_cases_and_pressure_bounds():
    cfg=StudyConfig(seed=678)
    a=random_cases(cfg,3)
    b=random_cases(cfg,3)
    assert len(a)==3*len(FAMILIES)
    assert len(fixed_cases())==10
    for x,y in zip(a,b):
        assert x==y
        assert x.loads.p_i.extrema[0]>=0
        assert x.loads.p_i.extrema[1]<=cfg.max_pressure_mpa
        assert x.loads.force.amplitude <= cfg.max_axial_amplitude_n
        assert x.loads.torque.amplitude <= cfg.max_torque_amplitude_nmm


def test_proportional_amplitudes_are_equal():
    cfg=StudyConfig()
    case=next(c for c in fixed_cases() if c.case_id=="control_pressure_axial_torque_proportional")
    history=harmonic_history(cfg.geometry,cfg.material,case.loads,"outer")
    normals=np.random.default_rng(123).normal(size=(51,3))
    vals=[evaluate_fs_ext(history,normals,algorithm=alg,k_fs=cfg.k_fs,
                          sigma_y=cfg.sigma_y_mpa) for alg in ("longest_chord","mcc","mce")]
    np.testing.assert_allclose(vals[0],vals[1],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(vals[0],vals[2],rtol=1e-13,atol=1e-14)


def test_known_mce_limited_search():
    cfg=StudyConfig()
    case=next(c for c in random_cases(cfg,18) if c.case_id=="torque_90_0014")
    result,_=evaluate_case(case,cfg,"mce",[7.5],quality="fine")
    assert result["fs_ext_reference"]==pytest.approx(.002441848476,rel=2e-6)
    assert result["error_wall_percent"]==pytest.approx(-3.768337,abs=.02)
    other,_=evaluate_case(case,cfg,"longest_chord",[7.5],quality="fine")
    assert other["error_wall_percent"]==pytest.approx(0,abs=.002)
    hist=harmonic_history(cfg.geometry,cfg.material,case.loads,"outer")
    independent,_=independent_global_check(hist,cfg,"mce")
    assert independent==pytest.approx(result["fs_ext_reference"],rel=1e-7)


def test_input_and_sampled_von_mises():
    cfg=StudyConfig()
    assert all(peak_vm_sampled(c,cfg,12)>=0 for c in fixed_cases())
    with pytest.raises(ValueError): StudyConfig(sigma_y_mpa=-1)
    with pytest.raises(ValueError): random_cases(cfg,-1)


def test_small_study_and_safe_resume(tmp_path: Path):
    cfg=StudyConfig(seed=1)
    first=run_study(tmp_path,n_per_family=0,top_n=2,config=cfg,plot=False)
    assert first["manifest"]["screening"]["candidates"]==10
    assert len(first["worst"])==2
    for filename in ("screening_cases.csv","screening_radial.csv",
                     "refined_candidates.csv","worst_four.csv",
                     "manifest.json","convergence_checks.csv","screening_signature.json"):
        assert (tmp_path/filename).exists()
    with (tmp_path/"screening_cases.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert {r["algorithm"] for r in rows}=={"longest_chord","mcc","mce"}
    next_run=run_study(tmp_path,n_per_family=0,top_n=2,config=cfg,plot=False,resume=True)
    assert next_run["manifest"]["screening"]["cache_reused"]
    with pytest.raises(ValueError,match="configuration differs"):
        run_study(tmp_path,n_per_family=1,top_n=2,config=cfg,plot=False,resume=True)


def test_figure_contains_plane_paths(tmp_path: Path):
    from seancp.study_plots import plot_damage_landscape
    c=StudyConfig();case=fixed_cases()[0]
    h=harmonic_history(c.geometry,c.material,case.loads,"outer")
    out=tmp_path/"figure.png"
    plot_damage_landscape(h,c,out,algorithm="mce")
    assert out.exists() and out.stat().st_size>5000
