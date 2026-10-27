"""Reproducible Phase 4 harmonic local-tensor FS_ext benchmarks."""
import json
from pathlib import Path

import numpy as np
import pytest

from seancp.core import evaluate_fs_ext, full_plane_scan, seancp_fs_ext
from seancp.unconstrained import (
    FAMILIES, TensorStudyConfig, case_from_data, elastic_strain,
    generate_cases, history, refine_case, screen_case, _normals,
)


def test_harmonic_tensor_generator_is_deterministic_and_symmetric():
    cfg=TensorStudyConfig(n_cases=14)
    a=list(generate_cases(cfg));b=list(generate_cases(cfg))
    assert len(a)==14 and len(FAMILIES)==7
    assert len({q.family for q in a})==7
    for first,second in zip(a,b):
        for key in ('stress_mean','stress_sin','stress_cos'):
            x=getattr(first,key)
            np.testing.assert_array_equal(x,getattr(second,key))
            np.testing.assert_allclose(x,x.T,atol=1e-12)
        assert history(first,cfg).S_sin.shape==(3,3)


def test_elastic_strain_and_tensor_harmonic_reconstruction():
    cfg=TensorStudyConfig(n_cases=7)
    case=next(q for q in generate_cases(cfg) if q.family=='mixed_3d')
    h=history(case,cfg)
    for phase in (0.,.41,1.3,2.7):
        S,E=h.at(phase)
        np.testing.assert_allclose(E,elastic_strain(S,cfg),atol=1e-16)
    assert not np.allclose(case.stress_sin,case.stress_cos)


def test_mcc_longest_chord_equivalence_for_all_harmonic_normals():
    cfg=TensorStudyConfig(n_cases=7)
    normal=_normals(cfg)
    for case in generate_cases(cfg):
        h=history(case,cfg)
        a=evaluate_fs_ext(h,normal,k_fs=cfg.k_fs,sigma_y=cfg.sigma_y_mpa,algorithm='mcc')
        b=evaluate_fs_ext(h,normal,k_fs=cfg.k_fs,sigma_y=cfg.sigma_y_mpa,algorithm='longest_chord')
        np.testing.assert_allclose(a,b,rtol=2e-14,atol=1e-13)
        rows=screen_case(case,cfg,normal,np.linspace(0,np.pi,49,endpoint=False))
        assert {x['algorithm'] for x in rows}=={'mcc','mce','longest_chord'}
        assert rows[0]['preliminary_error_percent']==rows[1]['preliminary_error_percent']


def test_large_error_case_is_reproduced_independently():
    reference=Path(__file__).resolve().parents[1]/'results/phase4_reference/selected_cases.json'
    if not reference.exists():pytest.skip('Reference case file not packaged')
    payload=json.loads(reference.read_text())
    cfg=TensorStudyConfig(**payload['config'])
    case=case_from_data(next(x for x in payload['selected_cases'] if x['family']=='normal_offphase'))
    result,_=refine_case(case,cfg,'mce',confirm=False)
    assert result['error_percent']==pytest.approx(-37.386,abs=.02)
    h=history(case,cfg)
    semi=seancp_fs_ext(h,k_fs=cfg.k_fs,sigma_y=cfg.sigma_y_mpa,algorithm='mce',samples_per_path=361)
    full,_=full_plane_scan(h,k_fs=cfg.k_fs,sigma_y=cfg.sigma_y_mpa,algorithm='mce',n_theta=121,n_phi=241)
    assert semi.value<.7*full


def test_plots_contain_exact_value_markers(tmp_path):
    from seancp.unconstrained_plots import plot_comparison
    cfg=TensorStudyConfig(n_cases=7)
    case=next(q for q in generate_cases(cfg) if q.family=='mixed_3d')
    image=tmp_path/'surface.png'
    out=plot_comparison(history(case,cfg),cfg,image,label='test',algorithm='mce',three_d=True)
    assert image.stat().st_size>20000
    assert out['semi']<=out['full']*(1+1e-7)
    assert out['error_percent']<=.001


def test_config_bounds():
    with pytest.raises(ValueError):TensorStudyConfig(n_cases=0)
    with pytest.raises(ValueError):TensorStudyConfig(von_mises_fraction=(1.0,.2))
