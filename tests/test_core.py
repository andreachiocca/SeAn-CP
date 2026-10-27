import math

import numpy as np
import pytest

from seancp import (HarmonicHistory, SampledHistory, evaluate_fs_ext,
                    full_plane_scan, seancp_fs_ext)
from seancp.amplitude import harmonic_amplitude


def diag(*x):
    return np.diag(x)


def zero():
    return np.zeros((3,3))


def fixture_history():
    # Non-coaxial stress/strain phases for analytical algorithm checks.
    sm, ss, sc = diag(20, 10, 5), diag(70, -30, -40), zero()
    sc[1,2] = sc[2,1] = 25.0
    em, es, ec = zero(), diag(.0005, -.00025, -.00025), zero()
    es[0,2] = es[2,0] = .0002
    ec[1,2] = ec[2,1] = .0003
    return HarmonicHistory(sm, ss, sc, em, es, ec)


def test_harmonic_amplitudes_analytic_ellipse():
    sin, cos = np.array([[10.,0.,0.]]), np.array([[0.,5.,0.]])
    assert harmonic_amplitude(sin, cos, 'longest_chord')[0] == pytest.approx(10)
    assert harmonic_amplitude(sin, cos, 'mcc')[0] == pytest.approx(10)
    assert harmonic_amplitude(sin, cos, 'mce')[0] == pytest.approx(np.sqrt(125))
    sin, cos = np.array([[10.,0.,0.]]), np.array([[0.,10.,0.]])
    assert harmonic_amplitude(sin, cos, 'mce')[0] == pytest.approx(10*np.sqrt(2))


def test_engineering_shear_and_normal_stress_one_plane():
    E = zero(); E[0,2] = E[2,0] = 0.005
    S = diag(0, 0, 120)
    h = HarmonicHistory(zero(), S, zero(), zero(), E, zero())
    q = np.array([[0.,0.,1.]])
    assert evaluate_fs_ext(h, q, k_fs=.4, sigma_y=300)[0] == pytest.approx(.0116)
    assert evaluate_fs_ext(h, q, k_fs=.4, sigma_y=300, algorithm='mce')[0] == pytest.approx(.0116)


def test_same_normal_for_stress_and_shear_and_clipping():
    E = zero(); E[0,2] = E[2,0] = .005
    h = HarmonicHistory(diag(-100,-100,-100), zero(), zero(), zero(), E, zero())
    assert evaluate_fs_ext(h, [[0,0,1]], k_fs=.4, sigma_y=300)[0] == pytest.approx(.01)
    assert evaluate_fs_ext(h, [[1,0,0]], k_fs=.4, sigma_y=300)[0] == pytest.approx(.01)
    assert evaluate_fs_ext(h, [[0,1,0]], k_fs=.4, sigma_y=300)[0] == pytest.approx(0)


def test_from_endpoints_and_proportional_mcc_mce_agree():
    s0, s1 = diag(100, 0, 0), diag(-100, 0, 0)
    e0, e1 = diag(.001,-.0003,-.0003), diag(-.001,.0003,.0003)
    h = HarmonicHistory.from_endpoints(s0,s1,e0,e1)
    n = np.array([[.6, .0, .8], [.2,.7,.68]])
    v1 = evaluate_fs_ext(h,n,algorithm='longest_chord')
    v2 = evaluate_fs_ext(h,n,algorithm='mcc')
    v3 = evaluate_fs_ext(h,n,algorithm='mce')
    np.testing.assert_allclose(v1,v2,rtol=1e-13)
    np.testing.assert_allclose(v1,v3,rtol=1e-13)


def test_nonproportional_ellipse_distinguishes_mce():
    h = fixture_history()
    n = np.array([[0,0,1.], [1,0,0], [0,1,0]])
    a = evaluate_fs_ext(h,n, algorithm='longest_chord')
    b = evaluate_fs_ext(h,n, algorithm='mcc')
    c = evaluate_fs_ext(h,n, algorithm='mce')
    np.testing.assert_allclose(a,b)
    assert np.all(c >= b - 1e-15)
    assert np.max(c-b) > 1e-6


def test_sampled_triangle_longest_chord_different_from_mcc():
    # Equilateral triangle in engineering shear space, side length 2.
    samples = np.array([[0,0], [2,0], [1,math.sqrt(3)]])
    es, ss = [], []
    for x,y in samples:
        e = zero(); e[0,2] = e[2,0] = x/2; e[1,2] = e[2,1] = y/2
        es.append(e); ss.append(zero())
    hist = SampledHistory(np.array(ss), np.array(es))
    n = [[0,0,1.]]
    assert evaluate_fs_ext(hist,n,algorithm='longest_chord')[0] == pytest.approx(1)
    assert evaluate_fs_ext(hist,n,algorithm='mcc')[0] == pytest.approx(2/math.sqrt(3))
    with pytest.raises(ValueError, match='MCE requires'):
        evaluate_fs_ext(hist,n,algorithm='mce')


def test_seancp_three_paths_and_never_exceeds_reference():
    h = fixture_history()
    for algorithm in ('longest_chord','mcc','mce'):
        semi = seancp_fs_ext(h,algorithm=algorithm,samples_per_path=90)
        exact, normal = full_plane_scan(h,algorithm=algorithm,n_theta=51,n_phi=101)
        assert len(semi.paths) == 3
        assert [p.family for p in semi.paths] == ['strain_range','stress_anchor_0','stress_anchor_1']
        assert semi.value <= exact + 1e-10
        assert math.isclose(np.linalg.norm(semi.normal),1)
        assert math.isclose(np.linalg.norm(normal),1)
        for path in semi.paths:
            assert path.normals.shape == (90,3)
            assert path.values.shape == (90,)
            assert semi.value >= path.best_value - 1e-12


def test_seancp_matches_full_scan_for_uniaxial_harmonic():
    s = diag(110,0,0)
    e = diag(.001,-.0003,-.0003)
    h = HarmonicHistory(zero(),s,zero(),zero(),e,zero())
    semi = seancp_fs_ext(h,samples_per_path=100)
    exact, _ = full_plane_scan(h,n_theta=61,n_phi=121)
    assert semi.value == pytest.approx(exact,abs=1e-10)


def test_rotation_invariance_fullscan_and_plane_values():
    h = fixture_history()
    rng = np.random.default_rng(78)
    Q,_ = np.linalg.qr(rng.normal(size=(3,3)))
    H = HarmonicHistory(*(Q.T @ getattr(h,k) @ Q for k in
                          ('S_mean','S_sin','S_cos','E_mean','E_sin','E_cos')))
    n = rng.normal(size=(23,3)); n /= np.linalg.norm(n,axis=1)[:,None]
    for alg in ('longest_chord','mcc','mce'):
        np.testing.assert_allclose(evaluate_fs_ext(h,n @ Q.T,algorithm=alg),
                                   evaluate_fs_ext(H,n,algorithm=alg), rtol=1e-12,atol=1e-14)


def test_invalid_input_and_nonharmonic_mce_rejected():
    with pytest.raises(ValueError):
        HarmonicHistory(zero(), np.array([[0,1,0],[0,0,0],[0,0,0]]),
                        zero(), zero(), zero(), zero())
    with pytest.raises(ValueError):
        evaluate_fs_ext(fixture_history(), [[0,0,0]])
    with pytest.raises(ValueError):
        evaluate_fs_ext(fixture_history(), [[0,0,1]], sigma_y=0)
    with pytest.raises(ValueError):
        seancp_fs_ext(fixture_history(),samples_per_path=2)


def test_harmonic_history_matches_dense_time_sampling_on_oblique_plane():
    h = fixture_history()
    n = np.array([.33,.51,.79]); n /= np.linalg.norm(n)
    t = np.arange(32768)*2*np.pi/32768
    stress = h.S_mean[None] + np.sin(t)[:,None,None]*h.S_sin + np.cos(t)[:,None,None]*h.S_cos
    strain = h.E_mean[None] + np.sin(t)[:,None,None]*h.E_sin + np.cos(t)[:,None,None]*h.E_cos
    en = np.einsum('tij,j->ti',strain,n)
    shear = 2*(en-(en@n)[:,None]*n)
    sn = np.einsum('i,tij,j->t',n,stress,n)
    sampled_amplitude = np.max(np.linalg.norm(shear - np.mean(shear,axis=0),axis=1))
    exact_damage = sampled_amplitude * (1+.4*max(0,float(np.max(sn)))/300)
    assert evaluate_fs_ext(h,[n],algorithm='longest_chord')[0] == pytest.approx(exact_damage,rel=2e-8)


def test_phase_shift_invariance_of_harmonic_damage_and_search():
    h = fixture_history()
    q = .313
    def rotate_pair(sine,cosine):
        return (sine*np.cos(q)-cosine*np.sin(q),
                sine*np.sin(q)+cosine*np.cos(q))
    ss,sc = rotate_pair(h.S_sin,h.S_cos)
    es,ec = rotate_pair(h.E_sin,h.E_cos)
    shifted = HarmonicHistory(h.S_mean,ss,sc,h.E_mean,es,ec)
    normal = np.array([[.21,.71,.67]])
    for alg in ('longest_chord','mcc','mce'):
        np.testing.assert_allclose(evaluate_fs_ext(h,normal,algorithm=alg),
                                   evaluate_fs_ext(shifted,normal,algorithm=alg),rtol=1e-13)
        original = seancp_fs_ext(h,algorithm=alg)
        alternative = seancp_fs_ext(shifted,algorithm=alg)
        assert original.value == pytest.approx(alternative.value,rel=1e-10)


def test_zero_loading_and_analytical_max_shear_strain():
    h = HarmonicHistory(zero(),zero(),zero(),zero(),zero(),zero())
    assert seancp_fs_ext(h,samples_per_path=24).value == 0
    a = diag(.002,0,-.001)
    h = HarmonicHistory(zero(),zero(),zero(),zero(),a,zero())
    assert seancp_fs_ext(h,k_fs=0,samples_per_path=61).value == pytest.approx(.003,rel=1e-9)
    assert full_plane_scan(h,k_fs=0,n_theta=41,n_phi=81)[0] == pytest.approx(.003,rel=1e-9)
