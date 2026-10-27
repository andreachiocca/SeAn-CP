"""Independent equilibrium, boundary, Hooke-law and harmonic tests for tube model."""

import math

import numpy as np
import pytest

from seancp import (ElasticMaterial, HarmonicLoad, TubeGeometry, TubeLoads,
                    harmonic_history, radial_profile, stress_tensor,
                    von_mises, evaluate_fs_ext)


@pytest.fixture
def geom():
    return TubeGeometry(D_int=12, thickness=1.5)


@pytest.fixture
def steel():
    return ElasticMaterial(E=210000, nu=0.3)


def test_original_specimen_geometry(geom):
    assert geom.r_i == 6
    assert geom.r_o == 7.5
    assert geom.area == pytest.approx(math.pi*20.25)
    assert geom.J == pytest.approx(math.pi/2*(7.5**4-6**4))
    assert geom.radius("mean") == 6.75
    for r in (geom.r_i, geom.r_o):
        assert geom.radius(r) == r


def test_pressure_boundary_and_analytic_lame(geom):
    pressure, exterior = 32.5, 4.0
    S_i = stress_tensor(geom, "inner", p_i=pressure, p_o=exterior)
    S_o = stress_tensor(geom, "outer", p_i=pressure, p_o=exterior)
    A = (pressure*6**2-exterior*7.5**2)/(7.5**2-6**2)
    B = 6**2*7.5**2*(pressure-exterior)/(7.5**2-6**2)
    assert S_i[0,0] == pytest.approx(-pressure)
    assert S_o[0,0] == pytest.approx(-exterior)
    assert S_i[1,1] == pytest.approx(A+B/6**2)
    assert S_o[1,1] == pytest.approx(A+B/7.5**2)
    assert S_i[2,2] == pytest.approx(A)
    assert S_o[2,2] == pytest.approx(A)
    assert S_i[1,1] > S_o[1,1]


def test_internal_pressure_reference_values(geom):
    # Independent closed-form result for D_i=12, t=1.5, p=32.5, p_o=0.
    S_i = stress_tensor(geom, "inner", p_i=32.5)
    S_o = stress_tensor(geom, "outer", p_i=32.5)
    assert np.diag(S_i) == pytest.approx([-32.5, 148.05555555555554, 57.77777777777777])
    assert np.diag(S_o) == pytest.approx([0, 115.55555555555554, 57.77777777777777])


def test_thick_equilibrium_differential_identity(geom):
    p_i, p_o = 50, 1.5
    r = 6.9
    h = 1e-5
    left = stress_tensor(geom, r-h, p_i=p_i,p_o=p_o)[0,0]
    right = stress_tensor(geom, r+h, p_i=p_i,p_o=p_o)[0,0]
    S = stress_tensor(geom,r,p_i=p_i,p_o=p_o)
    radial_balance = (right-left)/(2*h)+(S[0,0]-S[1,1])/r
    assert radial_balance == pytest.approx(0, abs=1e-8)


def test_axial_pressure_resultant_and_compensation(geom):
    pi, po, force = 32.5, 4, 1800
    A = (pi*geom.r_i**2-po*geom.r_o**2)/(geom.r_o**2-geom.r_i**2)
    closed = stress_tensor(geom,"outer",p_i=pi,p_o=po,force=force,end_condition="closed")
    compensated = stress_tensor(geom,"outer",p_i=pi,p_o=po,force=force,
                                end_condition="pressure_compensated")
    assert closed[2,2]*geom.area == pytest.approx(
        pi*math.pi*geom.r_i**2-po*math.pi*geom.r_o**2+force)
    assert compensated[2,2]*geom.area == pytest.approx(force)
    assert closed[2,2]-compensated[2,2] == pytest.approx(A)
    for i in (0,1):
        assert closed[i,i] == pytest.approx(compensated[i,i])


def test_pure_axial_load_exact_area(geom):
    F=15904.31
    sigma = F/geom.area
    for r in ("inner", "mean", "outer"):
        S = stress_tensor(geom, r, force=F)
        np.testing.assert_allclose(S,np.diag([0,0,sigma]),rtol=1e-13,atol=1e-13)


def test_saint_venant_torque_shear_distribution(geom):
    T = 50000
    tau_i=stress_tensor(geom,"inner",torque=T)[1,2]
    tau_o=stress_tensor(geom,"outer",torque=T)[1,2]
    assert tau_i == pytest.approx(T*geom.r_i/geom.J)
    assert tau_o == pytest.approx(T*geom.r_o/geom.J)
    assert tau_i/tau_o == pytest.approx(geom.r_i/geom.r_o)
    # Integral over the cross-section: T = integral( tau(r) * r dA ).
    radii = np.linspace(geom.r_i,geom.r_o,10001)
    tau = T*radii/geom.J
    assert np.sum(0.5*(2*math.pi*tau[1:]*radii[1:]**2+2*math.pi*tau[:-1]*radii[:-1]**2)*np.diff(radii)) == pytest.approx(T, rel=1e-8)


def test_elastic_hooke_strains_and_engineering_shear(geom,steel):
    T=32000
    F=7000
    S=stress_tensor(geom,"outer",p_i=18,force=F,torque=T)
    eps=steel.strain(S)
    assert eps[0,0] == pytest.approx((S[0,0]-steel.nu*(S[1,1]+S[2,2]))/steel.E)
    assert eps[1,1] == pytest.approx((S[1,1]-steel.nu*(S[0,0]+S[2,2]))/steel.E)
    assert eps[2,2] == pytest.approx((S[2,2]-steel.nu*(S[0,0]+S[1,1]))/steel.E)
    assert 2*eps[1,2] == pytest.approx(S[1,2]/steel.G)
    assert eps[0,1] == eps[0,2] == 0


def test_harmonic_reconstruction_at_phases(geom,steel):
    loads=TubeLoads(
        p_i=HarmonicLoad.from_amplitude_phase(22,7,math.pi/5),
        p_o=HarmonicLoad(2,0.3,-0.2),
        force=HarmonicLoad.from_amplitude_phase(1100,15000,-math.pi/3),
        torque=HarmonicLoad.from_amplitude_phase(1200,48000,math.pi/2),
    )
    for end in ("closed","pressure_compensated"):
        for position in ("inner",6.7,"outer"):
            history=harmonic_history(geom,steel,loads,position,end_condition=end)
            for t in np.linspace(0,2*math.pi,11):
                S,E=history.at(t)
                direct=stress_tensor(geom,position,**loads.at(t),end_condition=end)
                np.testing.assert_allclose(S,direct,rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(E,steel.strain(direct),rtol=1e-12,atol=1e-13)


def test_linearity_and_pressure_mean_independence_of_amplitude(geom,steel):
    base=TubeLoads(p_i=HarmonicLoad(mean=35,sine=4),
                   force=HarmonicLoad(sine=5000),torque=HarmonicLoad(cosine=24000))
    shifted=TubeLoads(p_i=HarmonicLoad(mean=52,sine=4),
                   force=HarmonicLoad(sine=5000),torque=HarmonicLoad(cosine=24000))
    a=harmonic_history(geom,steel,base)
    b=harmonic_history(geom,steel,shifted)
    np.testing.assert_allclose(a.S_sin,b.S_sin)
    np.testing.assert_allclose(a.S_cos,b.S_cos)
    np.testing.assert_allclose(a.E_sin,b.E_sin)
    np.testing.assert_allclose(a.E_cos,b.E_cos)
    assert not np.allclose(a.S_mean,b.S_mean)


def test_through_wall_profile_and_von_mises(geom,steel):
    loads=TubeLoads(p_i=HarmonicLoad(32.5),force=HarmonicLoad(15904.31),
                    torque=HarmonicLoad(50000))
    result=radial_profile(geom,steel,loads,phase_rad=1.6,n_points=41)
    assert result.radius_mm.shape == (41,)
    assert result.stress_MPa.shape == (41,3,3)
    assert result.strain.shape == (41,3,3)
    assert result.von_mises_MPa.shape == (41,)
    assert result.radius_mm[0] == geom.r_i
    assert result.radius_mm[-1] == geom.r_o
    for r,S,vm in zip(result.radius_mm,result.stress_MPa,result.von_mises_MPa):
        expected=stress_tensor(geom,float(r),p_i=32.5,force=15904.31,torque=50000)
        np.testing.assert_allclose(S,expected)
        sr,st,sz=S.diagonal()
        tau=S[1,2]
        assert vm == pytest.approx(math.sqrt(sr*sr+st*st+sz*sz-sr*st-st*sz-sz*sr+3*tau*tau))


def test_thin_wall_limit_from_lame():
    # For t/r_i -> 0, pressure hoop and capped axial stress approach p*r_i/t
    # and p*r_i/(2t). Radial stress is O(p) rather than O(p*r_i/t).
    g=TubeGeometry(20.,0.001)
    S=stress_tensor(g,"mean",p_i=4)
    assert S[1,1] == pytest.approx(4*g.r_i/g.thickness, rel=1e-4)
    assert S[2,2] == pytest.approx(4*g.r_i/(2*g.thickness), rel=1e-4)


def test_harmonic_phase_and_minimum_pressure_validation(geom):
    load=HarmonicLoad.from_amplitude_phase(23,8,math.pi/3)
    assert load.amplitude == pytest.approx(8)
    assert load.extrema == pytest.approx((15,31))
    for t in np.linspace(0,2*math.pi,13):
        assert load.at(t) == pytest.approx(23+8*math.sin(t+math.pi/3))


def test_thick_to_fs_ext_integration(geom,steel):
    loads=TubeLoads(
        p_i=HarmonicLoad.from_amplitude_phase(26,6,math.pi/4),
        force=HarmonicLoad.from_amplitude_phase(0,12000,0),
        torque=HarmonicLoad.from_amplitude_phase(0,42000,math.pi/2),
    )
    history=harmonic_history(geom,steel,loads,"inner")
    n=np.array([[0,0,1],[0,1,0],[0.33,0.5,0.8]])
    vals=[evaluate_fs_ext(history,n,algorithm=a) for a in ('longest_chord','mcc','mce')]
    np.testing.assert_allclose(vals[0], vals[1],rtol=1e-12,atol=1e-14)
    assert np.all(vals[2]>=vals[0]-1e-14)
    assert vals[2][-1]>vals[0][-1]


def test_invalid_inputs_and_boundary_modes(geom,steel):
    for x,y in ((0,1),(12,0),(-12,1),(12,float('nan'))):
        with pytest.raises(ValueError):
            TubeGeometry(x,y)
    for E,nu in ((0,.3),(210000,.5),(210000,-1),(210000,float('nan'))):
        with pytest.raises(ValueError):
            ElasticMaterial(E,nu)
    for location in (-1,6-1e-6,7.5+1e-6,"incorrect"):
        with pytest.raises(ValueError):
            stress_tensor(geom,location)
    with pytest.raises(ValueError,match='end_condition'):
        stress_tensor(geom,'inner',end_condition='open')
    with pytest.raises(TypeError):
        harmonic_history(geom,steel,TubeLoads(force=5))
    with pytest.raises(ValueError):
        radial_profile(geom,steel,TubeLoads(),n_points=1)
    with pytest.raises(ValueError):
        HarmonicLoad.from_amplitude_phase(0,-1)
    with pytest.raises(ValueError):
        HarmonicLoad(float('inf'))
    with pytest.raises(ValueError):
        stress_tensor(geom,'inner',p_i=float('nan'))


def test_combined_loads_match_original_main_v3_reference(geom):
    """Regression values from MAIN_v3.py, model='thick', p=32.5,
    F=15904.31 N, torque=43000 N mm, evaluated at all three radii.
    """
    reference = {
        'inner': [-32.5, 148.05555555555554, 307.777733626258, 87.92420021858797],
        'mean': [-13.552812071330585, 129.10836762688615, 307.777733626258, 98.91472524591147],
        'outer': [0.0, 115.55555555555556, 307.777733626258, 109.90525027323496],
    }
    for surface, expected in reference.items():
        stress = stress_tensor(geom, surface, p_i=32.5, force=15904.31, torque=43000)
        assert [stress[0,0], stress[1,1], stress[2,2], stress[1,2]] == pytest.approx(expected)


def test_phase2_history_feeds_both_seancp_and_full_scan(geom, steel):
    from seancp import seancp_fs_ext, full_plane_scan
    loads = TubeLoads(
        p_i=HarmonicLoad.from_amplitude_phase(26,6,math.pi/4),
        force=HarmonicLoad.from_amplitude_phase(0,12000,0),
        torque=HarmonicLoad.from_amplitude_phase(0,42000,math.pi/2),
    )
    history = harmonic_history(geom,steel,loads,'inner')
    for algorithm in ('longest_chord','mcc','mce'):
        semi = seancp_fs_ext(history, algorithm=algorithm, sigma_y=900,
                             samples_per_path=75)
        full, normal = full_plane_scan(history, algorithm=algorithm,
                                       sigma_y=900, n_theta=45, n_phi=91)
        assert len(semi.paths) == 3
        assert semi.value > 0
        assert full >= semi.value - 1e-12
        assert np.linalg.norm(normal) == pytest.approx(1)
