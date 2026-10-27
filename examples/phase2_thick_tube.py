"""Phase 2: thick tube with independently phase-shifted harmonic actuators.

Run from the repository root with `python examples/phase2_thick_tube.py`.
"""

from math import pi

import numpy as np

from seancp import (TubeGeometry, ElasticMaterial, HarmonicLoad, TubeLoads,
                    harmonic_history, radial_profile, evaluate_fs_ext)

geometry = TubeGeometry(D_int=12.0, thickness=1.5)
steel = ElasticMaterial(E=210000.0, nu=0.30)

# Gauge pressure fixed at 32.5 MPa; force fully reversed, torsion at 90 degrees.
# Units: MPa, N, and N mm (1 N m = 1000 N mm).
F_amp = 15904.31
lambda_target = 0.58
M_amp = lambda_target * (F_amp / geometry.area) * geometry.J / geometry.r_o
loads = TubeLoads(
    p_i=HarmonicLoad(mean=32.5),
    force=HarmonicLoad.from_amplitude_phase(0.0, F_amp, 0.0),
    torque=HarmonicLoad.from_amplitude_phase(0.0, M_amp, pi/2),
)

print(f"Tube ri={geometry.r_i:.3f} mm, ro={geometry.r_o:.3f} mm")
print(f"Area={geometry.area:.5f} mm2, J={geometry.J:.5f} mm4")
print(f"Force amplitude={F_amp:.2f} N; torque amplitude={M_amp/1000:.3f} N m")
print("Tensor order: (r, theta, z). Positive normal stresses are tensile.")

for position in ("inner", "outer"):
    history = harmonic_history(geometry, steel, loads, position, end_condition="closed")
    print(f"\n{position.upper()} SURFACE")
    for phase in (0.0, pi/2, pi):
        stress, strain = history.at(phase)
        print(f"phase {phase:5.2f} rad: sigma_r={stress[0,0]:9.3f}, "
              f"sigma_theta={stress[1,1]:9.3f}, sigma_z={stress[2,2]:9.3f}, "
              f"tau_theta_z={stress[1,2]:9.3f} MPa")
    # One plane check; full-plane exploration is part of Phase 3.
    n = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    for algorithm in ("longest_chord", "mcc", "mce"):
        print(f"{algorithm:14s} selected-plane FS_ext values: "
              f"{evaluate_fs_ext(history, n, sigma_y=900., algorithm=algorithm)}")

profile = radial_profile(geometry, steel, loads, phase_rad=pi/2, n_points=5)
print("\nThrough-wall stress states at phase pi/2")
for r, s, vm in zip(profile.radius_mm, profile.stress_MPa, profile.von_mises_MPa):
    print(f"r={r:.4f} mm: sigma_r={s[0,0]:9.3f} MPa, "
          f"sigma_theta={s[1,1]:9.3f} MPa, sigma_z={s[2,2]:9.3f} MPa, "
          f"tau_theta_z={s[1,2]:9.3f} MPa, VM={vm:9.3f} MPa")
