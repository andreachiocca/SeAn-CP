# Phase 2. Elastic thick-walled tube under independently controlled harmonic loads

This phase extends the specimen model supplied in `MAIN_v3.py`, keeping its
coordinates and units. It supplies the full time-dependent tensors needed by
`FS_ext`, without changing the three SeAn-CP searching paths.

## Geometry and assumptions

The circular tube has inner radius `a = D_int/2`, outer radius `b = a+t`,
and a uniform wall thickness `t`. Stresses are evaluated at a selected radius
`a <= r <= b` in the orthonormal cylindrical basis `(r, theta, z)`. Tension
and positive torque follow the sign convention in the original script.
Pressure is a gauge pressure with the sign convention `sigma_rr(a)=-p_i` and
`sigma_rr(b)=-p_o`. Length is in mm, force in N, torque in N mm, pressure,
Young's modulus and stress in MPa. Strains are dimensionless and tensorial.

The solution assumes a straight, circular, sufficiently long, homogeneous,
isotropic, **linearly elastic** tube, under axisymmetric pressure, pure axial
force, and Saint-Venant torsion. Small deformations and superposition are
assumed. It excludes end effects, pressure-port stress concentrations,
fixture-induced bending, buckling, yielding, residual stress, plasticity,
and changes in geometry. It is the local gauge-section solution, not an
experimental rig model.

For the reference specimen in `MAIN_v3.py`, `D_int=12 mm` and `t=1.5 mm`.
The wall ratio is `t/a=0.25`, so the original thin-wall approximation should
not be the reference for the fatigue calculation.

## Stress equations

For pressure values `p_i` and `p_o` (both may vary harmonically), define

```
A = (p_i*a^2 - p_o*b^2) / (b^2-a^2)
B = a^2*b^2*(p_i-p_o) / (b^2-a^2)
area = pi*(b^2-a^2)
J = pi*(b^4-a^4)/2

sigma_rr = A - B/r^2
sigma_tt = A + B/r^2
sigma_zz = force/area + (A if end_condition='closed' else 0)
tau_theta_z = torque*r/J

S = [[sigma_rr,       0,             0],
     [       0, sigma_tt, tau_theta_z],
     [       0, tau_theta_z, sigma_zz]]
```

`force` is the **additional axial force carried by the gauge section**, on top
of pressure-generated thrust if the end condition is `closed`. For closed
ends, the net pressure thrust carried by the wall is
`pi*(p_i*a^2-p_o*b^2)=A*area`. It must **not** be added again as part of
`force`. Conversely, if the load readout includes pressure thrust, convert
to the additional wall force before calling the model, with the correct rig
load-path and load-cell sign convention.

The alternative `end_condition='pressure_compensated'` represents a special
fixture in which the pressure-end thrust is reacted outside the gauge
section. The radial and circumferential Lamé stresses remain unchanged, while
`pressure` contributes no axial membrane stress. It does **not** describe an
ordinary open pressure vessel with pressure escaping: the pressure must
still be maintained by a sealing/fixture mechanism. Neither branch captures
local seal or grip effects. The actual experimental fixture must determine
which condition is appropriate.

## Elastic strains

With Young's modulus `E`, Poisson's ratio `nu`, and shear modulus
`G=E/[2*(1+nu)]`, tensorial strain is given by

```
E_tensor = ((1+nu)*S - nu*trace(S)*I) / E

epsilon_rr = (sigma_rr - nu*(sigma_tt+sigma_zz))/E
epsilon_tt = (sigma_tt - nu*(sigma_rr+sigma_zz))/E
epsilon_zz = (sigma_zz - nu*(sigma_rr+sigma_tt))/E
epsilon_theta_z = tau_theta_z/(2*G)
```

The factor two between tensorial and engineering shear is handled by the
existing `FS_ext` evaluator, not by the specimen module.

## Harmonic load histories

Each independently controlled quantity is represented exactly by
`q(t)=q_mean+q_sin*sin(omega*t)+q_cos*cos(omega*t)`, with an arbitrary phase
relative to the common angular coordinate. `HarmonicLoad.from_amplitude_phase`
implements `q_mean+q_amp*sin(omega*t+phase_rad)`.

As the stress solution is linear in the applied loads, each of the three
stress-tensor coefficients is formed from the corresponding load coefficients
using the same thick-wall equations. Hooke's law gives the three strain
coefficients. No temporal sampling or compression into two end states occurs.
Thus tension, torsion, and pressure can all be non-proportional and phase
shifted while using the previously implemented harmonic amplitude definitions.

The generated `seancp.HarmonicHistory` is directly usable by
`evaluate_fs_ext`, `seancp_fs_ext`, and `full_plane_scan`.

## Reproduction

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/phase2_thick_tube.py
```

Example:

```python
from math import pi
from seancp import (TubeGeometry, ElasticMaterial, TubeLoads,
                    HarmonicLoad, harmonic_history, seancp_fs_ext)

g = TubeGeometry(D_int=12, thickness=1.5)
m = ElasticMaterial(E=210000, nu=0.30)
loads = TubeLoads(
    p_i=HarmonicLoad(mean=32.5),
    force=HarmonicLoad.from_amplitude_phase(0, 15904.31, 0),
    torque=HarmonicLoad.from_amplitude_phase(0, 50000, pi/2),
)
history = harmonic_history(g, m, loads, "outer", end_condition="closed")
result = seancp_fs_ext(history, algorithm="longest_chord", sigma_y=900)
```

The example uses `sigma_y=900 MPa` only as an illustrative FS constant,
**not** as a measured yield strength of the specimen material. A valid
fatigue comparison needs actual material properties and an elastic admissibility
check over the cycle and wall. No load or pressure capacities are asserted.

## Verification and future steps

Numerical tests cover Lamé pressure boundary conditions at both surfaces,
closed-end pressure thrust, pressure compensation, radial equilibrium,
axial resultant, torque resultant, tensorial Hooke law, arbitrary phase
reconstruction, reference results from the supplied script, and integration
with the Phase 1 harmonic `FS_ext` calculations.

For the script's reference pressure of 32.5 MPa and zero external pressure:

| Surface | sigma_rr [MPa] | sigma_tt [MPa] | pressure sigma_zz [MPa] |
| --- | ---: | ---: | ---: |
| inner (6 mm) | -32.500000 | 148.055556 | 57.777778 |
| outer (7.5 mm) | 0.000000 | 115.555556 | 57.777778 |

Phase 3 will explore many harmonic loading histories and compare restricted
SeAn-CP with full-plane `FS_ext` scanning across both plane angles and the
wall thickness, reporting the three searching paths. The exact boundary
condition and the feasibility of a test rig remain to be verified in Phase 4.
