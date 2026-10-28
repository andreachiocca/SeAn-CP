# Phase 5. Mechanical realizability of the critical SeAn-CP loading histories

## Aim and scope

This study examines the **four selected cases from Phase 4**, without changing
any target tensor or its numerical FS_ext result. We separate three questions.

1. Is the target stress history admissible as an elastic boundary-value problem?
2. Can it be represented by a reduced set of ideal experimental load controls?
3. Can an existing physical specimen and apparatus reproduce it with adequate
   stress homogeneity, fatigue performance and phase accuracy?

**Only questions 1 and 2 are addressed quantitatively.** The third requires
specimen design, finite-element verification, material identification and
instrument-specific validation. A local stress tensor does not uniquely define
specimen geometry or actuator forces.

All histories remain **single-frequency harmonic**, and FS_ext remains the
**only** fatigue parameter. The three SeAn-CP searching paths and their shear
amplitude definitions are not modified. The original Phase 4 stresses are
read from `results/phase4_reference/selected_cases.json`, not regenerated.

## Reference local mechanics

At the material point,

\[
\sigma(t)=\sigma_m+\sigma_s\sin\omega t+\sigma_c\cos\omega t,
\qquad
\varepsilon_j=\frac{1+\nu}{E}\sigma_j-
\frac{\nu}{E}\operatorname{tr}(\sigma_j)I.
\]

We retain the Phase 4 illustrative material settings, `E=210000 MPa`,
`nu=0.3`, `Sy=600 MPa`, `k_FS=0.4`. They are **not calibrated material data**.

For a hypothetical cube, a constant stress field satisfies static equilibrium
and produces the exact prescribed surface traction

\[
\boxed{\mathbf t(\mathbf n,t)=\boldsymbol\sigma(t)\mathbf n.}
\]

Symmetry of the stress tensors ensures balance of the opposite-face forces
and zero net moment. We verify both conditions for **all three harmonic
coefficients**. The surface traction is generally a *vector*, not a simple
normal pressure. A block with perfectly distributed full-face tractions is an
**ideal quasi-static elasticity problem**. It is not a practical fatigue
specimen design. At finite frequency, inertia and wave propagation may prevent
an exactly homogeneous dynamic stress field.

The illustrative load scale is a **5 mm cube** with one face area of `25 mm²`.
For example, a stress component of `400 MPa` requires a component face force of
`10 kN`. These values are analytical surface resultants, not actuator loads
for an actual six-arm cruciform specimen. Different gripping areas, contact
tractions and arms can change the necessary actuator forces substantially.

## Numerical comparison of reduced stress spaces

All three harmonic tensors must be represented in one **fixed specimen basis**.
The closest tensor history within each model is selected numerically by
minimising the Frobenius norm of the differences over all three coefficients.
The reported distance is

\[
 d=100\,\frac{\left\|(\sigma_m,\sigma_s,\sigma_c)_\mathrm{fit}
 -(\sigma_m,\sigma_s,\sigma_c)_\mathrm{target}\right\|_F}
 {\left\|(\sigma_m,\sigma_s,\sigma_c)_\mathrm{target}\right\|_F}\,\%.
\]

An exact fit has `d=0`. Small `d` does **not** imply homogeneous gauge
stresses or easy implementation in a machine. Nonconvex orientation searches
use three independent optimisation seeds, but are still numerical fits,
not mathematical proofs of global minimisation.

| Original case | Target FS_ext error | Alternative loading space | Coefficient distance | FS_ext error after fitting |
| --- | ---: | --- | ---: | ---: |
| **2213** | **−37.386%** | Three independent orthogonal normal loads | **0%** | **−37.386%** |
| **1739** | **−12.330%** | Planar biaxial normal and independently controlled shear loads | **0%** | **−12.330%** |
| **6920** | **−12.126%** | Three normal loads on optimised fixed orthogonal axes | **16.31%** | **−7.57%** |
| **6920** | **−12.126%** | Thick tube with algebraically signed inner and outer pressures | **11.15%** | **−10.17%** |
| **7348** | **−11.889%** | Three normal loads on optimised fixed orthogonal axes | **33.14%** | **~0%** |

The `tube_signed` search is a *relaxed mathematical constraint*. It allows
negative gauge pressures and unconstrained axial force and torque, so its
results must **not** be used as a valid experimental loading proposal.

For further comparison, a closed-end `D_int=12 mm`, `t=1.5 mm` tube with
`p_o=0`, `p_i(t) >= 0`, arbitrary axial force and torque, and an optimised fixed
orientation was fitted to each case. The pressure positivity condition is
applied exactly using a Lorentz-cone projection of the harmonic coefficients,
`p_mean >= hypot(p_sin,p_cos)`. Fits are made to the *outer-surface tensor*.
Their original-case errors are not preserved, and the wall-wide comparison
uses 11 radial points. Thus, the familiar positive-internal-pressure tube does
not reproduce these four target histories. It may produce other cases with
smaller FS_ext discrepancies, as already found in Phase 3.

## Case 2213. Non-proportional three-normal-stress history

This is the **strongest experimental candidate**. The three stress tensors are
diagonal in the same basis, so no shear traction needs to be generated by the
ideal boundary model. Independent normal-stress controls are sufficient.

| Stress axis | Mean (MPa) | Amplitude (MPa) | Phase relative to sine (deg) | Minimum (MPa) | Maximum (MPa) |
| --- | ---: | ---: | ---: | ---: | ---: |
| x | −40.633 | 255.889 | −4.8 | −296.521 | 215.256 |
| y | −107.455 | 308.047 | 14.2 | −415.501 | 200.592 |
| z | 34.387 | 250.110 | 28.8 | −215.723 | 284.497 |

Use `sigma_i(t)=mean_i+amplitude_i*sin(omega*t+phase_i)` with the stresses above.
Each axis **changes sign** during the cycle, so compression-only triaxial
cells cannot reproduce the history. Independent servo-controlled
**tension–compression along all three axes** is required.

For an **ideal 5 mm cube**, the largest absolute face component force is
`10.388 kN`, along the y axis. These values are below the individual
`±50 kN` actuator rating of one published six-actuator research facility.
**This is not sufficient to declare that facility capable of the test**, because
the gauge cross-section, grips, cyclic frequency and achievable stress
uniformity are unknown. The published triaxial specimen research also reports
stress concentrations outside the gauge section, which must be avoided in
fatigue testing.

**Why the error is large:** the target stress tensors share principal axes.
The three original largest-Mohr-circle searches overlook a different
principal-plane family. The unrestricted plane maximum is in the x–z plane,
while the restricted search returns a plane in the y–z plane. The deficit is
**−37.386% for longest chord/MCC as well as MCE**, so it cannot be attributed
solely to the MCE amplitude definition.

## Case 1739. Exact plane stress with in-plane shear

All three target tensors have **zero z row and column**. They are therefore
exactly representable by two in-plane normal stresses and one independently
controlled in-plane shear stress. The complete harmonic control coefficients
are given in `results/phase5_reference/reduced_control_waveforms.csv`.

A standard cruciform with only two axial load signals does **not** generally
provide independent control of `sigma_xx`, `sigma_yy`, and `tau_xy`. A dedicated
planar combined biaxial/shear fixture, or an experimentally calibrated
multi-actuator arrangement, would be needed. The maximum ideal force component
on a 5 mm cube face is `11.780 kN`; the maximum shear component force is
`3.510 kN`.

This case cannot be reproduced exactly at the **outer surface of a tube with
nonnegative internal pressure and zero external pressure**. The stress-free
radial direction must be z. Even after optimising the fixed orientation of
the hoop direction, the minimum hoop stress over the cycle is **−143.77 MPa**.
For the reference geometry, that would require at least **−40.44 MPa gauge
internal pressure** at some point, which is not an achievable liquid pressure.
A signed-pressure mathematical reconstruction is therefore misleading here.

## Case 6920. Three-dimensional rotating principal tensors

The original history cannot be represented exactly as plane stress, as three
normal components on fixed axes, or in the signed-pressure tube subspace.
All three tensor coefficients involve non-commuting shear/normal components.

A fixed-axis three-normal fit changes the three stress coefficients by
**16.31% in Frobenius norm** and retains a **−7.57% FS_ext error**. This is a
**new loading history**, not the original case, and is a useful lower-complexity
follow-up. The actual rotated control axes and three normal-stress waveforms
are saved in `projected_histories.json` and `reduced_control_waveforms.csv`.
The associated 3D FS_ext plot includes the original three SeAn-CP searching
paths and both actual damage maxima.

An even closer **11.15% tensor-distance** fit with a **−10.17% FS_ext error**
exists within the *relaxed*, signed-pressure thick-tube stress subspace.
This does not establish a feasible fluid-pressure test. For strictly positive
internal pressure at the outer wall, the error is essentially lost.

## Case 7348. Fully non-coaxial harmonic history

Neither three-normal-stress control, plane stress, nor a four-dimensional
signed-pressure tube subspace retains a useful SeAn-CP error in the closest
numerical fit. An ideal six-independent-component stress-control fixture can
reproduce the homogeneous local stress history, but this requires normal
**and tangential tractions on multiple orthogonal faces**, with appropriate
synchronisation and gauge stress uniformity. A normal-only true triaxial cell
is insufficient. We have not identified a tested standard fatigue fixture
that guarantees this complete six-component history.

This is a **conceptual mechanical possibility**, not an established
experimental solution. It should be considered the lowest-priority case
unless an appropriate specialist fixture is available.

## Research evidence and boundaries of applicability

- Calloch and Marquis (1999), *Triaxial tension–compression tests for multiaxial
  cyclic plasticity*, International Journal of Plasticity,
  https://doi.org/10.1016/S0749-6419(99)00005-4. Experimental triaxial
  tension–compression cycling exists, but the authors explicitly use 3D finite
  elements because the local stress field is not homogeneous.
- Serna Moreno, Horta Muñoz and Ruiz Gracia (2022), *Design of Triaxial Tests
  with Polymer Matrix Composites*, Polymers 14, 837,
  https://doi.org/10.3390/polym14040837. A six-actuator triaxial
  tension–compression facility and detailed FEM-based specimen design are
  documented, including the need to control stress concentrations.
- Braunschweig Multiaxial Panel Test rig,
  https://www.tu-braunschweig.de/en/ifl/experimental-resources/testing-facilities/mpt.
  Documents combined normal and shear loading in panels, but does not prove
  that arbitrary homogeneous 3D stress histories are achievable at a point.
- GDS True Triaxial Apparatus,
  https://www.gdsinstruments.com/gds-products/gds-true-triaxial-apparatus.
  Independently controls three **normal** stresses on soft-ground samples,
  without providing the general shear-face control required by case 7348.

## Recommended engineering follow-up

1. **Prioritise 2213.** Develop a six-arm three-normal-direction specimen
   and use a 3D elastic FE model to obtain a sufficiently uniform centre
   region while avoiding fatigue cracking in the arms and grips. Drive the
   three independently phased signals in the waveform CSV. Confirm tension
   and compression reversal in all three axes.
2. **Consider 1739 next.** Use a dedicated planar tension–compression plus
   shear fixture and reconstruct the central three stress components from
   FE calibration and synchronized strain measurements. A simple cruciform
   without independent shear control is inadequate.
3. **Evaluate the rotated 6920 alternative.** Apply three normal load
   histories on the specified fixed axes; verify that an actual 3D specimen
   produces the retained `~7.57%` FS_ext discrepancy before fabrication.
4. **Do not prioritise 7348** until a demonstrably six-component-capable
   fixture is available, or until new constrained optimisation identifies
   another experimentally controllable history with comparable error.

Before any fatigue experiment, obtain actual material cyclic properties,
residual-stress and plasticity information, feasible load cells and their
bandwidth, appropriate strain measurement, fixture tolerances and fatigue
life requirements. Confirm centre-region tensor trajectories with FEM,
including geometric non-uniformity and end effects. Validate predicted
`FS_ext` with the **realised** tensor history, not merely with commanded
actuator signals. The idealised 5 mm cube is a bookkeeping device only.

## Reproduction and outputs

```bash
python -m pip install -e '.[test,plot]'
python -m pytest -q
python examples/phase5_feasibility.py --out results/phase5
```

The source program produces `feasibility_cases.csv`, `projection_results.csv`,
`reduced_control_waveforms.csv`, `ideal_cube_face_tractions.csv`,
`projected_histories.json`, a comparative error-retention chart, an ideal-face
force chart and four 3D FS_ext landscape figures. The figures retain the full
plane maximum, the SeAn-CP estimate, and the three searching paths, all at
correct orientations and values. The repository includes compact frozen
reference tables; the separate archive includes full 3D figures.
