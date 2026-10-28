# Real machines, specimen concepts and reproducible APDL/FS_ext workflow

**Scope.** This is a research implementation, not a proof that a particular
machine can reproduce a desired pointwise stress state. The APDL files are
intended for verification with an installed, licensed Mechanical APDL solver.
No Ansys solver was available in the authoring environment, so **no FE numerical
results are claimed here**. Python tests verify coefficient generation,
data consistency and reconstruction only.

## 1. Machines and evidence

| Source | What is actually documented | Implications |
| --- | --- | --- |
| STEP Lab (Italy), multi-actuator systems | Configurations up to eight actuators, synchronous operation; catalogue includes multiaxial, axial-torsion and four-actuator planar cruciforms. The planar model controls two in-plane axes, not an independent in-plane shear channel. | Commercial enquiry for a **custom six-actuator 3D fatigue system** or a **three-input planar system**. The manufacturer must verify fixture, simultaneous phase control and rated cyclic force in each channel. |
| UCLM six-actuator machine, Serna Moreno et al. (2022) | Orthogonal electromechanical tensile/compressive actuators, six total, up to +/-50 kN **per actuator** and +/-50 mm stroke. Specimen and fixtures optimised through FEM. | Most directly documented physical architecture for **2213** (triaxial normal tension/compression). Crucial caveat: the paper reports greater stresses/failure risk in arms, not uniform critical-region failure. |
| Calloch and Marquis/ASTREE (published research) | Six independently controlled servohydraulic actuators; 250 kN vertical, 100 kN horizontal in one account; triaxial tension-compression testing. | Historical real test-rig precedent. Availability and current operating status not established. |
| Hitachi three-direction planar fatigue research rig (2014; Inoue et al. 2016) | Three controlled actuators at 0,45,90 degrees; observed reproduction of arbitrary in-plane surface stresses. Linear guides reduce gauge-centre motion. | Mechanically suitable *architecture* for **1739**, with FEM calibration. Published prototype/research rig, **not verified as commercially obtainable**. A dedicated rig CAD/specimen model is needed to reproduce its detailed actuation. |
| MTS 833 | Three-axis excitation of elastomeric components, +/-25 kN per axis, >=80 Hz minimum stated operating upper frequency, phases independently controlled. | Three force directions are **not** three independent material-point normal stresses. Useful component-testing system, not a qualified direct test of 2213. |
| Cao, Liu and Song (2006), uploaded article, pp. 2-3 | Tapered concrete prism, 1000 kN axial cyclic machine, **two 250 kN manually controlled lateral screw jacks**. Lateral pressure held approximately constant; longitudinal load sinusoidal. | Valid precedent for mixed axial and biaxial confinement; **not** independent three-axis harmonic tension-compression. Materials and load magnitudes differ from current metallic reference. |

Primary URLs: https://step-lab.com/multi-actuator-systems/ ;
https://step-lab.com/multi-actuator-systems/planar-cruciform-biaxial-systems/ ;
https://doi.org/10.3390/polym14040837 ;
https://www.hitachi.com/New/cnews/month/2014/05/140515.html ;
https://doi.org/10.3221/IGF-ESIS.38.35 ;
https://www.mts.com/-/media/ground_vehicles/pdfs/brochures/100-015-996b_Triaxial833.pdf?as=1 .

## 2. Three FE boundary models and what they establish

1. **`ideal_cube.mac`.** Exactly prescribed uniform tensor surface tractions
   `t(n,t)=sigma(t)n` on six ideal faces of a 5 mm cube. Mapped SOLID185
   hexahedra. The three independently solved Fourier coefficient fields
   must reproduce the target uniform stress/strain in the gauge. This is a
   numerical reference **not an experimental fixture**.
2. **`six_arm.mac`.** A simplified six-arm cruciform specimen for independent
   x, y, z tension-compression end loads, intended for the three-diagonal
   case 2213. Seven overlapping BLOCK volumes glued and meshed with
   quadratic SOLID187 tetrahedra; forces on six end faces. End load
   nominally starts from stress times arm end area, but **does not produce
   prescribed target stress in the gauge without FE calibration**.
3. **`planar_traction_cruciform.mac`.** A four-arm, thin, 3D cruciform for
   case 1739 with independently controlled normal AND tangential
   end-face loads (normal plus shear on each x/y face). This is a
   **mathematical multiaxial end-traction fixture concept**. In particular,
   the standard STEP Lab four-axial-actuator frame does NOT have the
   two extra independently controlled shear forces represented here.
   The hardware control layout must be redesigned or replaced by a
   documented 0/45/90 architecture before a real experiment.

These simplified specimens have abrupt arm transitions, no engineered
fillet, grips, contact, misalignment or geometry optimisation. They are
**screening models**, not fatigue-ready machined specimen drawings.

## 3. APDL execution

The APDL macros **must run from the `apdl/` working directory**, so
`/INPUT,case_2213,mac` and `*USE,export_critical` can resolve their files.
Remove earlier `fe_stress.csv`, `fe_strain.csv` and `fe_coordinates.csv`
before every run, since the exporter uses **APPEND**.

Example, using your installed solver version/batch syntax:

```bash
cd apdl
# Mapdl command depends on the installation, replace <mapdl> accordingly
<mapdl> -b -i ideal_cube.mac -o cube.log
```

To run another cube case, change `CASEID=2213` at the beginning of
`ideal_cube.mac`. The six-arm and planar models are prepared specifically
for 2213 and 1739 respectively.

**Three load steps are NOT three sampled time steps.** They are
mean, sine and cosine **independent static analyses**, for linear-elastic,
quasi-static harmonic superposition. Frequencies, damping and inertia are
not included; for experimental frequency ranges where dynamic effects
matter, use a time-domain or harmonic dynamic model and compare the
reconstructed measured histories separately.

The exporter writes one numeric row per critical-region element per
coefficient to each file:

- `fe_stress.csv`: `coefficient_index, element_id, Sxx,Syy,Szz,Sxy,Syz,Sxz` (MPa).
- `fe_strain.csv`: corresponding `EPELxx,EPELyy,EPELzz,EPELxy,EPELyz,EPELxz`.
- `fe_coordinates.csv`: `coefficient_index, element_id, x,y,z` (mm).

**Important:** MAPDL `EPEL` shear is **engineering shear strain**. The
Python pipeline divides its off-diagonal components by 2 to reconstruct
a symmetric tensorial strain. It uses the FE strain rather than silently
recomputing it from the FE stress. A separate Hooke-law residual is
reported for comparison.

### 3A. Optional inverse actuator calibration

The realistic specimen models are strongly coupled. First solve the
unit-traction macros:

```bash
<mapdl> -b -i six_arm_calibration.mac -o calibration.log
# in project repo root (point --basis to the calibration run result)
python examples/calibrate_apdl_controls.py --geometry six_arm \
  --basis path/to/calibration/fe_stress.csv \
  --output apdl/case_2213_calibrated.mac
```

For planar, replace `six_arm` with `planar` and run
`planar_calibration.mac` for the basis. The script calculates a numerical
**3x3 influence matrix** from actual FE gauge stresses, solves for the
three desired independent harmonic actuation signals, and writes new
command coefficients. To use these, **change the `/INPUT` line** in the
corresponding real geometry macro and rerun a fresh set of three static
Fourier coefficient problems.

Calibration matches **mean gauge component stresses**, not the complete
stress tensor field or its spatial variation. It does not prevent fatigue
cracks forming in the arms. Gauge validation must use the **realised
full stress and strain tensors**.

### 3B. Postprocessing

```bash
python -m pip install -e '.[test,plot]'
python examples/postprocess_apdl_fs_ext.py \
  --fe-dir path/to/apdl/run_directory \
  --output results/real_machines/2213_sixarm \
  --case-id 2213 --algorithm mce --max-elements 0
```

For each exported element the pipeline evaluates the **complete FS_ext
parameter**, with identical normal-stress maxima and the same shear
amplitude rule on both sides. It independently optimises the full
critical-plane surface, optimises the three SeAn-CP paths, and records:

- FE-derived true local tensor mean/sine/cosine, from matched element IDs;
- per-element full-plane and SeAn-CP FS_ext and signed percentage error;
- maximum-parameter element among the processed region;
- difference between its FE harmonic stress tensors and Phase 4 targets;
- linear-elastic stress/strain consistency diagnostic;
- a 3D azimuth/polar/FS_ext surface with all three paths and both maxima,
  plus a complementary 2D map.

By default `--max-elements 16` controls runtime, so this is only a
**subset** search. Use `--max-elements 0` and refine the mesh before
publishing numerical results. The independent scan remains a numerical
reference, **not a mathematically proven global maximum**.

## 4. Requested mechanical verification before experimental conclusions

- Run the **ideal cube** with 6, 10, and 16 divisions. Check stress,
  elastic strain and the FS_ext errors against Phase 4 values of 2213
  (-37.39%, for LC/MCC/MCE) and 1739 (-12.33%, MCE).
- For the six-arm geometry, halve the mesh size in the central volume
  and verify central stress and strain changes. Measure centre field
  coefficient variation and identify maximum outside the gauge, since
  fracture in an arm would invalidate a centre-gauge fatigue experiment.
- For planar case 1739, verify `Szz`, `Sxz`, `Syz` are negligible in the
  centre, and that the shear stress controls are truly independent.
- Check dynamic machine capability, reversing tension/compression,
  actuator simultaneity and phase accuracy. Published force ratings do
  not prove cyclic load range at the required frequency.
- Replace the artificial pinning by properly modelled grips/reactions
  (or weak-spring restraints) in the final FE models. Verify force
  balance and negligible support reactions before drawing conclusions.
- Only then evaluate whether the **actual FE-derived stress/strain
  histories** preserve a sufficiently large FS_ext discrepancy, and
  obtain experimentally calibrated strains for a physical test.

## 5. What is validated in the provided package

- Source tensor transcription and APDL load-macro generation.
- APDL command presence and interface layout inspection.
- Synthetic exact-tensor file roundtrip, including tensorial shear.
- Sensitivity-matrix inversion for coupled actuation.
- Existing SeAn-CP model Python tests.

**Not validated:** APDL meshing, convergence or running any of the provided
macros, actual fixture availability, load-channel qualification, yield
or buckling limits, or a measured fatigue-test result. These require the
appropriate MAPDL executable, rig specifications and material data.
