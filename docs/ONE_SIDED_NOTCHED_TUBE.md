# Single-sided notched tube under pressure, axial force and torque

## Research question

Can the available axial-force/torsion/internal-pressure machine produce a
**locally non-proportional stress/strain tensor** for which SeAn-CP significantly
underestimates the fully searched FS_ext damage parameter? This is a useful,
testable extension of the smooth thick-wall tube in Phase 3. A one-sided notch
removes circumferential symmetry without requiring another actuator. It does
**not** add an independently adjustable fourth stress control.

The present package implements an **unexecuted three-dimensional APDL model**
plus a reproducible Python search that works on actual **unit-load FE results**.
We do not claim any nonzero notch-induced FS_ext error until the ANSYS model
has been executed and mesh independence checked.

## Mechanical interpretation

A smooth tube in the local cylindrical (r,theta,z) frame has

```text
[sigma_rr       0       0     ]
[0          sigma_tt  tau_tz  ]
[0          tau_tz    sigma_zz]
```

For a single outer-surface notch, this restricted form no longer holds in
its neighbourhood. Curvature and a three-dimensional local load path may
create previously absent tensor components, rotate principal axes during
non-proportional loading and alter the three eigenvector-based SeAn-CP search
families. The actual tensor must come from 3D FE, not from a notch factor
multiplied into the smooth-tube formula.

The notch **surface itself is traction-free**, so the local stress vector
normal to it is zero. Exact three-dimensional triaxiality and a maximum
FS_ext discrepancy may occur **subsurface**. Extract an element cloud, not a
single point on a face. Conversely, a severe notch may create a large stress
concentration but a *small* SeAn-CP error, or may initiate failure outside
the desired gauge. Stress concentration and methodological error are different
objectives.

Since the same three imposed harmonic controls are used in linear elasticity,
**only three stress influence tensors** exist at each FE point. A geometry
change alters their orientations and relative magnitudes, but cannot produce
six arbitrarily independent stress components from only three controls.

## Baseline geometry (research screening, not a manufacture drawing)

- Tube bore diameter 12 mm, outside diameter 15 mm, gauge/model length 48 mm.
- One external shallow spherical dimple at +X, axial midspan z=0.
- Dimple sphere radius 1.5 mm, depth 0.45 mm, minimum remaining radial
  ligament 1.05 mm. The inner pressure boundary is not penetrated.
- Full three-dimensional quadratic-tetrahedral SOLID187 mesh, nominal
  element size 0.75 mm. **This size is not mesh-converged**. End grips are
  simplified, bottom end fixed, top subjected to axial and torsional resultants.
- Material E=210,000 MPa and nu=0.30. Those are illustration parameters,
  not measured for the actual specimen material.

The spherical dimple is selected as a mechanically well-defined one-sided
machinable notch. The literature also reports **notched hollow specimens with
transverse holes** under in-phase and out-of-phase axial–torsion loading,
showing that the general experimental concept is credible. See the articles:

- *Notched fatigue behavior and stress analysis under multiaxial states of
  stress*, International Journal of Fatigue 67 (2014), pp. 2–14,
  DOI 10.1016/j.ijfatigue.2014.01.014.
- M. Firat, *A numerical analysis of combined bending–torsion fatigue of SAE
  notched shaft*, Finite Elements in Analysis and Design 54 (2012), pp.16–27,
  DOI 10.1016/j.finel.2012.01.005.

A through-hole is not interchangeable with an external one-sided dimple:
pressure containment, notch geometry and resulting stress gradients differ.

## Boundary conditions and phase handling

All three controls have one common frequency with independent phases:

```text
p_i(t) = pmean + psin*sin(t) + pcos*cos(t)        [MPa]
F(t)   = Fmean + Fsin*sin(t) + Fcos*cos(t)        [N]
T(t)   = Tmean + Tsin*sin(t) + Tcos*cos(t)        [N mm]
```

Internal pressure is applied to the cylindrical bore surface ONLY. The
external notch remains traction-free. A flag explicitly controls the
axial-pressure thrust convention:

- `CLOSED_END=1` adds `p_i*pi*ri**2` to the top end axial force.
- `CLOSED_END=0` assumes the hydraulic pressure thrust is reacted by a
  separate rig fixture and is not carried by the gauge tube.

The input `F` is the **additional applied axial machine force** in either
case. Applied torque is distributed through tangential top-end nodal forces.
The lower end has a simplified fully fixed annular face. Grip rigidity,
pressure-port effects, end transitions, actual seal geometry, shear-transfer
coupling and loading frequency are not included. They must be developed
before experimental qualification.

### APDL model and unit-load response

Files in `apdl/`:

- `notched_tube.mac` defines the tube, removes the spherical notch using
  `VSBV`, meshes the full 360-degree 3D domain, applies pressure, axial force
  and torsion, and solves three independent linear static load cases.
- `notch_unit_basis.mac` defines the three actuator **unit response** solves:
  1 MPa internal pressure, 1000 N additional axial force, 1000 N mm torque.
- `notch_loads.mac` is an illustrative **harmonic** pressure/force/torque
  program. It is not claimed to be within the user's machine limits.
- `export_notch.mac` writes critical-region element stresses, strains and
  centroid locations with the same CSV format as the existing Phase 6 FE
  postprocessing interface.

Set `BASIS_MODE=1` in `notched_tube.mac` for the *unit response* run. **Do not
interpret the three unit-load results as mean/sine/cosine time coefficients.**
The Python program superposes their real stress and strain responses to
construct physically constrained harmonic histories. Set `BASIS_MODE=0` to
solve an explicit selected harmonic history with the `notch_loads.mac` values.

On a machine with a licensed MAPDL executable, run from an isolated copy of
the folder **after deleting old fe_stress.csv / fe_strain.csv /
fe_coordinates.csv**, since APDL writes in APPEND mode. For example:

```bash
cd apdl
# Set BASIS_MODE=1 manually in notched_tube.mac first.
<mapdl-executable> -b -i notched_tube.mac -o notch_unit.log
```

The code currently includes an editable, simplified end loading concept and
is **not MAPDL-solver-verified** in this environment. Inspect the boolean
volume, inner-bore pressure, end reactions, mesh quality and selection
counts before trusting the export.

### Geometry sweep

Generate separate input folders with different notch depths and radii:

```bash
python examples/generate_notch_geometry_sweep.py \
  --out results/notch_designs \
  --depths 0.25,0.45,0.65 \
  --radii 1.0,1.5,2.0 --mesh-h 0.75 --mode 1
```

The generator also prepares a `smooth_reference/` control model with
`NOTCH_ON=0`, so smooth and notched specimens can be compared at identical
loading coefficients. Run MAPDL in each folder independently. `geometry_sweep.csv` records the
remaining wall ligament. Repeat the best candidate with smaller mesh sizes
near the notch root, such as 0.5 and 0.3 mm, and follow a proper convergence
study; no particular size is guaranteed adequate.

### Postprocessing and pressure-constrained screening

```bash
python -m pip install -e '.[test,plot]'
python examples/notched_tube_search.py \
  --basis-dir results/notch_designs/R1p50_D0p45 \
  --out results/notch_screen \
  --trials 60 --algorithm mce --pressure-max 40 \
  --force-max 10000 --torque-max 40000 --screen-elements 12
```

The example pressure/force/torque caps above are configurable *illustrative
ranges*, NOT verified ratings of the laboratory machine.

The script rejects pressure histories violating
`pmean >= hypot(psin, pcos)` and, when a pressure maximum is specified,
`pmean + hypot(psin, pcos) <= pmax`. Axial and torsional extrema are
likewise bounded. It can screen local von Mises stresses against a chosen
illustrative limit. The best candidate histories are independently
re-evaluated with denser full-plane scanning and the original three SeAn-CP
paths. MCE and longest chord/MCC can be analysed separately. The output
contains generated `notch_loads_rankN.mac`, `screening.json` and 3D plane
landscapes with both maxima and all three paths at their true FS_ext values.

For a direct comparison with a selected **Phase 4 target stress history**:

```bash
python examples/fit_notch_to_phase4.py \
  --basis-dir results/notch_designs/R1p50_D0p45 \
  --out results/notch_target_2213 --case 2213
```

This minimises the harmonic stress-coefficient mismatch at each extracted FE
point while imposing the same nonnegative pressure / force / torque limits.
The tensor-fit distance and actual **recalculated** FS_ext error are both
reported. A low tensor-fit distance does not guarantee a large methodology
error; conversely, a new FE tensor may have a large error even when it is
unlike any of the four previously studied cases.

Finally, re-run the selected harmonic coefficients with `BASIS_MODE=0` and
verify the directly solved FE tensors against superposition. Use
`examples/postprocess_apdl_fs_ext.py` on that **harmonic** run for the final
per-element comparison. Unlike the unit-response outputs, it contains real
mean/sine/cosine FE histories and is accepted by the generic Phase 6 pipeline.

## Required physical/numerical validation

1. Confirm internal pressure acts only on the bore and not on the free notch
   surface; verify closed-end pressure thrust and axial force definitions.
2. Check equilibrium of resultant force/torque and compare a **smooth-tube
   no-notch** reference with the Phase 2 Lamé/Saint-Venant analytical field.
3. Verify mesh convergence, including peak critical-plane FS_ext, not just
   peak von Mises. Smooth the notch and grips if the selected geometry has
   geometric singularities. Identify any nonphysical clamp concentration.
4. Examine subsurface and surface positions as well as the full selected
   notch volume. Locate maximum full FS_ext and independently maximum
   SeAn-CP **over the same element population**.
5. Check fatigue-criticality outside the notch/gauge, failure of grips and
   ligaments, stress homogeneity, notch root manufacturability and whether
   the elastic assumption is satisfied throughout the cycle.
6. Confirm actual machine capacities, pressure minimum and maximum,
   hydraulic phase response, frequency bandwidth and control cross-talk.
7. Interpret a large *numerical* FS_ext discrepancy as a demonstration of the
   restricted critical-plane search, **not** as experimental fatigue-life
   validation until tests and measurements are acquired.

Until a solver run is supplied, **no notch-induced SeAn-CP error or critical
location is known**. This is a computational experiment proposal, not an FE
result.
