# Phase 4. Unconstrained harmonic tensor loading and SeAn-CP limitations

## Scope

The present phase removes the geometry and machine restrictions introduced in
Phases 2 and 3. No hollow specimen, pressure actuator, axial-force limit, or
torque limit is imposed. The aim is to identify **four distinct tensor loading
conditions** that are particularly poorly estimated by the corrected
semi-analytical SeAn-CP search.

The evaluated criterion is **FS_ext only**. We retain the three searching paths
from the original SeAn-CP concept, as corrected and tested in Phase 1. The
reference is a numerical maximum over all critical-plane orientations, not a
separate fatigue criterion.

We deliberately use a *local material point* description. For a given symmetric
3x3 stress tensor, the strain follows isotropic, linear elasticity. This
ensures internal stress-strain consistency without making any claim about
whether a particular specimen or actuator system can realise the stress
history. That question is deferred to Phase 5.

## Stress and strain histories

All stress components vary with a **single frequency**, including independently
phased components:

\[
\boldsymbol\sigma(t)=\boldsymbol\sigma_m+
  \boldsymbol\sigma_s\sin(\omega t)+
  \boldsymbol\sigma_c\cos(\omega t).
\]

\[
\boldsymbol\varepsilon_j=
  \frac{1+\nu}{E}\boldsymbol\sigma_j-
  \frac{\nu}{E}\operatorname{tr}(\boldsymbol\sigma_j)\mathbf I,
\qquad j\in\{m,s,c\}.
\]

The strain tensors are *tensorial* small strains. There is no imposed specimen
coordinate system. Each harmonic history can represent any symmetric elastic
stress-tensor trajectory at one material point. It does **not** follow that
all such trajectories can be obtained using a given set of boundary loads.

Illustrative constants are 
\(E=210\,000\) MPa, \(\nu=0.30\), \(S_y=600\) MPa and \(k_{FS}=0.4\).
A sampled maximum von Mises stress is scaled between 30% and 92% of
\(S_y\) for numerical comparability. This is **not** a proof of elastic
admissibility, nor an experimental feasibility screen. The constants are
not from a particular material's calibrated fatigue model.

## FS_ext and plane-search comparison

For each unit plane normal \(\mathbf n\), we evaluate

\[
FS_{ext}(\mathbf n)=\gamma_a(\mathbf n)
\left(1+k_{FS}\frac{\max(0,\sigma_{n,max}(\mathbf n))}{S_y}\right).
\]

The shear-amplitude definitions are **longest chord**, **MCC** and **MCE**.
For a complete single-harmonic ellipse, MCC and longest chord are identical,
while MCE combines the two ellipse semiaxes. Both the full-plane reference and
each SeAn-CP path use the **same selected amplitude definition**.

The error is defined by

\[
e=100\left(\frac{FS_{ext}^{SeAnCP}}{FS_{ext}^{full}}-1\right)\,\%.
\]

A negative value is an underestimation of the full-plane maximum. It is not
a measured fatigue-life error, which would require an additional calibrated
material relationship.

The coarse screening compares 600 fixed random normals with 49 points on each
of the three semi-analytical path families. Screening errors are **not** used
as final errors. A shortlist is refined by a dense plane grid and local
continuous optimisation, and the four selected cases are verified using
721 points per path, a 181x361 full-plane grid and three independent
differential-evolution runs per case. These numerical checks agree closely,
although they cannot certify global optimality mathematically.

## Sampled tensor families

The search includes proportional tensors, non-proportional diagonal normal
stresses, rotating shear pairs, a planar biaxial tensor subspace, independently
rotated principal stress directions, non-coaxial mean stresses, and fully
mixed 3D tensors. The study uses a fixed random seed, 44713.

A total of **8,000 histories** were tested, with **24,000 amplitude-method
comparisons** (longest chord, MCC, MCE). To avoid selecting four nearly
identical realisations of one family, we report the worst refined example
from each of four distinct tensor families. They are not necessarily the
four largest errors without this diversity constraint.

## Four selected cases

| Case ID | Harmonic tensor family | Amplitude | Full FS_ext | SeAn-CP FS_ext | Error |
|---:|---|---|---:|---:|---:|
| 2213 | Non-proportional diagonal normal stresses | MCE | 0.0010506490 | 0.0006578529 | **-37.386%** |
| 1739 | Biaxial mixed normal and shear stresses | MCE | 0.0031928126 | 0.0027991383 | **-12.330%** |
| 6920 | Independently rotated principal tensors | MCE | 0.0025824683 | 0.0022693055 | **-12.126%** |
| 7348 | Non-coaxial mean and alternating tensors | MCE | 0.0006511027 | 0.0005736924 | **-11.889%** |

The first case is remarkable because its three stress tensors are diagonal,
with no imposed shear stress in that coordinate system. It still generates
non-zero plane shear strain for oblique plane normals, and the full maximum
lies outside the three searched eigenvector circles. Its longest-chord and
MCC errors are also **-37.386%**, so this example does not depend on the MCE
convention.

For reference, the diagonal tensors of case 2213, in MPa, are

\[
\sigma_m=\mathrm{diag}(-40.633,-107.455,34.387),\\
\sigma_s=\mathrm{diag}(255.003,298.694,219.105),\\
\sigma_c=\mathrm{diag}(-21.273,75.331,120.615).
\]

The other three cases exhibit a greater sensitivity to the MCE definition.
The corresponding longest-chord errors are -10.56%, -0.00056% and -0.239%.
These differences indicate that amplitude selection and restricted-plane
search interact. We must not attribute the full MCE discrepancy to
plane-search geometry alone without specifying the amplitude definition.

## 3D comparison figures

Each selected case is plotted as a true 3D critical-plane damage landscape.
The horizontal axes are azimuth \(\varphi\in[0,360^\circ]\) and polar angle
\(\theta\in[0,90^\circ]\); the vertical axis is \(10^3 FS_{ext}\).
Because the plane normal \(\mathbf n\) and \(-\mathbf n\) represent the same
plane, the upper hemisphere contains all distinct orientations, with
seam splitting at 0/360 degrees.

The **white star** is the independently scanned full-plane maximum. The
**gold circle** marks the plane and *actual FS_ext value* found by SeAn-CP.
The three coloured curves represent the strain-range, stress-anchor-0 and
stress-anchor-1 search paths. Curves and markers use their actual damage
values, with no arbitrary shift along the z axis. A separate 2D damage map
is supplied for each case to improve visibility of coincident curves.

### Files and reproducibility

The repository contains `seancp/unconstrained.py`,
`seancp/unconstrained_plots.py`, the Phase 4 example/CLI,
`tests/test_unconstrained.py`, and compact reference CSV/JSON records.
The downloadable numerical archive also includes the complete 24,000-row
screening table, refined shortlist, all diagnostic images and this report.

```bash
python -m pip install -e '.[test,plot]'
python -m pytest -q
python examples/phase4_unconstrained.py --n-cases 8000 --output results/phase4_search
```

The random seed, matrix coefficients and complete material constants are
preserved in `results/phase4_reference/selected_cases.json`. The saved JSON
allows any selected tensor history to be reconstructed independently of the
random generator or proposed experimental setup.

## Limitations and next steps

This is a finite seeded exploration, not a mathematical proof of worst-case
error. It does not quantify uncertainty in the Fatemi-Socie material
parameters. The present results are **damage-parameter discrepancies**, not
fatigue-life discrepancies. No specimen shape, machine capability or
realistic boundary-load decomposition has been assessed.

**Phase 5** will examine which of these local tensor histories can be
reconstructed by realistic boundary conditions or multiaxial test rigs.
If a selected tensor field is unattainable, that should be reported explicitly
rather than changing the Phase 4 benchmark retrospectively.
